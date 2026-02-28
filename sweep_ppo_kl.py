"""
sweep_ppo_kl.py — PPO KL 参数快速扫描脚本
每组参数只跑 5 个 Update，自动汇总 KL 趋势，帮你在 10 分钟内找到稳定配置。

用法:
    python sweep_ppo_kl.py
"""

import os, sys, csv, json, time, contextlib
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
from trl import PPOTrainer, PPOConfig

# ── Monkey patches for PyTorch / TRL compat ──
_orig_getitem = torch.Tensor.__getitem__
def _patched_getitem(self, idx):
    if isinstance(idx, np.ndarray):
        idx = torch.from_numpy(idx).to(self.device)
    return _orig_getitem(self, idx)
torch.Tensor.__getitem__ = _patched_getitem

_orig_tensor = torch.tensor
def _patched_tensor(data, *a, **kw):
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], torch.Tensor) and data[0].dim() == 0:
        data = [d.item() for d in data]
    if isinstance(data, np.integer):
        data = int(data)
    elif isinstance(data, np.floating):
        data = float(data)
    return _orig_tensor(data, *a, **kw)
torch.tensor = _patched_tensor

from model_utils import load_model_and_tokenizer
from env import Arithmetic24Env, compute_rewards_parallel
from train_ppo import MathDataset, collator

# ── 扫描配置 ──
SWEEP_CONFIGS = [
    # (名称, init_kl_coef, adaptive, target_kl, kl_penalty)
    ("A: coef=0.2 adaptive target=1.0",  0.2,  True,  1.0,  "kl"),
    ("B: coef=0.5 adaptive target=2.0",  0.5,  True,  2.0,  "kl"),
    ("C: coef=1.0 adaptive target=3.0",  1.0,  True,  3.0,  "kl"),
    ("D: coef=0.2 static",              0.2,  False, None, "kl"),
    ("E: coef=0.5 static",              0.5,  False, None, "kl"),
    ("F: coef=1.0 static",              1.0,  False, None, "kl"),
]

N_UPDATES = 5  # 每组跑几个 Update
BATCH_SIZE = 16
MINI_BATCH_SIZE = 2
GRAD_ACCUM = 8
LR = 2e-6
PPO_EPOCHS = 4


def _to_float(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().item()
    return float(v)


def run_one_config(name, init_kl_coef, adaptive, target_kl, kl_penalty,
                   sft_path="saved_models/sft_final"):
    """跑单个配置 N_UPDATES 步，返回 KL 序列"""
    print(f"\n{'='*60}")
    print(f"  扫描配置: {name}")
    print(f"  init_kl_coef={init_kl_coef}, adaptive={adaptive}, target_kl={target_kl}, penalty={kl_penalty}")
    print(f"{'='*60}")

    torch.cuda.empty_cache()

    # 加载 Policy
    model, tokenizer = load_model_and_tokenizer(
        with_value_head=True,
        lora_resume_path=sft_path,
        gradient_checkpointing=True
    )
    model.is_peft_model = True

    # 加载 Reference
    ref_base, _ = load_model_and_tokenizer(
        with_value_head=False,
        lora_resume_path=sft_path,
        gradient_checkpointing=False
    )
    for p in ref_base.parameters():
        p.requires_grad = False
    ref_base.eval()

    config_kwargs = dict(
        learning_rate=LR,
        batch_size=BATCH_SIZE,
        mini_batch_size=MINI_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        seed=42,
        ppo_epochs=PPO_EPOCHS,
        init_kl_coef=init_kl_coef,
        adap_kl_ctrl=adaptive,
        cliprange=0.2,
        max_grad_norm=0.5,
        kl_penalty=kl_penalty,
    )
    if target_kl is not None:
        config_kwargs["target_kl"] = target_kl
    config = PPOConfig(**config_kwargs)

    env = Arithmetic24Env()
    dataset = MathDataset("data/train.csv", tokenizer, env, max_samples=None)
    ppo_trainer = PPOTrainer(
        config=config, model=model, ref_model=None,
        tokenizer=tokenizer, dataset=dataset, data_collator=collator
    )

    # 注入 ref_model
    original_forward = ref_base.forward
    def patched_forward(*args, **kwargs):
        out = original_forward(*args, **kwargs)
        dummy = torch.zeros(out.logits.shape[0], out.logits.shape[1],
                            device=out.logits.device, dtype=out.logits.dtype)
        return (out.logits, out.loss, dummy)
    ref_base.forward = patched_forward

    ppo_trainer.ref_model = ref_base
    ppo_trainer.is_peft_model = False
    ppo_trainer.optional_peft_ctx = contextlib.nullcontext

    gen_kwargs = {
        "temperature": 0.7, "top_p": 0.90, "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 512,
    }

    results = []
    dataloader = ppo_trainer.dataloader
    step = 0

    for batch in dataloader:
        if step >= N_UPDATES:
            break

        query_tensors = [q.squeeze(0) for q in batch["input_ids"]]
        with torch.no_grad():
            response_tensors = ppo_trainer.generate(
                query_tensors, batch_size=4, **gen_kwargs
            )

        responses = tokenizer.batch_decode(
            [r[len(q):] for q, r in zip(query_tensors, response_tensors)],
            skip_special_tokens=True
        )
        input_nums_list = batch["input_nums"]
        all_rewards, correct_count = [], 0
        for nums, resp in zip(input_nums_list, responses):
            reward_list, corrects = compute_rewards_parallel([nums], [resp])
            all_rewards.append(reward_list[0])
            correct_count += corrects

        rewards = [torch.tensor(r, dtype=torch.float32) for r in all_rewards]
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        kl = _to_float(stats.get("objective/kl", 0.0))
        kl_coef = _to_float(stats.get("objective/kl_coef", 0.0))
        succ = correct_count / len(rewards)
        r_mean = np.mean(all_rewards)

        results.append({
            "step": step, "kl": kl, "kl_coef": kl_coef,
            "succ": succ, "r": r_mean
        })
        print(f"  Update {step} | Succ: {succ:.3f} | R: {r_mean:.2f} | KL: {kl:.4f} | kl_coef: {kl_coef:.4f}")
        step += 1

    # 清理 GPU
    del ppo_trainer, model, ref_base
    torch.cuda.empty_cache()

    return results


def main():
    print("=" * 60)
    print("  PPO KL 参数快速扫描")
    print(f"  每组跑 {N_UPDATES} 个 Update")
    print("=" * 60)

    all_results = {}
    for name, coef, adaptive, target, penalty in SWEEP_CONFIGS:
        try:
            results = run_one_config(name, coef, adaptive, target, penalty)
            all_results[name] = results
        except Exception as e:
            print(f"  ❌ 配置 {name} 失败: {e}")
            all_results[name] = [{"error": str(e)}]
            torch.cuda.empty_cache()

    # ── 汇总报告 ──
    print("\n" + "=" * 70)
    print("  扫描结果汇总")
    print("=" * 70)
    print(f"{'配置名称':<35} {'KL最终':>8} {'KL趋势':>10} {'Succ均值':>8} {'推荐':>4}")
    print("-" * 70)

    for name, results in all_results.items():
        if "error" in results[0]:
            print(f"{name:<35} {'ERROR':>8} {'':>10} {'':>8} {'❌':>4}")
            continue

        kls = [r["kl"] for r in results]
        succs = [r["succ"] for r in results]
        final_kl = kls[-1] if kls else 0
        mean_succ = np.mean(succs) if succs else 0

        # 判断趋势
        if len(kls) >= 3:
            slope = (kls[-1] - kls[0]) / max(len(kls) - 1, 1)
            if abs(final_kl) < 3.0 and slope < 1.0:
                trend = "✅ 稳定"
                recommend = "✅"
            elif final_kl < 5.0:
                trend = "⚠️ 缓升"
                recommend = "⚠️"
            else:
                trend = "❌ 发散"
                recommend = "❌"
        else:
            trend = "?"
            recommend = "?"

        print(f"{name:<35} {final_kl:>8.4f} {trend:>10} {mean_succ:>8.3f} {recommend:>4}")

    # 保存详细结果
    with open("logs/sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n详细结果已保存到 logs/sweep_results.json")


if __name__ == "__main__":
    main()
