"""
train_ppo.py — PPO (Proximal Policy Optimization) 训练脚本
使用 TRL PPOTrainer + Value Head (Critic)
参数与 GRPO 对齐，确保公平对比

用法:
    python train_ppo.py                          # 默认配置
    python train_ppo.py --lr 3e-6 --batch-size 128
    python train_ppo.py --epochs 2
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import torch
import numpy as np
import csv
import gc

# ── PyTorch 2.8 + TRL 0.9.6 全面兼容补丁 ──
# 问题1: TRL 用 numpy.int64 数组索引 tensor → PyTorch 2.8 不再隐式转换
_orig_tensor_getitem = torch.Tensor.__getitem__
def _numpy_compat_getitem(self, indices):
    if type(indices).__module__ == 'numpy' or type(indices).__name__ == 'ndarray':
        if hasattr(indices, 'tolist'):
            return _orig_tensor_getitem(self, indices.tolist())
        elif hasattr(indices, 'item'):
            return _orig_tensor_getitem(self, indices.item())
    
    if isinstance(indices, tuple):
        new_indices = []
        changed = False
        for idx in indices:
            if type(idx).__module__ == 'numpy' or type(idx).__name__ == 'ndarray':
                if hasattr(idx, 'tolist'):
                    new_indices.append(idx.tolist())
                elif hasattr(idx, 'item'):
                    new_indices.append(idx.item())
                else:
                    new_indices.append(idx)
                changed = True
            else:
                new_indices.append(idx)
        if changed:
            return _orig_tensor_getitem(self, tuple(new_indices))
            
    return _orig_tensor_getitem(self, indices)
torch.Tensor.__getitem__ = _numpy_compat_getitem

# 问题2: TRL 用 torch.tensor(list_of_0d_tensors) → PyTorch 2.8 对 0-d tensor 调 len() 崩溃
_orig_torch_tensor = torch.tensor
def _compat_torch_tensor(data, *args, **kwargs):
    if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], torch.Tensor):
        device = kwargs.pop('device', None)
        dtype = kwargs.pop('dtype', None)
        stacked = torch.stack([d.detach().cpu() for d in data])
        if dtype is not None:
            stacked = stacked.to(dtype=dtype)
        if device is not None:
            stacked = stacked.to(device=device)
        return stacked
    return _orig_torch_tensor(data, *args, **kwargs)
torch.tensor = _compat_torch_tensor

from torch.utils.data import Dataset
from trl import PPOTrainer, PPOConfig
from model_utils import load_model_and_tokenizer, collect_per_layer_grad_stats
from env import Arithmetic24Env, compute_rewards_parallel


class MathDataset(Dataset):
    def __init__(self, data_file, tokenizer, env, max_samples=None):
        self.queries = []
        self.prompts = []
        self.input_nums = []

        if data_file.endswith('.jsonl'):
            with open(data_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples:
                        break
                    record = json.loads(line.strip())
                    nums = record['nums']
                    self.input_nums.append(nums)
                    prompt = env.get_prompt(nums)
                    self.prompts.append(prompt)
                    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
                    self.queries.append(tokens)
        else:
            with open(data_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if max_samples and i >= max_samples:
                        break
                    nums = row['nums']
                    self.input_nums.append(nums)
                    prompt = env.get_prompt(nums)
                    self.prompts.append(prompt)
                    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
                    self.queries.append(tokens)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return {
            "query": self.queries[idx],
            "prompt": self.prompts[idx],
            "input_nums": self.input_nums[idx]
        }


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def parse_args():
    parser = argparse.ArgumentParser(description="PPO Training")

    # ── 对齐参数 ──
    parser.add_argument("--batch-size", type=int, default=32,
                        help="PPO batch size (应等于 GRPO 的 bs×G×accum)")
    parser.add_argument("--mini-batch-size", type=int, default=4, help="PPO mini-batch")
    parser.add_argument("--grad-accum-steps", type=int, default=8,
                        help="梯度累积 (batch/mini_batch)")

    # ── 优化器 ──
    parser.add_argument("--lr", type=float, default=5e-7, help="学习率 (保守值防止 KL 爆炸)")
    parser.add_argument("--init-kl-coef", type=float, default=0.2,
                        help="KL 惩罚系数 (abs 模式下需较大以抑制 KL 爆炸)")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--target-kl", type=float, default=2.0, help="自适应 KL 目标值")
    parser.add_argument("--ppo-epochs", type=int, default=1, help="PPO 更新轮数 (对齐 GRPO)")

    # ── 训练控制 ──
    parser.add_argument("--max-new-tokens", type=int, default=128, help="生成最大长度 (24点答案通常<80 tokens)")
    parser.add_argument("--save-every", type=int, default=40)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=200, help="最多更新的 step 数量，到达则停止训练并保存模型")

    # ── 路径 ──
    parser.add_argument("--data-file", type=str, default="data/train.csv")
    parser.add_argument("--sft-path", type=str, default="saved_models/sft_final")
    parser.add_argument("--output-dir", type=str, default="saved_models")
    parser.add_argument("--log-dir", type=str, default="logs")

    # ── 日志 ──
    parser.add_argument("--log-layer-grads", action="store_true", help="记录逐层梯度统计")

    # ── 生成参数 ──
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)

    # ── 自适应 KL ──
    parser.add_argument("--adaptive-kl", action="store_true", default=True,
                        help="启用 TRL 自适应 KL (防止 KL 爆炸)")

    return parser.parse_args()


def train(args):
    B_eff = args.batch_size  # PPO 的 B_eff 就是 batch_size (TRL 内部处理)

    print(f"\n{'='*60}")
    print(f"  PPO 训练配置")
    print(f"{'='*60}")
    print(f"  batch_size           = {args.batch_size}")
    print(f"  mini_batch_size      = {args.mini_batch_size}")
    print(f"  grad_accum_steps     = {args.grad_accum_steps}")
    print(f"  B_eff (per update)   = {B_eff}")
    print(f"  lr                   = {args.lr}")
    print(f"  init_kl_coef         = {args.init_kl_coef}")
    print(f"  ppo_epochs           = {args.ppo_epochs}")
    print(f"  data                 = {args.data_file}")
    print(f"  sft_path             = {args.sft_path}")
    print(f"{'='*60}\n")

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 日志文件 ──
    log_file = open(os.path.join(args.log_dir, 'ppo_metrics.csv'), 'w', newline='')
    csv_writer = csv.writer(log_file)
    csv_writer.writerow([
        "step", "success_rate", "value_loss", "policy_entropy",
        "kl_ref", "approxkl", "mean_advantage", "adv_std", "grad_norm", "grad_second_moment", "mean_response_length",
        "vram_allocated_gb", "vram_peak_gb", "vram_reserved_gb"
    ])

    response_file = open(os.path.join(args.log_dir, 'ppo_responses.txt'), 'w', encoding='utf-8')
    response_file.write("=== PPO Training Responses ===\n\n")

    layer_grad_file = None
    if args.log_layer_grads:
        layer_grad_file = open(os.path.join(args.log_dir, 'ppo_layer_grads.jsonl'), 'w')

    # ── 模型加载 ──
    env = Arithmetic24Env()
    
    sft_path = args.sft_path
    if sft_path and not os.path.exists(sft_path):
        print(f"⚠️  WARNING: SFT path '{sft_path}' does NOT exist! Falling back to fresh base model + LoRA.")
        sft_path = None
    elif sft_path:
        print(f"✅ Found SFT checkpont path: {sft_path}")

    # 1. Policy model (trainable): base 8-bit + LoRA + ValueHead
    print("\n[1/2] Loading policy model (with ValueHead)...")
    model, tokenizer = load_model_and_tokenizer(
        with_value_head=True,
        lora_resume_path=sft_path,
        gradient_checkpointing=True
    )
    model.is_peft_model = True  # ValueHead.forward() 需要此属性 (LoRA 会跳过 PREFIX_TUNING 分支)

    # 2. Reference model (frozen): base 8-bit + LoRA, 无 ValueHead
    print("\n[2/2] Loading reference model (no ValueHead, frozen)...")
    ref_base, _ = load_model_and_tokenizer(
        with_value_head=False,
        lora_resume_path=sft_path,
        gradient_checkpointing=False  # ref 不训练
    )
    for p in ref_base.parameters():
        p.requires_grad = False
    ref_base.eval()

    n_policy_train = sum(p.requires_grad for p in model.parameters())
    n_ref_train = sum(p.requires_grad for p in ref_base.parameters())
    print(f"  Policy model:  {n_policy_train} trainable params (with ValueHead)")
    print(f"  Ref model:     {n_ref_train} trainable params (should be 0, no ValueHead)")
    print(f"  model is ref:  {model is ref_base}  (should be False)")

    config = PPOConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        target_kl=args.target_kl,
        seed=42,
        ppo_epochs=args.ppo_epochs,
        init_kl_coef=args.init_kl_coef,
        adap_kl_ctrl=args.adaptive_kl,
        cliprange=args.clip_range,
        max_grad_norm=0.5,             # 强制梯度裁剪，防止梯度爆炸
        kl_penalty="abs"  # 用 |logp - ref_logp| 防止负 KL 被利用
    )

    dataset = MathDataset(args.data_file, tokenizer, env, max_samples=args.max_samples)

    # ── 初始化 PPOTrainer（先不传 ref_model，之后注入以绕过 init 检查）──
    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=None,   # 先不传，初始化后注入
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator
    )

    # ── 注入无 ValueHead 的 ref_model (保持 AutoModelForCausalLM 类型) ──
    # TRL batched_forward_pass 会解包 3 个值: ref_logits, _, _ = self.ref_model(...)
    # 而普通的 CausalLM 返回 CausalLMOutputWithPast (通常解包出 2 个值)。
    # 我们用补丁直接修改 forward，使得类型原汁原味，又能通过 TRL 的元组解包：
    original_forward = ref_base.forward
    def patched_forward(*args, **kwargs):
        out = original_forward(*args, **kwargs)
        # 用一组 dummy 值凑足 TRL 期望的 3 元组 (logits, loss, values)
        dummy_values = torch.zeros(
            out.logits.shape[0], out.logits.shape[1],
            device=out.logits.device, dtype=out.logits.dtype
        )
        return (out.logits, out.loss, dummy_values)
    
    ref_base.forward = patched_forward

    import contextlib
    ppo_trainer.ref_model = ref_base                     # 直接就是 PeftModel/AutoModel
    ppo_trainer.is_peft_model = False                    # 彻底关闭 disable_adapter 机制
    ppo_trainer.optional_peft_ctx = contextlib.nullcontext
    print("  ✅ ref_model injected (AutoModelForCausalLM, frozen)")

    # ── 梯度拦截器 ──
    metric_cache = {"second_moment": 0.0, "total_norm": 0.0, "layer_stats": {}}

    if hasattr(ppo_trainer, "optimizer"):
        original_step = ppo_trainer.optimizer.step

        def hooked_optimizer_step(*args_inner, **kwargs_inner):
            # 强制执行一次独立的梯度裁剪（防止 TRL 设置失效导致 KL 爆炸）
            torch.nn.utils.clip_grad_norm_(ppo_trainer.model.parameters(), 0.5)

            sm = 0.0
            tn = 0.0
            pc = 0
            for p in ppo_trainer.model.parameters():
                if p.grad is not None:
                    sm += (p.grad.data ** 2).mean().item()
                    tn += p.grad.data.norm(2).item() ** 2
                    pc += 1
            if pc > 0:
                metric_cache["second_moment"] = sm / pc
                metric_cache["total_norm"] = tn ** 0.5

            # 逐层梯度
            if args.log_layer_grads:
                metric_cache["layer_stats"] = collect_per_layer_grad_stats(ppo_trainer.model)

            return original_step(*args_inner, **kwargs_inner)

        ppo_trainer.optimizer.step = hooked_optimizer_step

    gen_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,  # 显式开启缓存，加速生成
    }

    # ── 安全转换函数：防止 numpy scalar 的 __str__ 崩溃 ──
    def _to_float(v):
        """将 numpy scalar / torch tensor / 任意数值安全转为 Python float"""
        if isinstance(v, torch.Tensor):
            return float(v.detach().cpu().item())
        if hasattr(v, 'item'):  # numpy scalar
            return float(v.item())
        return float(v)

    step = 0
    for epoch, batch in enumerate(ppo_trainer.dataloader):
        query_tensors = batch["query"]
        prompts = batch["prompt"]
        input_nums = batch["input_nums"]

        with torch.no_grad():
            ppo_trainer.model.eval()  # 生成阶段必须使用 eval 模式，避免产生 gradient checkpointing 警告，并停用 dropout
            inner_m = ppo_trainer.model.pretrained_model if hasattr(ppo_trainer.model, "pretrained_model") else ppo_trainer.model
            inner_m.gradient_checkpointing_disable()
            inner_m.config.use_cache = True
            
            response_tensors = []
            gen_chunk = 4  # PPO 分块生成 (调小至 4 防止 7B OOM)
            for i in range(0, len(query_tensors), gen_chunk):
                batch_q = [q.to(ppo_trainer.accelerator.device) for q in query_tensors[i:i + gen_chunk]]
                batch_resp = ppo_trainer.generate(batch_q, return_prompt=False, **gen_kwargs)
                response_tensors.extend(batch_resp)
            
            inner_m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            inner_m.config.use_cache = False
            ppo_trainer.model.train()  # 恢复训练模式
        
        resp_lens = float(torch.stack([(r != tokenizer.pad_token_id).float().sum() for r in response_tensors]).mean().item())

        responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        if step % 10 == 0:
            response_file.write(f"Update {step}:\n{responses[0]}\n{'-'*60}\n")
            response_file.flush()

        if step == 0:
            print(f"\n[模型原始输出观察]:\n{responses[0]}\n")

        reward_vals, correct_count = compute_rewards_parallel(input_nums, responses)
        rewards = [torch.tensor(r * 0.05, dtype=torch.float32, device=ppo_trainer.accelerator.device)
                   for r in reward_vals]  # reward scaling: 缩小 reward 幅度防止梯度过大

        gc.collect()                # 回收 Python 引用，释放 tensor 持有的显存
        torch.cuda.empty_cache()    # 释放 CUDA 缓存，为 training step 腾出空间

        # ── step-0 诊断：ref_model 结构 ──
        if step == 0:
            print(f"\n{'='*60}")
            print(f"  [DEBUG] PPO ref_model 诊断")
            print(f"{'='*60}")
            print(f"  id(model)     = {id(ppo_trainer.model)}")
            print(f"  id(ref_model) = {id(ppo_trainer.ref_model)}")
            print(f"  model is ref  = {ppo_trainer.model is ppo_trainer.ref_model}")
            print(f"  is_peft_model = {ppo_trainer.is_peft_model}")
            print(f"  ref_model type= {type(ppo_trainer.ref_model).__name__}")
            print(f"  kl_penalty    = {ppo_trainer.config.kl_penalty}")
            print(f"  kl_coef       = {ppo_trainer.kl_ctl.value}")
            
            # 由于去掉了 _CausalLMRefWrapper，现在 ref_model 就是 ref_base (PeftModel)
            n_grad = sum(p.requires_grad for p in ppo_trainer.ref_model.parameters())
            n_total = sum(1 for _ in ppo_trainer.ref_model.parameters())
            print(f"  ref params    = {n_total} total, {n_grad} trainable (should be 0)")
            print(f"  ref has ValueHead = {hasattr(ppo_trainer.ref_model, 'v_head')}")
            print(f"{'='*60}\n")

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        # ── step-0 诊断：stats keys & KL 值 ──
        if step == 0:
            print(f"\n[DEBUG] stats keys: {sorted(stats.keys())}")
            print(f"[DEBUG] objective/kl       = {_to_float(stats.get('objective/kl', 0.0)):.6f}")
            print(f"[DEBUG] ppo/policy/approxkl = {_to_float(stats.get('ppo/policy/approxkl', 0.0)):.6f}")
            print(f"[DEBUG] ppo/policy/policykl = {_to_float(stats.get('ppo/policy/policykl', 0.0)):.6f}")
            print(f"[DEBUG] objective/kl_coef   = {_to_float(stats.get('objective/kl_coef', 0.0)):.6f}")

        # ── 将所有 stats 值强制转为 Python float ──

        success_rate = _to_float(correct_count / len(rewards))
        val_loss = _to_float(stats.get("ppo/loss/value", 0.0))
        policy_entropy = _to_float(stats.get("ppo/policy/entropy", 0.0))
        kl_ref = _to_float(stats.get("objective/kl", 0.0))      # policy vs reference KL (真正的 KL)
        approxkl = _to_float(stats.get("ppo/policy/approxkl", 0.0))  # PPO ratio-KL (当前 vs rollout)
        returns = _to_float(stats.get("ppo/returns/mean", 0.0))
        vpred = _to_float(stats.get("ppo/val/vpred", 0.0))
        mean_adv = _to_float(returns - vpred)
        adv_std = _to_float(stats.get("ppo/val/error", 0.0))

        total_norm = _to_float(metric_cache["total_norm"])
        second_moment = _to_float(metric_cache["second_moment"])

        # ── VRAM 监控 ──
        vram_alloc = torch.cuda.memory_allocated() / 1e9
        vram_peak = torch.cuda.max_memory_allocated() / 1e9
        vram_reserved = torch.cuda.memory_reserved() / 1e9
        torch.cuda.reset_peak_memory_stats()  # 重置峰值以追踪每步

        csv_writer.writerow([
            step, success_rate, val_loss, policy_entropy, kl_ref, approxkl,
            mean_adv, adv_std, total_norm, second_moment, resp_lens,
            f"{vram_alloc:.2f}", f"{vram_peak:.2f}", f"{vram_reserved:.2f}"
        ])
        log_file.flush()

        # 逐层梯度日志
        if args.log_layer_grads and layer_grad_file and metric_cache["layer_stats"]:
            layer_grad_file.write(json.dumps({
                "update_step": step,
                "layers": metric_cache["layer_stats"]
            }, ensure_ascii=False) + '\n')
            layer_grad_file.flush()

        mean_reward = float(torch.stack(rewards).mean().item())
        print(f"Update {step} | Succ: {success_rate:.3f} | R: {mean_reward:.2f} | "
              f"KL(ref): {kl_ref:.4f} | VLoss: {val_loss:.4f} | |g|: {total_norm:.4f} | "
              f"VRAM: {vram_alloc:.1f}/{vram_peak:.1f} GB")
        step += 1

        if step > 0 and step % args.save_every == 0:
            save_dir = os.path.join(args.output_dir, f"ppo_step_{step}")
            os.makedirs(save_dir, exist_ok=True)
            ppo_trainer.model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"  💾 Model saved → {save_dir}")

        if args.max_steps and step >= args.max_steps:
            print(f"\n[!] 达到最大训练步数 --max-steps {args.max_steps}，提前终止。")
            break


    # ── 保存最终模型 ──
    save_dir = os.path.join(args.output_dir, "ppo_final")
    os.makedirs(save_dir, exist_ok=True)
    ppo_trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    log_file.close()
    response_file.close()
    if layer_grad_file:
        layer_grad_file.close()

    print(f"\n=== PPO 训练完成 ===")
    print(f"  模型: {save_dir}")
    print(f"  指标: {args.log_dir}/ppo_metrics.csv")


if __name__ == "__main__":
    args = parse_args()
    print("=== PPO 训练开始 ===")
    train(args)