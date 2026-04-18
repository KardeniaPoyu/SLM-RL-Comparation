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
    parser.add_argument("--batch-size", type=int, default=64,
                        help="PPO batch size (应等于 GRPO 的 bs×G×accum)")
    parser.add_argument("--mini-batch-size", type=int, default=4, help="PPO mini-batch")
    parser.add_argument("--grad-accum-steps", type=int, default=16,
                        help="梯度累积 (batch/mini_batch)")

    # ── 优化器 ──
    parser.add_argument("--lr", type=float, default=2e-6, help="学习率 (适当提高加快收敛)")
    parser.add_argument("--init-kl-coef", type=float, default=0.05,
                        help="KL 惩罚系数 (适度放开以供探索)")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--target-kl", type=float, default=1.0, help="自适应 KL 目标值 (适度放开)")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="PPO 更新轮数 (提高样本利用率)")

    # ── 训练控制 ──
    parser.add_argument("--max-new-tokens", type=int, default=512, help="生成最大长度 (需容纳 Long-CoT 的长思考过程)")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HuggingFace 模型名或本地路径")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=200, help="最多更新的 step 数量，到达则停止训练并保存模型")

    # ── 路径 ──
    parser.add_argument("--data-file", type=str, default="data/train.csv")
    parser.add_argument("--sft-path", type=str, default="saved_models/sft_final")
    parser.add_argument("--resume-step", type=int, default=0, help="从指定的 step 继续训练日志和步数统计")
    parser.add_argument("--output-dir", type=str, default="saved_models")
    parser.add_argument("--log-dir", type=str, default="logs")

    # ── 日志 ──
    parser.add_argument("--log-layer-grads", action="store_true", help="记录逐层梯度统计")

    # ── 生成参数 ──
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.90)

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
    mode = 'a' if args.resume_step > 0 else 'w'
    log_file = open(os.path.join(args.log_dir, 'ppo_metrics.csv'), mode, newline='')
    csv_writer = csv.writer(log_file)
    if args.resume_step == 0:
        csv_writer.writerow([
            "step", "success_rate", "value_loss", "policy_entropy",
            "kl_ref", "approxkl", "mean_advantage", "adv_std", "grad_norm", "grad_second_moment", "mean_response_length",
            "vram_allocated_gb", "vram_peak_gb", "vram_reserved_gb"
        ])

    response_file = open(os.path.join(args.log_dir, 'ppo_responses.txt'), mode, encoding='utf-8')
    if args.resume_step == 0:
        response_file.write("=== PPO Training Responses ===\n\n")
    else:
        response_file.write(f"\n=== Resumed PPO Training from step {args.resume_step} ===\n\n")

    layer_grad_file = None
    if args.log_layer_grads:
        layer_grad_file = open(os.path.join(args.log_dir, 'ppo_layer_grads.jsonl'), mode)

    # ── 模型加载 ──
    env = Arithmetic24Env()
    
    sft_path = args.sft_path
    if sft_path and not os.path.exists(sft_path):
        print(f"⚠️  WARNING: SFT path '{sft_path}' does NOT exist! Falling back to fresh base model + LoRA.")
        sft_path = None
    elif sft_path:
        print(f"✅ Found SFT checkpont path: {sft_path}")

    # ── 单模型加载：Policy 和 Reference 共享同一个 4-bit 基座 ──
    # 计算 Reference Logits 时临时调用 disable_adapter() 关闭 LoRA，
    # 数学等价于加载独立的冻结 SFT 模型，但节省约 6GB 显存。
    print("\n[1/1] Loading policy model (with ValueHead, shared base for ref)...")
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        lora_resume_path=sft_path,
        with_value_head=True,
        gradient_checkpointing=True
    )
    model.is_peft_model = True

    n_policy_train = sum(p.requires_grad for p in model.parameters())
    print(f"  Policy model:  {n_policy_train} trainable params (with ValueHead)")
    print(f"  Reference: shares same base, LoRA disabled at inference time")

    config = PPOConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        target_kl=args.target_kl,  # 使用命令行参数，不再硬编码
        seed=42,
        ppo_epochs=args.ppo_epochs,
        init_kl_coef=args.init_kl_coef,
        adap_kl_ctrl=args.adaptive_kl,
        cliprange=args.clip_range,
        max_grad_norm=0.5,
        kl_penalty="kl",  # 恢复默认的 KL 惩罚，不要用 abs，以避免数值失控
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

    # ── 使用 disable_adapter 模式作为 ref_model ──
    # 构建一个轻量 wrapper：forward 时临时关闭 LoRA，让基座权重直接参与计算
    # 以此实现「Reference = 冻结 SFT 基座」而无需再加载一份权重
    import contextlib

    class DisabledAdapterRef(torch.nn.Module):
        """Wrap policy's pretrained_model, disable LoRA during forward for KL reference."""
        def __init__(self, peft_model):
            super().__init__()
            # peft_model 是 PeftModel（ValueHead 的内层）
            self._peft = peft_model

        def forward(self, *args, **kwargs):
            with self._peft.disable_adapter():
                out = self._peft(*args, **kwargs)
            # TRL batched_forward_pass 解包: logits, _, values = ref_model(...)
            dummy_values = torch.zeros(
                out.logits.shape[0], out.logits.shape[1],
                device=out.logits.device, dtype=out.logits.dtype
            )
            return (out.logits, out.loss, dummy_values)

        def parameters(self, recurse=True):
            # 让 TRL 认为 ref_model 没有可训练参数
            return iter([])

    ref_wrapper = DisabledAdapterRef(model.pretrained_model)
    ppo_trainer.ref_model = ref_wrapper
    ppo_trainer.is_peft_model = False   # 关闭 TRL 内部的 disable_adapter 重复调用
    ppo_trainer.optional_peft_ctx = contextlib.nullcontext
    print("  ✅ ref_model = DisabledAdapterRef (shared base, zero extra VRAM)")

    # ── 梯度拦截器 ──
    metric_cache = {"second_moment": 0.0, "total_norm": 0.0, "layer_stats": {}}

    if hasattr(ppo_trainer, "optimizer"):
        original_step = ppo_trainer.optimizer.step

        def hooked_optimizer_step(*args_inner, **kwargs_inner):
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

    # ── 生成停止准则：加入 </think> 的 token id 防止无尽循环 ──
    think_close_ids = tokenizer.encode("</think>", add_special_tokens=False)
    eos_ids = [tokenizer.eos_token_id] + think_close_ids

    gen_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": eos_ids,
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

    step = args.resume_step
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
            gen_chunk = 2  # 4090 24GB generation
            for i in range(0, len(query_tensors), gen_chunk):
                batch_q = [q.to(ppo_trainer.accelerator.device) for q in query_tensors[i:i + gen_chunk]]
                # 关闭 return_prompt 防止 prompt 污染模型回答，但需要 env 适应
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
        
        # Reward Normalization (类 GRPO 的 Z-score 处理，或适当放缩)
        # 直接使用任务给出的 RLVR reward（TRL 默认会对 advantage 进行归一化）
        rewards = [torch.tensor(r, dtype=torch.float32, device=ppo_trainer.accelerator.device)
                   for r in reward_vals]

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