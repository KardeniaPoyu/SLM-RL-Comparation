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
    parser.add_argument("--mini-batch-size", type=int, default=8, help="PPO mini-batch")
    parser.add_argument("--grad-accum-steps", type=int, default=8,
                        help="梯度累积 (batch/mini_batch)")

    # ── 优化器 ──
    parser.add_argument("--lr", type=float, default=3e-6, help="学习率 (论文建议 3e-6)")
    parser.add_argument("--init-kl-coef", type=float, default=0.04,
                        help="KL 惩罚系数 (对齐 GRPO beta=0.04)")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--target-kl", type=float, default=0.1, help="KL 目标上限")
    parser.add_argument("--ppo-epochs", type=int, default=1, help="PPO 更新轮数 (对齐 GRPO)")

    # ── 训练控制 ──
    parser.add_argument("--max-new-tokens", type=int, default=128, help="生成最大长度 (24点答案通常<80 tokens)")
    parser.add_argument("--save-every", type=int, default=40)
    parser.add_argument("--max-samples", type=int, default=None)

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
    parser.add_argument("--adaptive-kl", action="store_true", default=False,
                        help="启用 TRL 内置自适应 KL (默认关闭以对齐 GRPO)")

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
        "kl_div", "mean_advantage", "adv_std", "grad_norm", "grad_second_moment", "mean_response_length"
    ])

    response_file = open(os.path.join(args.log_dir, 'ppo_responses.txt'), 'w', encoding='utf-8')
    response_file.write("=== PPO Training Responses ===\n\n")

    layer_grad_file = None
    if args.log_layer_grads:
        layer_grad_file = open(os.path.join(args.log_dir, 'ppo_layer_grads.jsonl'), 'w')

    # ── 模型加载 ──
    env = Arithmetic24Env()
    sft_path = args.sft_path if os.path.exists(args.sft_path) else None
    model, tokenizer = load_model_and_tokenizer(
        with_value_head=True,
        lora_resume_path=sft_path,
        gradient_checkpointing=True  # PPO 需要梯度检查点：policy + ref model + value head 同时占用显存
    )
    model.is_peft_model = True

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
        cliprange=args.clip_range
    )

    dataset = MathDataset(args.data_file, tokenizer, env, max_samples=args.max_samples)

    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator
    )

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

    gen_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "max_new_tokens": args.max_new_tokens,
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
            response_tensors = []
            gen_chunk = 16  # PPO 分块生成：节省显存给 training step 阶段的双模型前向
            for i in range(0, len(query_tensors), gen_chunk):
                batch_q = [q.to(ppo_trainer.accelerator.device) for q in query_tensors[i:i + gen_chunk]]
                batch_resp = ppo_trainer.generate(batch_q, return_prompt=False, **gen_kwargs)
                response_tensors.extend(batch_resp)
        
        resp_lens = float(torch.stack([(r != tokenizer.pad_token_id).float().sum() for r in response_tensors]).mean().item())

        responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        if step % 10 == 0:
            response_file.write(f"Update {step}:\n{responses[0]}\n{'-'*60}\n")
            response_file.flush()

        if step == 0:
            print(f"\n[模型原始输出观察]:\n{responses[0]}\n")

        reward_vals, correct_count = compute_rewards_parallel(input_nums, responses)
        rewards = [torch.tensor(r, dtype=torch.float32, device=ppo_trainer.accelerator.device)
                   for r in reward_vals]

        gc.collect()                # 回收 Python 引用，释放 tensor 持有的显存
        torch.cuda.empty_cache()    # 释放 CUDA 缓存，为 training step 腾出空间
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        # ── 将所有 stats 值强制转为 Python float ──

        success_rate = _to_float(correct_count / len(rewards))
        val_loss = _to_float(stats.get("ppo/loss/value", 0.0))
        policy_entropy = _to_float(stats.get("ppo/policy/entropy", 0.0))
        kl = _to_float(stats.get("ppo/policy/approxkl", 0.0))
        returns = _to_float(stats.get("ppo/returns/mean", 0.0))
        vpred = _to_float(stats.get("ppo/val/vpred", 0.0))
        mean_adv = _to_float(returns - vpred)
        adv_std = _to_float(stats.get("ppo/val/error", 0.0))

        total_norm = _to_float(metric_cache["total_norm"])
        second_moment = _to_float(metric_cache["second_moment"])

        csv_writer.writerow([
            step, success_rate, val_loss, policy_entropy, kl,
            mean_adv, adv_std, total_norm, second_moment, resp_lens
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
              f"KL: {kl:.4f} | VLoss: {val_loss:.4f} | |g|: {total_norm:.4f}")
        step += 1

        if step > 0 and step % args.save_every == 0:
            save_dir = os.path.join(args.output_dir, f"ppo_step_{step}")
            ppo_trainer.model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"  💾 Model saved → {save_dir}")


    # ── 保存最终模型 ──
    save_dir = os.path.join(args.output_dir, "ppo_final")
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