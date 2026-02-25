"""
train_grpo.py — GRPO (Group Relative Policy Optimization) 训练脚本
支持 G∈{8,16,32,64} 消融实验、逐层梯度追踪、Batch Size 对齐

用法:
    python train_grpo.py                          # 默认 G=32
    python train_grpo.py --group-size 8           # G=8 消融
    python train_grpo.py --group-size 64 --lr 3e-6
    python train_grpo.py --epochs 2 --save-every 20
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import csv
import gc
from model_utils import load_model_and_tokenizer, collect_per_layer_grad_stats
from env import Arithmetic24Env


class MathDataset(Dataset):
    def __init__(self, data_file, tokenizer, env, max_samples=None):
        self.queries = []
        self.input_nums = []

        # 支持 CSV 和 JSONL 两种格式
        if data_file.endswith('.jsonl'):
            with open(data_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples:
                        break
                    record = json.loads(line.strip())
                    nums = record['nums']
                    self.input_nums.append(nums)
                    prompt = env.get_prompt(nums)
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
                    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
                    self.queries.append(tokens)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return {"query": self.queries[idx], "input_nums": self.input_nums[idx]}


def get_per_token_logps(logits, input_ids):
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, 2, input_ids.unsqueeze(-1)).squeeze(-1)


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO Training")

    # ── 核心消融参数 ──
    parser.add_argument("--group-size", "-G", type=int, default=32,
                        help="组采样大小 G ∈ {8, 16, 32, 64}")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="每步题目数。B_eff = batch_size × G × accum_steps")
    parser.add_argument("--accum-steps", type=int, default=1,
                        help="梯度累积步数 (默认1, 即每步更新)")

    # ── 优化器 ──
    parser.add_argument("--lr", type=float, default=5e-6, help="学习率")
    parser.add_argument("--beta", type=float, default=0.04, help="KL 惩罚系数")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip 范围")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--entropy-coef", type=float, default=0.005, help="Entropy bonus 系数")

    # ── 训练控制 ──
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--ppo-epochs", type=int, default=1, help="每次 rollout 的 PPO 更新轮数")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="生成最大长度 (24点答案通常<80 tokens)")
    parser.add_argument("--save-every", type=int, default=40, help="每 N 个 update 保存一次")
    parser.add_argument("--max-samples", type=int, default=None, help="限制训练样本数")

    # ── 路径 ──
    parser.add_argument("--data-file", type=str, default="data/train.csv", help="训练数据路径")
    parser.add_argument("--sft-path", type=str, default="saved_models/sft_final", help="SFT 预训练权重路径")
    parser.add_argument("--output-dir", type=str, default="saved_models", help="模型保存目录")
    parser.add_argument("--log-dir", type=str, default="logs", help="日志目录")

    # ── 日志控制 ──
    parser.add_argument("--log-layer-grads", action="store_true", help="记录逐 LoRA 层梯度统计")

    # ── 生成参数 ──
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)

    # ── 自适应 KL ──
    parser.add_argument("--adaptive-kl", action="store_true", default=True,
                        help="启用自适应 KL 惩罚 (默认开启)")
    parser.add_argument("--kl-high", type=float, default=6.0, help="KL 上界阈值")
    parser.add_argument("--kl-low", type=float, default=1.0, help="KL 下界阈值")

    return parser.parse_args()


def train(args):
    G = args.group_size
    bs = args.batch_size
    accum = args.accum_steps
    B_eff = bs * G * accum

    print(f"\n{'='*60}")
    print(f"  GRPO 训练配置")
    print(f"{'='*60}")
    print(f"  G (group size)       = {G}")
    print(f"  batch_size           = {bs}")
    print(f"  accum_steps          = {accum}")
    print(f"  B_eff (per update)   = {B_eff}")
    print(f"  lr                   = {args.lr}")
    print(f"  beta (KL)            = {args.beta}")
    print(f"  epochs               = {args.epochs}")
    print(f"  data                 = {args.data_file}")
    print(f"  sft_path             = {args.sft_path}")
    print(f"{'='*60}\n")

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 日志文件 ──
    log_tag = f"grpo_G{G}"
    log_file = open(os.path.join(args.log_dir, f'{log_tag}_metrics.csv'), 'w', newline='')
    csv_writer = csv.writer(log_file)
    csv_writer.writerow([
        "step", "success_rate", "policy_entropy",
        "kl_div", "mean_advantage", "adv_std", "grad_norm", "grad_second_moment", "mean_response_length"
    ])

    response_file = open(os.path.join(args.log_dir, f'{log_tag}_responses.txt'), 'w', encoding='utf-8')
    response_file.write(f"=== GRPO G={G} Training Responses ===\n\n")

    # 逐层梯度日志
    layer_grad_file = None
    if args.log_layer_grads:
        layer_grad_file = open(os.path.join(args.log_dir, f'{log_tag}_layer_grads.jsonl'), 'w')

    # ── 模型加载 ──
    env = Arithmetic24Env()
    sft_path = args.sft_path if os.path.exists(args.sft_path) else None
    model, tokenizer = load_model_and_tokenizer(with_value_head=False, lora_resume_path=sft_path)
    model.is_peft_model = True

    dataset = MathDataset(args.data_file, tokenizer, env, max_samples=args.max_samples)
    # num_workers=0: 数据已在 __init__ 全部预加载到内存，workers 只会增加 IPC 开销
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, collate_fn=lambda x: x)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr
    )

    beta = args.beta
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    step = 0
    update_step = 0
    device = model.device
    optimizer.zero_grad()

    metric_acc = {"succ": 0.0, "adv": 0.0, "adv_std": 0.0, "kl": 0.0, "entropy": 0.0, "resp_len": 0.0}

    for epoch in range(args.epochs):
        print(f"\n── Epoch {epoch+1}/{args.epochs} ──")

        for batch in dataloader:
            model.eval()
            model.gradient_checkpointing_disable()  # 生成阶段无需梯度，关掉加速推理
            rollouts = []

            # ── Batch 生成 ──
            all_q_tensors = []
            all_num_strs = []
            max_q_len = 0

            for item in batch:
                q_tensor = item["query"].to(device, non_blocking=True)
                all_q_tensors.append(q_tensor)
                all_num_strs.append(item["input_nums"])
                max_q_len = max(max_q_len, q_tensor.shape[0])

            padded_q_list = []
            for q_tensor in all_q_tensors:
                pad_len = max_q_len - q_tensor.shape[0]
                if pad_len > 0:
                    padded_q = F.pad(q_tensor, (pad_len, 0), value=tokenizer.pad_token_id)
                else:
                    padded_q = q_tensor
                padded_q_list.append(padded_q.unsqueeze(0).repeat(G, 1))

            huge_q_tensors = torch.cat(padded_q_list, dim=0)

            # 分块生成防 OOM
            with torch.no_grad():
                gen_chunk = max(G * bs, 64)  # 尽量一次性生成整个 batch
                all_outputs = []
                for ci in range(0, huge_q_tensors.shape[0], gen_chunk):
                    chunk = huge_q_tensors[ci:ci + gen_chunk]
                    out = model.generate(chunk, **gen_kwargs)
                    all_outputs.append(out)

                # Pad 到同一长度后拼接
                max_out_len = max(o.shape[1] for o in all_outputs)
                padded_outputs = []
                for o in all_outputs:
                    if o.shape[1] < max_out_len:
                        o = F.pad(o, (0, max_out_len - o.shape[1]), value=tokenizer.pad_token_id)
                    padded_outputs.append(o)
                outputs = torch.cat(padded_outputs, dim=0)

            # ── 拆分 rollout ──
            q_len = huge_q_tensors.shape[1]

            for i, num_str in enumerate(all_num_strs):
                start_idx = i * G
                end_idx = start_idx + G

                group_out = outputs[start_idx:end_idx]
                resp_tensors = group_out[:, q_len:]
                responses = tokenizer.batch_decode(resp_tensors, skip_special_tokens=True)
                
                # 计算实际生成的 token 长度 (排除 padding)
                resp_lens = (resp_tensors != tokenizer.pad_token_id).float().sum(dim=1).mean().item()

                if step % 10 == 0 and i == 0:
                    response_file.write(f"Step {step}:\n{responses[0]}\n{'-'*60}\n")
                    response_file.flush()

                if step == 0 and i == 0:
                    print(f"\n[模型原始输出观察]:\n{responses[0]}\n")

                group_rewards_list, corrects = env.compute_rewards_batch(
                    [num_str] * G, responses
                )

                group_rewards = torch.tensor(group_rewards_list, dtype=torch.float32).to(device, non_blocking=True)
                mean_r = group_rewards.mean()
                std_r = group_rewards.std() + 1e-8
                advantages = (group_rewards - mean_r) / std_r

                input_ids = group_out
                attention_mask = (input_ids != tokenizer.pad_token_id).long()

                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        mini_bs = G  # 云 GPU 全量推理
                        old_log_probs_list, ref_log_probs_list = [], []

                        for mi in range(0, G, mini_bs):
                            mb_ids = input_ids[mi:mi + mini_bs]
                            mb_mask = attention_mask[mi:mi + mini_bs]
                            mb_resp = resp_tensors[mi:mi + mini_bs]
                            mb_loss_mask = (mb_resp != tokenizer.pad_token_id).float()

                            logits = model(mb_ids, attention_mask=mb_mask).logits
                            mb_old_lp = get_per_token_logps(logits[:, q_len-1:-1, :], mb_resp).detach()
                            old_log_probs_list.append(mb_old_lp)
                            del logits

                            with model.disable_adapter():
                                ref_logits = model(mb_ids, attention_mask=mb_mask).logits
                                mb_ref_lp = get_per_token_logps(ref_logits[:, q_len-1:-1, :], mb_resp).detach()
                                mb_ref_lp = mb_ref_lp * mb_loss_mask
                                ref_log_probs_list.append(mb_ref_lp)
                                del ref_logits

                        old_log_probs = torch.cat(old_log_probs_list, dim=0)
                        ref_log_probs = torch.cat(ref_log_probs_list, dim=0)

                rollouts.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "q_len": q_len,
                    "resp_tensors": resp_tensors,
                    "advantages": advantages.detach(),
                    "old_log_probs": old_log_probs,
                    "ref_log_probs": ref_log_probs,
                    "reward_mean": mean_r.item(),
                    "reward_std": group_rewards.std().item(),
                    "success_rate": corrects / G,
                    "mean_resp_len": resp_lens
                })

            # ── 优化阶段 ──
            model.train()
            model.gradient_checkpointing_enable()  # 训练阶段重新启用
            total_entropy = 0
            total_kl = 0

            for _ in range(args.ppo_epochs):
                for r in rollouts:
                    input_ids = r["input_ids"]
                    attention_mask = r["attention_mask"]
                    q_len_r = r["q_len"]
                    resp_tensors_r = r["resp_tensors"]
                    adv = r["advantages"].unsqueeze(1)

                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        mini_bs = input_ids.shape[0]  # 云 GPU 全量推理
                        lp_list = []
                        for mi in range(0, input_ids.shape[0], mini_bs):
                            mb_ids = input_ids[mi:mi + mini_bs]
                            mb_mask = attention_mask[mi:mi + mini_bs]
                            mb_resp = resp_tensors_r[mi:mi + mini_bs]
                            mb_loss_mask = (mb_resp != tokenizer.pad_token_id).float()

                            logits = model(mb_ids, attention_mask=mb_mask).logits
                            mb_lp = get_per_token_logps(logits[:, q_len_r-1:-1, :], mb_resp)
                            mb_lp = mb_lp * mb_loss_mask
                            lp_list.append(mb_lp)
                            del logits

                        log_probs = torch.cat(lp_list, dim=0)

                    loss_mask = (resp_tensors_r != tokenizer.pad_token_id).float()
                    ratio = torch.exp(log_probs - r["old_log_probs"])
                    kl = torch.exp(r["ref_log_probs"] - log_probs) - (r["ref_log_probs"] - log_probs) - 1

                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv
                    policy_loss = -torch.min(surr1, surr2)

                    loss = ((policy_loss + beta * kl) * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1)
                    loss = loss.mean()

                    prob = torch.exp(log_probs)
                    entropy = -(prob * log_probs * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1)

                    loss = loss / accum
                    entropy_bonus = args.entropy_coef * entropy.mean() / accum
                    loss = loss - entropy_bonus
                    loss.backward()

                    total_entropy += entropy.mean().item()

                    seq_kl = (kl * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1)
                    mean_kl_item = seq_kl.mean().item()
                    total_kl += mean_kl_item

                    # 自适应 KL
                    if args.adaptive_kl:
                        if mean_kl_item > args.kl_high:
                            beta = min(beta * 1.5, 0.2)
                        elif mean_kl_item < args.kl_low:
                            beta = max(beta / 1.5, 0.001)

            step += 1

            # ── 累积指标 ──
            metric_acc["succ"] += sum(r["success_rate"] for r in rollouts) / len(rollouts)
            metric_acc["adv"] += sum(r["reward_mean"] for r in rollouts) / len(rollouts)
            metric_acc["adv_std"] += sum(r["reward_std"] for r in rollouts) / len(rollouts)
            metric_acc["kl"] += total_kl
            metric_acc["entropy"] += total_entropy
            metric_acc["resp_len"] += sum(r["mean_resp_len"] for r in rollouts) / len(rollouts)

            rollouts.clear()

            # ── 梯度更新 ──
            if step % accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                second_moment = 0.0
                param_count = 0
                for p in model.parameters():
                    if p.grad is not None:
                        second_moment += (p.grad.data ** 2).mean().item()
                        param_count += 1
                if param_count > 0:
                    second_moment /= param_count

                # 逐层梯度统计
                if args.log_layer_grads and layer_grad_file:
                    layer_stats = collect_per_layer_grad_stats(model)
                    layer_grad_file.write(json.dumps({
                        "update_step": update_step,
                        "layers": layer_stats
                    }, ensure_ascii=False) + '\n')
                    layer_grad_file.flush()

                optimizer.step()
                optimizer.zero_grad()

                avg_succ = metric_acc["succ"] / accum
                avg_adv = metric_acc["adv"] / accum
                avg_adv_std = metric_acc["adv_std"] / accum
                avg_kl = metric_acc["kl"] / accum
                avg_entropy = metric_acc["entropy"] / accum
                avg_resp_len = metric_acc["resp_len"] / accum

                csv_writer.writerow([
                    update_step, avg_succ, avg_entropy, avg_kl,
                    avg_adv, avg_adv_std,
                    grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm,
                    second_moment, avg_resp_len
                ])
                log_file.flush()

                print(f"Update {update_step} (Step {step}) | Succ: {avg_succ:.3f} | "
                      f"R: {avg_adv:.2f} | KL: {avg_kl:.4f} | |g|: {grad_norm:.4f} | β: {beta:.4f}")

                if update_step > 0 and update_step % args.save_every == 0:
                    save_dir = os.path.join(args.output_dir, f"grpo_G{G}_update_{update_step}")
                    model.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
                    print(f"  💾 Model saved → {save_dir}")

                metric_acc = {"succ": 0.0, "adv": 0.0, "adv_std": 0.0, "kl": 0.0, "entropy": 0.0, "resp_len": 0.0}
                update_step += 1

            if step % 50 == 0:
                torch.cuda.empty_cache()

    # ── 保存最终模型 ──
    save_dir = os.path.join(args.output_dir, f"grpo_G{G}_final")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    log_file.close()
    response_file.close()
    if layer_grad_file:
        layer_grad_file.close()

    print(f"\n=== GRPO G={G} 训练完成 ===")
    print(f"  模型: {save_dir}")
    print(f"  指标: {args.log_dir}/{log_tag}_metrics.csv")
    print(f"  B_eff = {B_eff} 样本/更新")


if __name__ == "__main__":
    args = parse_args()
    print("=== GRPO 训练开始 ===")
    train(args)