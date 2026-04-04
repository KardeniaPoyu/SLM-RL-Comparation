"""
train_grpo.py — GRPO (Group Relative Policy Optimization) 训练脚本
支持 G∈{8,16,32,64} 消融实验、逐层梯度追踪、Batch Size 对齐

论文改进:
  1. Length-Aware Reward Normalization (--length-norm)
     对 advantage 乘以 mean_len / resp_len 因子，消除长度偏差
  2. Dual-Phase Reward Schedule (--reward-schedule dual)
     训练前期用 dense continuous reward，后期自动切换为 sparse binary reward

用法:
    python train_grpo.py                          # 默认 G=32
    python train_grpo.py --group-size 8           # G=8 消融
    python train_grpo.py --group-size 64 --lr 3e-6
    python train_grpo.py --epochs 2 --save-every 20
    python train_grpo.py --length-norm --reward-schedule dual  # 启用两项改进
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
from env import Arithmetic24Env, compute_rewards_parallel


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
    parser.add_argument("--lr", type=float, default=2e-6, help="学习率 (提高)")
    parser.add_argument("--beta", type=float, default=0.01, help="KL 惩罚系数 (初始适度放开)")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip 范围")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="梯度裁剪，缩紧以防止爆炸")
    parser.add_argument("--entropy-coef", type=float, default=0.005, help="Entropy bonus 系数")

    # ── 训练控制 ──
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="每次 rollout 的 PPO 更新轮数 (提高样本利用率)")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="生成最大长度 (需容纳 Long-CoT 的长思考过程)")
    parser.add_argument("--save-every", type=int, default=10, help="每 N 个 update 保存一次")
    parser.add_argument("--max-samples", type=int, default=None, help="限制训练样本数")
    parser.add_argument("--max-steps", type=int, default=200, help="最多更新的 update step 数量，到达则停止训练并保存模型")

    # ── 路径 ──
    parser.add_argument("--data-file", type=str, default="data/train.csv", help="训练数据路径")
    parser.add_argument("--sft-path", type=str, default="saved_models/sft_final", help="SFT 预训练权重路径")
    parser.add_argument("--resume-step", type=int, default=0, help="从指定的 update_step 继续训练日志和步数统计")
    parser.add_argument("--output-dir", type=str, default="saved_models", help="模型保存目录")
    parser.add_argument("--log-dir", type=str, default="logs", help="日志目录")

    # ── 日志控制 ──
    parser.add_argument("--log-layer-grads", action="store_true", help="记录逐 LoRA 层梯度统计")

    # ── 生成参数 ──
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.90)

    # ── 自适应 KL ──
    parser.add_argument("--adaptive-kl", action="store_true", default=True,
                        help="启用自适应 KL 惩罚 (默认开启)")
    parser.add_argument("--kl-high", type=float, default=4.0, help="KL 上界阈值 (适度放开以允许探索)")
    parser.add_argument("--kl-low", type=float, default=1.0, help="KL 下界阈值")

    # ── 论文改进 1: Length-Aware Reward Normalization ──
    parser.add_argument("--length-norm", action="store_true", default=False,
                        help="启用长度感知归一化: advantage *= mean_len / resp_len")

    # ── 论文改进 2: Dual-Phase Reward Schedule ──
    parser.add_argument("--reward-schedule", type=str, default="fixed",
                        choices=["fixed", "dual"],
                        help="奖励调度策略。fixed=始终用 dense reward; dual=前期 dense 后期 binary")
    parser.add_argument("--phase-switch-threshold", type=float, default=0.10,
                        help="Dual-Phase: EMA 成功率超过此阈值时切换为 binary reward (默认 0.10)")
    parser.add_argument("--ema-alpha", type=float, default=0.05,
                        help="EMA 平滑系数 (越小越平滑，默认 0.05)")

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
    print(f"  length_norm          = {args.length_norm}")
    print(f"  reward_schedule      = {args.reward_schedule}")
    if args.reward_schedule == 'dual':
        print(f"  phase_switch_thresh  = {args.phase_switch_threshold}")
        print(f"  ema_alpha            = {args.ema_alpha}")
    print(f"{'='*60}\n")

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 日志文件 ──
    log_tag = f"grpo_G{G}"
    mode = 'a' if args.resume_step > 0 else 'w'
    log_file = open(os.path.join(args.log_dir, f'{log_tag}_metrics.csv'), mode, newline='')
    csv_writer = csv.writer(log_file)
    if args.resume_step == 0:
        csv_writer.writerow([
            "step", "success_rate", "policy_entropy",
            "kl_div", "mean_advantage", "adv_std", "grad_norm", "grad_second_moment", "mean_response_length",
            "vram_allocated_gb", "vram_peak_gb", "vram_reserved_gb",
            "reward_phase", "ema_success_rate"
        ])

    response_file = open(os.path.join(args.log_dir, f'{log_tag}_responses.txt'), mode, encoding='utf-8')
    if args.resume_step == 0:
        response_file.write(f"=== GRPO G={G} Training Responses ===\n\n")
    else:
        response_file.write(f"\n=== Resumed GRPO G={G} Training from update_step {args.resume_step} ===\n\n")

    # 逐层梯度日志
    layer_grad_file = None
    if args.log_layer_grads:
        layer_grad_file = open(os.path.join(args.log_dir, f'{log_tag}_layer_grads.jsonl'), mode)

    # ── 模型加载 ──
    env = Arithmetic24Env()
    sft_path = args.sft_path
    if sft_path and not os.path.exists(sft_path):
        print(f"⚠️  WARNING: SFT path '{sft_path}' does NOT exist! Falling back to fresh base model + LoRA.")
        sft_path = None
    elif sft_path:
        print(f"✅ Found SFT checkpoint path: {sft_path}")
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

    update_step = args.resume_step
    step = args.resume_step * accum
    device = model.device
    optimizer.zero_grad()
    training_done = False  # 用于跳出双层循环

    # ── Dual-Phase Reward Schedule 状态 ──
    ema_success = 0.0           # EMA 滑动成功率
    reward_phase = "dense"      # 当前奖励阶段: "dense" (连续距离) / "binary" (二元)
    phase_switched = False      # 是否已切换过

    metric_acc = {"succ": 0.0, "adv": 0.0, "adv_std": 0.0, "kl": 0.0, "entropy": 0.0, "resp_len": 0.0}

    for epoch in range(args.epochs):
        if training_done:
            break
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
                gen_chunk = 16  # 减小 chunk 避免 7B 长序列 OOM
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

                # ── Dual-Phase Reward Schedule: 根据当前阶段选择奖励模式 ──
                if args.reward_schedule == 'dual' and reward_phase == 'binary':
                    # Phase 2: Binary reward (简化绝对信号)
                    group_rewards_list, corrects = compute_rewards_parallel(
                        [num_str] * G, responses, simple_mode='binary'
                    )
                else:
                    # Phase 1 / fixed: Dense continuous reward (原始模式)
                    group_rewards_list, corrects = compute_rewards_parallel(
                        [num_str] * G, responses
                    )

                group_rewards = torch.tensor(group_rewards_list, dtype=torch.float32).to(device, non_blocking=True)
                mean_r = group_rewards.mean()
                std_r = group_rewards.std() + 1e-4  # 增大平滑项，防止极寒环境下的优势爆炸
                advantages = (group_rewards - mean_r) / std_r

                # ── 改进 1: Length-Aware Reward Normalization ──
                # 消除 sequence-level advantage 广播到不同长度 response 时的隐性 length bias
                # 原理: advantage *= (group_mean_len / per_response_len)
                #   → 更长的 response 获得更小的 per-token advantage
                if args.length_norm:
                    resp_lengths = (resp_tensors != tokenizer.pad_token_id).float().sum(dim=1)
                    mean_len = resp_lengths.mean()
                    length_factor = mean_len / resp_lengths.clamp(min=1.0)
                    advantages = advantages * length_factor

                input_ids = group_out
                attention_mask = (input_ids != tokenizer.pad_token_id).long()

                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        mini_bs = min(8, G)  # 减小评估用的 mini_bs 以防 OOM
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
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )  # 训练阶段重新启用 (use_reentrant=False 兼容 4-bit)
            total_entropy = 0
            total_kl = 0

            for _ in range(args.ppo_epochs):
                for r in rollouts:
                    input_ids = r["input_ids"]
                    attention_mask = r["attention_mask"]
                    q_len_r = r["q_len"]
                    resp_tensors_r = r["resp_tensors"]
                    adv = r["advantages"].unsqueeze(1)
                    old_log_probs_r = r["old_log_probs"]
                    ref_log_probs_r = r["ref_log_probs"]
                    G_size = input_ids.shape[0]

                    # 降低 mini_bs 并做 per-minibatch backward 防止 OOM
                    mini_bs = 2  
                    for mi in range(0, G_size, mini_bs):
                        mb_ids = input_ids[mi:mi + mini_bs]
                        mb_mask = attention_mask[mi:mi + mini_bs]
                        mb_resp = resp_tensors_r[mi:mi + mini_bs]
                        mb_adv = adv[mi:mi + mini_bs]
                        mb_old_lp = old_log_probs_r[mi:mi + mini_bs]
                        mb_ref_lp = ref_log_probs_r[mi:mi + mini_bs]
                        mb_loss_mask = (mb_resp != tokenizer.pad_token_id).float()

                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            logits = model(mb_ids, attention_mask=mb_mask).logits
                            resp_logits = logits[:, q_len_r-1:-1, :]
                            
                            # 直接计算 log_softmax 避免重复分配显存
                            log_softmax = F.log_softmax(resp_logits, dim=-1)
                            mb_lp = torch.gather(log_softmax, 2, mb_resp.unsqueeze(-1)).squeeze(-1)
                            mb_lp = mb_lp * mb_loss_mask

                            # 正确的 entropy 计算：对完整词表分布求和
                            softmax = torch.exp(log_softmax)
                            mb_ent = -(softmax * log_softmax).sum(dim=-1)  # [batch, seq_len]
                            mb_ent = (mb_ent * mb_loss_mask).sum(dim=1) / mb_loss_mask.sum(dim=1).clamp(min=1)

                            ratio = torch.exp(mb_lp - mb_old_lp)
                            kl = torch.exp(mb_ref_lp - mb_lp) - (mb_ref_lp - mb_lp) - 1

                            surr1 = ratio * mb_adv
                            surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * mb_adv
                            policy_loss = -torch.min(surr1, surr2)

                            mb_loss = ((policy_loss + beta * kl) * mb_loss_mask).sum(dim=1) / mb_loss_mask.sum(dim=1).clamp(min=1)
                            
                            loss_for_backward = mb_loss.sum() / G_size / accum
                            entropy_bonus = args.entropy_coef * mb_ent.sum() / G_size / accum
                            loss_for_backward = loss_for_backward - entropy_bonus

                        # 将反向传播移动到小批量内部，立即释放计算图
                        loss_for_backward.backward()

                        total_entropy += (mb_ent.sum().item() / G_size)

                        seq_kl = (kl * mb_loss_mask).sum(dim=1) / mb_loss_mask.sum(dim=1).clamp(min=1)
                        total_kl += (seq_kl.sum().item() / G_size)

                        del logits, resp_logits, log_softmax, softmax, loss_for_backward, mb_loss, policy_loss, surr1, surr2, kl, ratio, mb_lp, mb_ent

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

                # ── VRAM 监控 ──
                vram_alloc = torch.cuda.memory_allocated() / 1e9
                vram_peak = torch.cuda.max_memory_allocated() / 1e9
                vram_reserved = torch.cuda.memory_reserved() / 1e9
                torch.cuda.reset_peak_memory_stats()  # 重置峰值以追踪每步

                # ── Dual-Phase Reward Schedule: EMA 更新与自动切换 ──
                ema_success = (1 - args.ema_alpha) * ema_success + args.ema_alpha * avg_succ
                if (args.reward_schedule == 'dual'
                        and not phase_switched
                        and ema_success >= args.phase_switch_threshold):
                    reward_phase = 'binary'
                    phase_switched = True
                    print(f"\n  🔄 [Dual-Phase] 奖励模式切换: dense → binary "
                          f"(EMA_succ={ema_success:.3f} ≥ {args.phase_switch_threshold})")

                csv_writer.writerow([
                    update_step, avg_succ, avg_entropy, avg_kl,
                    avg_adv, avg_adv_std,
                    grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm,
                    second_moment, avg_resp_len,
                    f"{vram_alloc:.2f}", f"{vram_peak:.2f}", f"{vram_reserved:.2f}",
                    reward_phase, f"{ema_success:.4f}"
                ])
                log_file.flush()

                phase_tag = f" [{reward_phase}]" if args.reward_schedule == 'dual' else ""
                print(f"Update {update_step} (Step {step}) | Succ: {avg_succ:.3f} | "
                      f"R: {avg_adv:.2f} | KL: {avg_kl:.4f} | |g|: {grad_norm:.4f} | β: {beta:.4f} | "
                      f"VRAM: {vram_alloc:.1f}/{vram_peak:.1f} GB{phase_tag}")

                if update_step > 0 and update_step % args.save_every == 0:
                    save_dir = os.path.join(args.output_dir, f"grpo_G{G}_update_{update_step}")
                    os.makedirs(save_dir, exist_ok=True)
                    model.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
                    print(f"  💾 Model saved → {save_dir}")

                metric_acc = {"succ": 0.0, "adv": 0.0, "adv_std": 0.0, "kl": 0.0, "entropy": 0.0, "resp_len": 0.0}
                update_step += 1

                if args.max_steps and update_step >= args.max_steps:
                    print(f"\n[!] 达到最大训练步数 --max-steps {args.max_steps}，提前终止。")
                    training_done = True
                    break


    # ── 保存最终模型 ──
    save_dir = os.path.join(args.output_dir, f"grpo_G{G}_final")
    os.makedirs(save_dir, exist_ok=True)
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