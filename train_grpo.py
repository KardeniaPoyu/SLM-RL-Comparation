import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
if "HF_HOME" not in os.environ:
    if os.name == 'nt':
        os.environ["HF_HOME"] = "E:/hf_cache"
    else:
        # Default AutoDL cache path
        os.environ["HF_HOME"] = "/root/autodl-tmp/.cache/huggingface"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import csv
import gc
from model_utils import load_model_and_tokenizer
from env import Arithmetic24Env

class MathDataset(Dataset):
    def __init__(self, csv_file, tokenizer, env, max_samples=None):
        self.queries = []
        self.input_nums = []
        with open(csv_file, 'r') as f:
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
        return {
            "query": self.queries[idx],
            "input_nums": self.input_nums[idx]
        }

def get_per_token_logps(logits, input_ids):
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, 2, input_ids.unsqueeze(-1)).squeeze(-1)

def train():
    os.makedirs('logs', exist_ok=True)
    
    # 初始化 CSV 日志
    log_file = open('logs/grpo_metrics.csv', 'w', newline='')
    csv_writer = csv.writer(log_file)
    csv_writer.writerow([
        "step", "success_rate", "policy_entropy",
        "kl_div", "mean_advantage", "adv_std", "grad_norm", "grad_second_moment"
    ])
    
    # 初始化 Response 日志
    response_file = open('logs/grpo_responses.txt', 'w', encoding='utf-8')
    response_file.write("=== GRPO Training Responses Log ===\n\n")
    
    env = Arithmetic24Env()
    model, tokenizer = load_model_and_tokenizer(with_value_head=False, lora_resume_path="saved_models/sft_final")

    # 【修复 TRL 库缺失属性的 Bug】
    model.is_peft_model = True
    
    dataset = MathDataset('data/train.csv', tokenizer, env)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: x, num_workers=8)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    # 核心对齐参数
    G = 32
    accumulation_steps = 8
    
    beta = 0.01
    clip_eps = 0.2
    ppo_epochs = 1
    
    gen_kwargs = {
        "max_new_tokens": 256, # 给足空间防止截断
        "temperature": 1.0,   # 甚至试1.5（配合top_p=0.95）
        "top_p": 0.95,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    step = 0
    update_step = 0
    device = model.device
    
    optimizer.zero_grad() # 初始化梯度清零
    
    # ====================================================
    # 【核心修复】：在此处定义指标累加器（小篮子）
    # 用来收集 8 步的平均数据，以便最真实地画出折线图
    # ====================================================
    metric_acc = {"succ": 0.0, "adv": 0.0, "adv_std": 0.0, "kl": 0.0, "entropy": 0.0}
    
    for epoch in range(1):
        for batch in dataloader:
            model.eval()
            rollouts = []
            
            # 【优化】全量 Padding & 并发生成 (Batch Generation)
            all_q_tensors = []
            all_num_strs = []
            max_q_len = 0
            
            # 第一遍：收集所有 query，并找出最长长度用于 padding
            for item in batch:
                q_tensor = item["query"].to(device)
                all_q_tensors.append(q_tensor)
                all_num_strs.append(item["input_nums"])
                max_q_len = max(max_q_len, q_tensor.shape[0])
                
            # 第二遍：对齐 Padding 并复制 G 份
            padded_q_list = []
            for q_tensor in all_q_tensors:
                pad_len = max_q_len - q_tensor.shape[0]
                if pad_len > 0:
                    padded_q = F.pad(q_tensor, (pad_len, 0), value=tokenizer.pad_token_id)
                else:
                    padded_q = q_tensor
                # 复制 G 份
                padded_q_list.append(padded_q.unsqueeze(0).repeat(G, 1))
                
            # 拼成一个巨大的 Tensor: (BatchSize * G, max_q_len)
            huge_q_tensors = torch.cat(padded_q_list, dim=0)
            
            with torch.no_grad():
                # 一次性让 GPU 火力全开生成所有的回复
                outputs = model.generate(huge_q_tensors, **gen_kwargs)
            
            # 第三遍：拆除大 Batch，分配回各自的 rollout
            for i, num_str in enumerate(all_num_strs):
                start_idx = i * G
                end_idx = start_idx + G
                
                group_out = outputs[start_idx:end_idx]
                q_len = huge_q_tensors.shape[1] # 对齐以巨大的 padding 长度为准
                resp_tensors = group_out[:, q_len:]
                responses = tokenizer.batch_decode(resp_tensors, skip_special_tokens=True)

                # 【提速核心】减少硬盘 I/O：降低记录频率，每次只记 1 个样本观察即可
                if step % 10 == 0: 
                    response_file.write(f"Step {step} - Sample {i}:\n")
                    response_file.write(f"{responses[0]}\n")
                    response_file.write("-" * 80 + "\n")
                    response_file.flush()

                if step == 0 and i == 0: 
                    print(f"\n[模型原始输出观察]:\n{responses[0]}\n")
                
                group_rewards = []
                corrects = 0
                for r in responses:
                    reward_val, is_correct = env.compute_reward(num_str, r)
                    group_rewards.append(reward_val)
                    if is_correct: corrects += 1
                    
                group_rewards = torch.tensor(group_rewards, dtype=torch.float32, device=device)
                mean_r = group_rewards.mean()
                std_r = group_rewards.std() + 1e-8
                advantages = (group_rewards - mean_r) / std_r
                
                input_ids = group_out
                attention_mask = (input_ids != tokenizer.pad_token_id).long()
                
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        # Chunking batch to prevent OOM
                        mini_batch_size = 64
                        old_log_probs_list = []
                        ref_log_probs_list = []
                        for i in range(0, G, mini_batch_size):
                            mb_input_ids = input_ids[i:i+mini_batch_size]
                            mb_attention_mask = attention_mask[i:i+mini_batch_size]
                            mb_resp_tensors = resp_tensors[i:i+mini_batch_size]
                            
                            logits = model(mb_input_ids, attention_mask=mb_attention_mask).logits
                            mb_old_log_probs = get_per_token_logps(logits[:, q_len-1:-1, :], mb_resp_tensors).detach()
                            old_log_probs_list.append(mb_old_log_probs)
                            del logits
                            
                            with model.disable_adapter():
                                ref_logits = model(mb_input_ids, attention_mask=mb_attention_mask).logits
                                mb_ref_log_probs = get_per_token_logps(ref_logits[:, q_len-1:-1, :], mb_resp_tensors).detach()
                                ref_log_probs_list.append(mb_ref_log_probs)
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
                    "success_rate": corrects / G
                })
                
            # 2. Optimization Phase
            model.train()
            
            total_entropy = 0
            total_kl = 0
            
            for _ in range(ppo_epochs):
                for r in rollouts:
                    input_ids = r["input_ids"]
                    attention_mask = r["attention_mask"]
                    q_len = r["q_len"]
                    resp_tensors = r["resp_tensors"]
                    advantages = r["advantages"].unsqueeze(1)
                    
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        mini_batch_size = 64
                        log_probs_list = []
                        for i in range(0, input_ids.shape[0], mini_batch_size):
                            mb_input_ids = input_ids[i:i+mini_batch_size]
                            mb_attention_mask = attention_mask[i:i+mini_batch_size]
                            mb_resp_tensors = resp_tensors[i:i+mini_batch_size]
                            
                            logits = model(mb_input_ids, attention_mask=mb_attention_mask).logits
                            mb_log_probs = get_per_token_logps(logits[:, q_len-1:-1, :], mb_resp_tensors)
                            log_probs_list.append(mb_log_probs)
                            del logits
                        log_probs = torch.cat(log_probs_list, dim=0)
                    
                    loss_mask = (resp_tensors != tokenizer.pad_token_id).float()
                    ratio = torch.exp(log_probs - r["old_log_probs"])
                    kl = torch.exp(r["ref_log_probs"] - log_probs) - (r["ref_log_probs"] - log_probs) - 1
                    
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
                    policy_loss = -torch.min(surr1, surr2)
                    
                    loss = ((policy_loss + beta * kl) * loss_mask).sum(dim=1) / loss_mask.sum(dim=1)
                    loss = loss.mean()
                    
                    prob = torch.exp(log_probs)
                    entropy = -(prob * log_probs * loss_mask).sum(dim=1) / loss_mask.sum(dim=1)
                    
                    # 梯度累积
                    loss = loss / accumulation_steps
                    entropy_bonus = 0.005 * entropy.mean()
                    entropy_bonus = entropy_bonus / accumulation_steps   # ← 加这一行
                    loss = loss - entropy_bonus
                    loss.backward()
                                        
                    total_entropy += entropy.mean().item()
                    total_kl += (kl * loss_mask).sum(dim=1).mean().item()
                    
            step += 1
            
            # 【完美收集数据】：将这一步的数据放进篮子里
            metric_acc["succ"] += sum([r["success_rate"] for r in rollouts]) / len(rollouts)
            metric_acc["adv"] += sum([r["reward_mean"] for r in rollouts]) / len(rollouts)
            metric_acc["adv_std"] += sum([r["reward_std"] for r in rollouts]) / len(rollouts)
            metric_acc["kl"] += total_kl
            metric_acc["entropy"] += total_entropy
            
            # 及时释放引用
            rollouts.clear() 

            # 【满 8 步】：求平均值 -> 更新权重 -> 写入 CSV
            if step % accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                second_moment = 0.0
                param_count = 0
                for p in model.parameters():
                    if p.grad is not None:
                        second_moment += (p.grad.data ** 2).mean().item()
                        param_count += 1
                if param_count > 0:
                    second_moment /= param_count
                    
                optimizer.step()
                optimizer.zero_grad() 
                
                # 计算这 8 步的真实平均值
                avg_succ = metric_acc["succ"] / accumulation_steps
                avg_adv = metric_acc["adv"] / accumulation_steps
                avg_adv_std = metric_acc["adv_std"] / accumulation_steps
                avg_kl = metric_acc["kl"] / accumulation_steps
                avg_entropy = metric_acc["entropy"] / accumulation_steps
                
                # 完美写入 CSV（用于画图）
                csv_writer.writerow([
                    update_step, avg_succ, avg_entropy, avg_kl,
                    avg_adv, avg_adv_std, grad_norm.item(), second_moment
                ])
                log_file.flush()
                
                # 终端汇报 Update
                print(f"Update {update_step} (Step {step}) | Succ: {avg_succ:.2f} | Adv: {avg_adv:.2f} | KL: {avg_kl:.4f} | |g|: {grad_norm.item():.4f}")
                
                if update_step > 0 and update_step % 40 == 0:
                    save_dir = f"saved_models/grpo_update_{update_step}"
                    model.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
                    print(f"[{update_step}] Model saved to {save_dir}")

                # 清空篮子，迎接下一个 8 步
                metric_acc = {"succ": 0.0, "adv": 0.0, "adv_std": 0.0, "kl": 0.0, "entropy": 0.0}
                update_step += 1
    
    # 保存最终模型
    save_dir = "saved_models/grpo_final"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Final model saved to {save_dir}")

    # 关闭文件
    log_file.close()
    response_file.close()
    print(f"\n=== GRPO 训练完成 ===")
    print(f"指标已保存到: logs/grpo_metrics.csv")
    print(f"响应已保存到: logs/grpo_responses.txt")

if __name__ == "__main__":
    print("=== GRPO 训练开始 ===")
    train()