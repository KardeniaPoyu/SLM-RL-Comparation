import os
os.environ["HF_HOME"] = "E:/hf_cache" 
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
    log_file = open('logs/grpo_metrics.csv', 'w', newline='')
    csv_writer = csv.writer(log_file)
    csv_writer.writerow([
        "step", "success_rate", "policy_entropy",
        "kl_div", "mean_advantage", "adv_std", "grad_norm", "grad_second_moment"
    ])
    
    env = Arithmetic24Env()
    model, tokenizer = load_model_and_tokenizer(with_value_head=False)

    # 【修复 TRL 库缺失属性的 Bug】
    model.is_peft_model = True
    
    dataset = MathDataset('data/train.csv', tokenizer, env)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    # 核心对齐参数
    G = 4
    accumulation_steps = 8 # 攒够8道题更新一次，等效于 batch_size=8
    beta = 0.04
    clip_eps = 0.2
    ppo_epochs = 1
    
    gen_kwargs = {
        "max_new_tokens": 128, # 统一对齐为 96
        "temperature": 0.8,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    step = 0
    update_step = 0
    device = model.device
    
    optimizer.zero_grad() # 初始化梯度清零
    
    for epoch in range(1):
        for batch in dataloader:
            model.eval()
            rollouts = []
            
            # 1. Rollout Phase
            for item in batch:
                q_tensor = item["query"].to(device)
                num_str = item["input_nums"]
                q_tensors = q_tensor.unsqueeze(0).repeat(G, 1)
                
                with torch.no_grad():
                    outputs = model.generate(q_tensors, **gen_kwargs)
                
                q_len = q_tensors.shape[1]
                resp_tensors = outputs[:, q_len:]
                responses = tokenizer.batch_decode(resp_tensors, skip_special_tokens=True)

                if step == 0: 
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
                
                input_ids = outputs
                attention_mask = (input_ids != tokenizer.pad_token_id).long()
                
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(input_ids, attention_mask=attention_mask).logits
                        old_log_probs = get_per_token_logps(logits[:, q_len-1:-1, :], resp_tensors).detach()
                        del logits
                        
                        with model.disable_adapter():
                            ref_logits = model(input_ids, attention_mask=attention_mask).logits
                            ref_log_probs = get_per_token_logps(ref_logits[:, q_len-1:-1, :], resp_tensors).detach()
                            del ref_logits
                
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
            
            for _ in range(ppo_epochs):
                total_entropy = 0
                total_kl = 0
                
                for r in rollouts:
                    input_ids = r["input_ids"]
                    attention_mask = r["attention_mask"]
                    q_len = r["q_len"]
                    resp_tensors = r["resp_tensors"]
                    advantages = r["advantages"].unsqueeze(1)
                    
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(input_ids, attention_mask=attention_mask).logits
                        log_probs = get_per_token_logps(logits[:, q_len-1:-1, :], resp_tensors)
                        del logits
                    
                    loss_mask = (resp_tensors != tokenizer.pad_token_id).float()
                    ratio = torch.exp(log_probs - r["old_log_probs"])
                    kl = torch.exp(r["ref_log_probs"] - log_probs) - (r["ref_log_probs"] - log_probs) - 1
                    
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
                    policy_loss = -torch.min(surr1, surr2)
                    
                    loss = ((policy_loss + beta * kl) * loss_mask).sum(dim=1) / loss_mask.sum(dim=1)
                    loss = loss.mean()
                    
                    # 梯度累积：除以 accumulation_steps 保证梯度幅值稳定
                    loss = loss / accumulation_steps
                    loss.backward()
                    
                    prob = torch.exp(log_probs)
                    entropy = -(prob * log_probs * loss_mask).sum(dim=1) / loss_mask.sum(dim=1)
                    total_entropy += entropy.mean().item()
                    total_kl += (kl * loss_mask).sum(dim=1).mean().item()
                    
            step += 1
            
            # 【关键修改 1：每次反向传播完，立刻清空张量释放显存】
            avg_succ = sum([r["success_rate"] for r in rollouts]) / len(rollouts)
            avg_adv = sum([r["reward_mean"] for r in rollouts]) / len(rollouts)
            avg_adv_std = sum([r["reward_std"] for r in rollouts]) / len(rollouts)
            
            rollouts.clear() 
            torch.cuda.empty_cache()
            gc.collect()

            # 【关键修改 2：满 8 步，执行一次底层权重更新，并写入日志】
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
                optimizer.zero_grad() # 更新完毕后，清空底层累积的梯度
                
                # 记录这 8 步的平均统计数据
                csv_writer.writerow([
                    update_step, avg_succ, total_entropy, total_kl,
                    avg_adv, avg_adv_std, grad_norm.item(), second_moment
                ])
                log_file.flush()
                
                print(f"Update {update_step} (Step {step}) | Succ: {avg_succ:.2f} | Adv: {avg_adv:.2f} | KL: {total_kl:.4f} | |g|: {grad_norm.item():.4f}")
                update_step += 1

if __name__ == "__main__":
    print("=== GRPO 训练开始 ===")
    train()