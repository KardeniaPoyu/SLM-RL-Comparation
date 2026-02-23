import os
os.environ["HF_HOME"] = "E:/hf_cache" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import csv
import gc
from torch.utils.data import Dataset
from trl import PPOTrainer, PPOConfig
from model_utils import load_model_and_tokenizer
from env import Arithmetic24Env

class MathDataset(Dataset):
    def __init__(self, csv_file, tokenizer, env, max_samples=None):
        self.queries = []
        self.prompts = []
        self.input_nums = []
        with open(csv_file, 'r') as f:
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

def train():
    os.makedirs('logs', exist_ok=True)
    log_file = open('logs/ppo_metrics.csv', 'w', newline='')
    csv_writer = csv.writer(log_file)
    csv_writer.writerow([
        "step", "success_rate", "value_loss", "policy_entropy",
        "kl_div", "mean_advantage", "adv_std", "grad_norm", "grad_second_moment"
    ])
    
    env = Arithmetic24Env()
    model, tokenizer = load_model_and_tokenizer(with_value_head=True)
    
    config = PPOConfig(
        output_dir="logs/ppo_model",
        learning_rate=1e-5,
        mini_batch_size=1, 
        batch_size=8,      
        gradient_accumulation_steps=8, 
        num_ppo_epochs=1,
        max_grad_norm=1.0,
    )
    
    dataset = MathDataset('data/train.csv', tokenizer, env)
    
    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=None, 
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator
    )
    
    # 【黑科技：拦截器】在TRL清空梯度前抢救“梯度二阶矩”数据
    metric_cache = {"second_moment": 0.0, "total_norm": 0.0}
    original_step = ppo_trainer.optimizer.step
    
    def hooked_optimizer_step(*args, **kwargs):
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
        return original_step(*args, **kwargs)
        
    ppo_trainer.optimizer.step = hooked_optimizer_step
    
    gen_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "max_new_tokens": 128, # 【对齐】与GRPO的128保持一致
        "temperature": 0.8,
    }
    
    step = 0
    for epoch, batch in enumerate(ppo_trainer.dataloader):
        query_tensors = batch["query"]
        prompts = batch["prompt"]
        input_nums = batch["input_nums"]
        
        with torch.no_grad():
            response_tensors = ppo_trainer.generate(
                [q.to(ppo_trainer.accelerator.device) for q in query_tensors], 
                return_prompt=False, 
                **gen_kwargs
            )
            
        responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        
        rewards = []
        correct_count = 0
        for nums, resp in zip(input_nums, responses):
            r, is_corr = env.compute_reward(nums, resp)
            rewards.append(torch.tensor(r, dtype=torch.float32, device=ppo_trainer.accelerator.device))
            if is_corr:
                correct_count += 1
                
        # TRL 执行这一步时，会触发我们的 hooked_optimizer_step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        success_rate = correct_count / len(rewards)
        val_loss = stats.get("ppo/loss/value", 0.0)
        policy_entropy = stats.get("ppo/policy/entropy", 0.0)
        kl = stats.get("ppo/policy/approxkl", 0.0)
        returns = stats.get("ppo/returns/mean", 0.0)
        vpred = stats.get("ppo/val/vpred", 0.0)
        mean_adv = returns - vpred
        adv_std = stats.get("ppo/val/error", 0.0) 
        
        # 直接从拦截器的缓存中读取论文需要的数据
        total_norm = metric_cache["total_norm"]
        second_moment = metric_cache["second_moment"]
            
        csv_writer.writerow([
            step, success_rate, val_loss, policy_entropy, kl, mean_adv, adv_std, total_norm, second_moment
        ])
        log_file.flush()
        
        mean_reward = torch.stack(rewards).mean().item()
        print(f"Update {step} | Succ: {success_rate:.2f} | R: {mean_reward:.2f} | KL: {kl:.4f} | VLoss: {val_loss:.4f} | |g|: {total_norm:.4f}")
        step += 1
        
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    print("=== PPO 训练开始 ===")
    train()