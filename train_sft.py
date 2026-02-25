"""
train_sft.py — SFT 预热训练
教会 0.5B 模型输出 <think>...</think> + 表达式 格式

用法:
    python train_sft.py                            # 默认配置
    python train_sft.py --data data/sft_train.csv
    python train_sft.py --epochs 6 --lr 2e-4
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import torch
import csv
import json
from datasets import Dataset
from transformers import TrainingArguments
from model_utils import load_model_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="SFT Training")
    parser.add_argument("--data", type=str, default="data/sft_train.csv",
                        help="SFT 数据路径 (CSV 或 JSONL)")
    parser.add_argument("--output-dir", type=str, default="saved_models/sft_final")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    return parser.parse_args()


def load_sft_data(data_path):
    """加载 SFT 数据，支持 CSV 和 JSONL"""
    prompts, responses = [], []

    if data_path.endswith('.jsonl'):
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())
                prompts.append(record['prompt'])
                responses.append(record['response'])
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompts.append(row['prompt'])
                responses.append(row['response'])

    return prompts, responses


def train_sft(args):
    os.makedirs(os.path.dirname(args.output_dir) or '.', exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(with_value_head=False)
    model.is_peft_model = True

    prompts, responses = load_sft_data(args.data)
    print(f"Loaded {len(prompts)} SFT examples from {args.data}")

    # 预格式化为 'text' 列（兼容所有 TRL 版本）
    texts = [
        f"{p}{r}{tokenizer.eos_token}"
        for p, r in zip(prompts, responses)
    ]
    hf_dataset = Dataset.from_dict({"text": texts})

    # 只对 response 部分计算 Loss
    response_template = "</think> 后只输出最终公式。\n<think>\n"
    try:
        from trl import DataCollatorForCompletionOnlyLM
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template, tokenizer=tokenizer
        )
    except ImportError:
        try:
            from trl.trainer import DataCollatorForCompletionOnlyLM
            collator = DataCollatorForCompletionOnlyLM(
                response_template=response_template, tokenizer=tokenizer
            )
        except ImportError:
            from transformers import DataCollatorForLanguageModeling
            class CustomCompletionCollator(DataCollatorForLanguageModeling):
                def __init__(self, response_template, tokenizer, *args_c, **kwargs_c):
                    super().__init__(tokenizer=tokenizer, mlm=False, *args_c, **kwargs_c)
                    self.response_template_ids = tokenizer.encode(
                        response_template, add_special_tokens=False
                    )
                def torch_call(self, examples):
                    batch = super().torch_call(examples)
                    for i in range(len(batch["labels"])):
                        labels = batch["labels"][i]
                        tmpl_len = len(self.response_template_ids)
                        idx = -1
                        for j in range(len(labels) - tmpl_len + 1):
                            if labels[j:j + tmpl_len].tolist() == self.response_template_ids:
                                idx = j + tmpl_len
                                break
                        if idx != -1:
                            batch["labels"][i, :idx] = -100
                        else:
                            batch["labels"][i, :] = -100
                    return batch
            collator = CustomCompletionCollator(
                response_template=response_template, tokenizer=tokenizer
            )

    config = TrainingArguments(
        output_dir=args.output_dir + "_checkpoints",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        logging_steps=10,
        optim="adamw_torch",
        bf16=True,
        max_grad_norm=1.0,
        dataloader_num_workers=0,
    )

    from trl import SFTTrainer

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=hf_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )

    print(f"\n=== SFT 训练开始 ({args.epochs} epochs, lr={args.lr}) ===")
    trainer.train()

    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"=== SFT 完成，模型已保存至 {args.output_dir} ===")


if __name__ == "__main__":
    args = parse_args()
    train_sft(args)