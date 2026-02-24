import os
os.environ["HF_HOME"] = "E:/hf_cache" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import csv
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from model_utils import load_model_and_tokenizer

def train_sft():
    os.makedirs('saved_models', exist_ok=True)
    
    model, tokenizer = load_model_and_tokenizer(with_value_head=False)
    
    # 修复 PEFT 状态标识
    model.is_peft_model = True

    # 读取数据集
    prompts = []
    responses = []
    with open('data/sft_train.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append(row['prompt'])
            responses.append(row['response'])
            
    # SFT 数据集格式化
    def formatting_func(example):
        example["text"] = f"{example['prompt']}{example['response']}{tokenizer.eos_token}"
        return example

    hf_dataset = Dataset.from_dict({"prompt": prompts, "response": responses})
    hf_dataset = hf_dataset.map(formatting_func)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_dataset = hf_dataset.map(tokenize_function, batched=True, remove_columns=["prompt", "response", "text"])
    
    config = TrainingArguments(
        output_dir="saved_models/sft_training",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        logging_steps=10,
        optim="adamw_torch"
    )

    from transformers import Trainer, DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=config,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    print("=== 开始 SFT 训练 ===")
    trainer.train()
    
    save_dir = "saved_models/sft_final"
    trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"=== SFT 训练完成，模型已保存至 {save_dir} ===")

if __name__ == "__main__":
    train_sft()
