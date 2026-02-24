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
    model.is_peft_model = True

    prompts = []
    responses = []
    with open('data/sft_train.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append(row['prompt'])
            responses.append(row['response'])
            
    # 【修复1】：不需要手动分词，交给 SFTTrainer
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['prompt'])):
            text = f"{example['prompt'][i]}{example['response'][i]}{tokenizer.eos_token}"
            output_texts.append(text)
        return output_texts

    hf_dataset = Dataset.from_dict({"prompt": prompts, "response": responses})

    # 【修复2】：利用 TRL 神器，只对输出部分计算 Loss (增加对不同 TRL 版本的兼容)
    response_template = "输出：\n"
    try:
        from trl import DataCollatorForCompletionOnlyLM
        collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)
    except ImportError:
        try:
            from trl.trainer import DataCollatorForCompletionOnlyLM
            collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)
        except ImportError:
            from transformers import DataCollatorForLanguageModeling
            class CustomCompletionCollator(DataCollatorForLanguageModeling):
                def __init__(self, response_template, tokenizer, *args, **kwargs):
                    super().__init__(tokenizer=tokenizer, mlm=False, *args, **kwargs)
                    self.response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
                def torch_call(self, examples):
                    batch = super().torch_call(examples)
                    for i in range(len(batch["labels"])):
                        labels = batch["labels"][i]
                        idx = -1
                        tmpl_len = len(self.response_template_ids)
                        for j in range(len(labels) - tmpl_len + 1):
                            if labels[j:j+tmpl_len].tolist() == self.response_template_ids:
                                idx = j + tmpl_len
                                break
                        if idx != -1:
                            batch["labels"][i, :idx] = -100
                        else:
                            batch["labels"][i, :] = -100 # 如果找不到模板，则设为忽略
                    return batch
            collator = CustomCompletionCollator(response_template=response_template, tokenizer=tokenizer)

    # 【修复3】：调整小模型的 SFT 超参数
    config = TrainingArguments(
        output_dir="saved_models/sft_training",
        save_strategy="epoch",
        learning_rate=2e-4, 
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=4, # 小模型+严格格式要求，建议 Epoch 增加到 3-5
        logging_steps=10,
        optim="adamw_torch",
        bf16=True, # 如果显卡支持，强烈建议开 bf16 防溢出，若报错可改为 False
        max_grad_norm=1.0,
        dataloader_num_workers=8,
    )

    # 【终极修复4】：动态绕过 TRL 版本冲突导致的 tokenizer 报错
    import inspect
    from transformers import Trainer
    _original_init = Trainer.__init__
    sig = inspect.signature(_original_init)
    
    def _patched_init(self, *args, **kwargs):
        if "tokenizer" in kwargs:
            if "processing_class" in sig.parameters:
                kwargs["processing_class"] = kwargs.pop("tokenizer")
            elif "tokenizer" not in sig.parameters:
                kwargs.pop("tokenizer")
        _original_init(self, *args, **kwargs)
    Trainer.__init__ = _patched_init

    from trl import SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=hf_dataset,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        max_seq_length=1024, # SFTTrainer 可以直接接管截断
        tokenizer=tokenizer, # ⬅️ 交给 monkey patch 自动处理成 processing_class
    )

    print("=== 开始严格的 SFT 训练 ===")
    trainer.train()
    
    save_dir = "saved_models/sft_final"
    trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"=== SFT 训练完成，模型已保存至 {save_dir} ===")

if __name__ == "__main__":
    train_sft()