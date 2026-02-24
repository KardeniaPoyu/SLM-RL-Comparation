import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import AutoModelForCausalLMWithValueHead

def load_model_and_tokenizer(model_name="Qwen/Qwen2.5-0.5B-Instruct", with_value_head=False, lora_resume_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Loading base model {model_name} in bfloat16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    if lora_resume_path and os.path.exists(lora_resume_path):
        from peft import PeftModel
        print(f"Resuming LoRA from {lora_resume_path}...")
        peft_model = PeftModel.from_pretrained(base_model, lora_resume_path, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            task_type="CAUSAL_LM",
            bias="none"
        )
        print("Applying fresh LoRA...")
        peft_model = get_peft_model(base_model, lora_config)
        
    peft_model.gradient_checkpointing_enable()
    
    if with_value_head:
        print("Wrapping model with Value Head for PPO...")
        # AutoModelForCausalLMWithValueHead can wrap a peft_model directly
        model = AutoModelForCausalLMWithValueHead(peft_model)
    else:
        model = peft_model
        
    return model, tokenizer
