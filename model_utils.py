import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import AutoModelForCausalLMWithValueHead

def load_model_and_tokenizer(model_name="Qwen/Qwen2.5-0.5B-Instruct", with_value_head=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    print(f"Loading base model {model_name} in 4-bit...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    print("Preparing model for kbit training...")
    base_model = prepare_model_for_kbit_training(base_model)
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
        bias="none"
    )
    
    print("Applying LoRA...")
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.gradient_checkpointing_enable()
    
    if with_value_head:
        print("Wrapping model with Value Head for PPO...")
        # AutoModelForCausalLMWithValueHead can wrap a peft_model directly
        model = AutoModelForCausalLMWithValueHead(peft_model)
    else:
        model = peft_model
        
    return model, tokenizer
