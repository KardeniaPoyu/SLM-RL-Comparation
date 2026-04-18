"""
model_utils.py — 模型加载工具
支持 Qwen2.5-0.5B + LoRA，可选 Value Head (PPO)
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 防止 tokenizer 多线程抢占 CPU

import json
import shutil
import inspect
import torch
def sanitize_lora_config(lora_path):
    """
    清理 adapter_config.json 中不被当前 LoraConfig 支持的非法键。
    解决 'TypeError: LoraConfig.__init__() got an unexpected keyword argument' 问题。
    """
    config_file = os.path.join(lora_path, "adapter_config.json")
    if not os.path.exists(config_file):
        return

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        # 获取当前 LoraConfig 构造函数支持的参数列表
        import inspect
        valid_keys = set(inspect.signature(LoraConfig.__init__).parameters.keys())
        # PEFT 内部还可能使用一些基类参数或特殊映射，补全常见基础键
        valid_keys.update(["peft_type", "auto_mapping", "base_model_name_or_path", "revision", "task_type", "inference_mode"])

        original_keys = set(config_data.keys())
        invalid_keys = original_keys - valid_keys

        if invalid_keys:
            print(f"  [sanitize] Found {len(invalid_keys)} invalid keys in LoRA config: {list(invalid_keys)}")
            # 备份原文件
            import shutil
            backup_file = config_file + ".bak"
            if not os.path.exists(backup_file):
                shutil.copy(config_file, backup_file)
            
            # 过滤并保存
            new_config = {k: v for k, v in config_data.items() if k in valid_keys}
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(new_config, f, indent=2)
            print(f"  [sanitize] Cleaned config saved to {config_file}")
    except Exception as e:
        print(f"  ⚠️ [sanitize] Failed to sanitize config: {e}")
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model_and_tokenizer(model_name="Qwen/Qwen2.5-7B-Instruct",
                              with_value_head=False,
                              lora_resume_path=None,
                              gradient_checkpointing=True):
    """
    加载基座模型 + LoRA，可选加载 Value Head。

    Args:
        model_name: HuggingFace 模型名或本地路径
        with_value_head: 是否加 Value Head (PPO 需要)
        lora_resume_path: LoRA 权重路径（SFT 预训练后恢复）
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model {model_name} in 4-bit (for 7B VRAM safety)...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    # Flash Attention 2: 仅在已安装 flash_attn 时启用，避免 double-load 破坏 CUDA 状态
    fa_kwargs = {}
    try:
        import flash_attn  # noqa: F401
        fa_kwargs["attn_implementation"] = "flash_attention_2"
        print("  ✅ Flash Attention 2 will be used")
    except ImportError:
        pass

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        **fa_kwargs,
    )

    if lora_resume_path and os.path.exists(lora_resume_path):
        from peft import PeftModel
        print(f"Resuming LoRA from {lora_resume_path}...")
        # 自动清洗非标准配置参数
        sanitize_lora_config(lora_resume_path)
        peft_model = PeftModel.from_pretrained(base_model, lora_resume_path, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
            bias="none"
        )
        print("Applying fresh LoRA...")
        peft_model = get_peft_model(base_model, lora_config)

    # ── 先包裹 ValueHead，再设置梯度检查点 ──
    if with_value_head:
        from trl import AutoModelForCausalLMWithValueHead
        print("Wrapping model with Value Head for PPO...")
        model = AutoModelForCausalLMWithValueHead(peft_model)
    else:
        model = peft_model

    # 获取实际的 pretrained_model（ValueHead 包裹后需要通过 .pretrained_model 访问）
    inner_model = model.pretrained_model if with_value_head else model

    if hasattr(inner_model, "enable_input_require_grads"):
        inner_model.enable_input_require_grads()

    if gradient_checkpointing:
        inner_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        inner_model.config.use_cache = False  # 禁用 config 层面的 cache，避免 transformers 抛出显式警告，生成时再强行 kwargs 传入开启
        print("  ✅ Gradient checkpointing enabled (use_reentrant=False)")
    else:
        print("  ⏭️ Gradient checkpointing disabled (sufficient VRAM)")

    return model, tokenizer


def collect_per_layer_grad_stats(model):
    """
    按 LoRA 层分组记录梯度统计。
    返回 dict[layer_key -> {"norm": float, "second_moment": float}]
    """
    layer_stats = {}
    for name, p in model.named_parameters():
        if p.grad is not None and "lora" in name:
            layer_key = name.rsplit(".", 1)[0]
            if layer_key not in layer_stats:
                layer_stats[layer_key] = {"norm_sq": 0.0, "second_moment": 0.0, "count": 0}
            layer_stats[layer_key]["norm_sq"] += p.grad.data.norm(2).item() ** 2
            layer_stats[layer_key]["second_moment"] += (p.grad.data ** 2).mean().item()
            layer_stats[layer_key]["count"] += 1

    result = {}
    for k, v in layer_stats.items():
        result[k] = {
            "norm": v["norm_sq"] ** 0.5,
            "second_moment": v["second_moment"] / max(v["count"], 1)
        }
    return result
