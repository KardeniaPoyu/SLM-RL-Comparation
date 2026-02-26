"""
evaluate.py — 论文最终评估脚本
加载训练好的模型权重，在全新测试集上做推理，记录成功率

用法:
    python evaluate.py                                    # 评估所有模型
    python evaluate.py --models saved_models/ppo_final    # 仅评估 PPO
    python evaluate.py --n-samples 200                    # 每个难度 200 道
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import csv
import glob
import torch
import numpy as np
import gc

# ── PyTorch 2.8 + numpy 全面兼容补丁 ──
_orig_tensor_getitem = torch.Tensor.__getitem__
def _numpy_compat_getitem(self, indices):
    if type(indices).__module__ == 'numpy' or type(indices).__name__ == 'ndarray':
        if hasattr(indices, 'tolist'):
            return _orig_tensor_getitem(self, indices.tolist())
        elif hasattr(indices, 'item'):
            return _orig_tensor_getitem(self, indices.item())
    
    if isinstance(indices, tuple):
        new_indices = []
        changed = False
        for idx in indices:
            if type(idx).__module__ == 'numpy' or type(idx).__name__ == 'ndarray':
                if hasattr(idx, 'tolist'):
                    new_indices.append(idx.tolist())
                elif hasattr(idx, 'item'):
                    new_indices.append(idx.item())
                else:
                    new_indices.append(idx)
                changed = True
            else:
                new_indices.append(idx)
        if changed:
            return _orig_tensor_getitem(self, tuple(new_indices))
            
    return _orig_tensor_getitem(self, indices)
torch.Tensor.__getitem__ = _numpy_compat_getitem

_orig_torch_tensor = torch.tensor
def _compat_torch_tensor(data, *args, **kwargs):
    if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], torch.Tensor):
        device = kwargs.pop('device', None)
        dtype = kwargs.pop('dtype', None)
        stacked = torch.stack([d.detach().cpu() for d in data])
        if dtype is not None:
            stacked = stacked.to(dtype=dtype)
        if device is not None:
            stacked = stacked.to(device=device)
        return stacked
    return _orig_torch_tensor(data, *args, **kwargs)
torch.tensor = _compat_torch_tensor
from datetime import datetime
from model_utils import load_model_and_tokenizer
from env import Arithmetic24Env


def load_test_data(data_file, n_filter=None, max_samples=None):
    """加载测试数据，支持按 N 过滤"""
    samples = []

    if data_file.endswith('.jsonl'):
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())
                if n_filter and record.get('n') != n_filter:
                    continue
                samples.append(record['nums'])
                if max_samples and len(samples) >= max_samples:
                    break
    else:
        with open(data_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append(row['nums'])
                if max_samples and len(samples) >= max_samples:
                    break

    return samples


def evaluate_model(model, tokenizer, env, test_samples, max_new_tokens=128,
                   temperature=0.7, top_p=0.95, batch_size=16):
    """
    在测试集上做推理，返回成功率和详细结果。
    使用较低的 temperature (0.7) 以获得更稳定的评估结果。
    """
    model.eval()
    device = next(model.parameters()).device

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    total = 0
    correct = 0
    results = []

    for i in range(0, len(test_samples), batch_size):
        batch_nums = test_samples[i:i + batch_size]
        prompts = [env.get_prompt(nums) for nums in batch_nums]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        q_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs
            )

        resp_tokens = outputs[:, q_len:]
        responses = tokenizer.batch_decode(resp_tokens, skip_special_tokens=True)

        for nums, resp in zip(batch_nums, responses):
            reward, is_correct = env.compute_reward(nums, resp)
            total += 1
            if is_correct:
                correct += 1
            results.append({
                "nums": nums,
                "response": resp[:200],  # 截断过长的输出
                "correct": is_correct,
                "reward": reward
            })

        # 进度
        print(f"    [{min(i + batch_size, len(test_samples))}/{len(test_samples)}] "
              f"Acc: {correct}/{total} ({correct/max(total,1)*100:.1f}%)", end='\r')

    print()  # 换行
    success_rate = correct / max(total, 1)
    return success_rate, results


def find_model_dirs(base_dir="saved_models"):
    """自动发现所有 *_final 模型目录"""
    dirs = []
    patterns = [
        os.path.join(base_dir, "ppo_final"),
        os.path.join(base_dir, "sft_final"),
    ]
    # GRPO 消融模型
    patterns += glob.glob(os.path.join(base_dir, "grpo_G*_final"))

    for d in patterns:
        if os.path.isdir(d):
            dirs.append(d)

    return sorted(dirs)


def main():
    parser = argparse.ArgumentParser(description="论文最终评估")
    parser.add_argument("--models", nargs='+', type=str, default=None,
                        help="要评估的模型路径 (默认: 自动发现所有 *_final 模型)")
    parser.add_argument("--test-file", type=str, default="data/test.csv",
                        help="测试数据路径")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="每个难度的测试题数 (默认: 100)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="推理 batch size")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="生成温度 (评估用较低值)")
    parser.add_argument("--output-dir", type=str, default="logs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    env = Arithmetic24Env()

    # ── 加载测试数据 ──
    test_file = args.test_file
    test_jsonl = test_file.replace('.csv', '.jsonl')

    # 优先用 JSONL（含 N 标签），否则用 CSV
    if os.path.exists(test_jsonl):
        test_file = test_jsonl

    test_data_by_n = {}
    if test_file.endswith('.jsonl'):
        for n in [3, 4, 5]:
            samples = load_test_data(test_file, n_filter=n, max_samples=args.n_samples)
            if samples:
                test_data_by_n[n] = samples
                print(f"  N={n}: {len(samples)} 道测试题")
    else:
        # CSV 没有 N 标签，全部加载
        samples = load_test_data(test_file, max_samples=args.n_samples * 2)
        if samples:
            test_data_by_n["all"] = samples
            print(f"  全部: {len(samples)} 道测试题")

    if not test_data_by_n:
        print("❌ 没有找到测试数据！请先运行 data_gen_multi.py")
        return

    # ── 发现模型 ──
    model_dirs = args.models or find_model_dirs()
    if not model_dirs:
        print("❌ 没有找到任何训练好的模型！请先运行训练脚本。")
        return

    print(f"\n找到 {len(model_dirs)} 个模型: {model_dirs}")

    # ── 评估结果汇总表 ──
    summary = []

    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        print(f"\n{'='*60}")
        print(f"  评估模型: {model_name}")
        print(f"{'='*60}")

        try:
            model, tokenizer = load_model_and_tokenizer(
                lora_resume_path=model_dir,
                with_value_head=False
            )
        except Exception as e:
            print(f"  ⚠️ 加载失败: {e}")
            continue

        row = {"model": model_name}

        for n_key, samples in test_data_by_n.items():
            label = f"N={n_key}" if isinstance(n_key, int) else n_key
            print(f"\n  ── {label} ({len(samples)} 道) ──")

            success_rate, results = evaluate_model(
                model, tokenizer, env, samples,
                batch_size=args.batch_size,
                temperature=args.temperature
            )

            row[f"acc_{n_key}"] = success_rate
            print(f"  ✅ {label} 成功率: {success_rate*100:.1f}%")

            # 保存详细结果
            detail_path = os.path.join(
                args.output_dir, f"eval_{model_name}_{n_key}.jsonl"
            )
            with open(detail_path, 'w', encoding='utf-8') as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')

        summary.append(row)

        # 释放显存
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    # ── 汇总表输出 ──
    print(f"\n{'='*60}")
    print("📊 最终评估汇总")
    print(f"{'='*60}")

    # 表头
    n_keys = list(test_data_by_n.keys())
    header = f"{'Model':<25}"
    for n_key in n_keys:
        label = f"N={n_key}" if isinstance(n_key, int) else n_key
        header += f" | {label:>8}"
    print(header)
    print("-" * len(header))

    for row in summary:
        line = f"{row['model']:<25}"
        for n_key in n_keys:
            acc = row.get(f"acc_{n_key}", 0)
            line += f" | {acc*100:7.1f}%"
        print(line)

    # 保存 CSV
    summary_path = os.path.join(args.output_dir, "eval_summary.csv")
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        cols = ["model"] + [f"acc_N{k}" for k in n_keys]
        writer.writerow(cols)
        for row in summary:
            writer.writerow([row["model"]] + [row.get(f"acc_{k}", 0) for k in n_keys])

    print(f"\n📝 汇总表: {summary_path}")
    print(f"📂 详细结果: {args.output_dir}/eval_*.jsonl")


if __name__ == "__main__":
    main()
