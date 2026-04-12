import subprocess
import os
import time
import argparse

"""
run_all_experiments.py — 论文消融实验全自动运行流水线
支持一键完成: SFT -> GRPO Baseline -> V-GRPO (Full Combo) -> Evaluation

用法:
    python run_all_experiments.py --model-name Qwen/Qwen2.5-0.5B-Instruct
"""

def run_command(cmd, desc):
    print(f"\n{'='*60}")
    print(f"🚀 开始执行: {desc}")
    print(f"命令: {cmd}")
    print(f"{'='*60}")
    start_time = time.time()
    try:
        # 使用 shell=True 兼容 Windows
        process = subprocess.Popen(cmd, shell=True)
        process.wait()
        if process.returncode != 0:
            print(f"❌ 执行失败 (Exit Code: {process.returncode})")
            return False
    except Exception as e:
        print(f"❌ 运行报错: {e}")
        return False
    
    elapsed = time.time() - start_time
    print(f"✅ 执行完毕! 耗时: {elapsed:.1f}s")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--steps", type=int, default=100, help="GRPO 训练步数")
    parser.add_argument("--skip-sft", action="store_true")
    args = parser.parse_args()

    model_name = args.model_name
    sft_out = "saved_models/sft_0.5b_local"
    grpo_base_out = "saved_models/grpo_baseline_0.5b"
    grpo_v_out = "saved_models/grpo_vgrpo_0.5b"

    # 1. SFT 预热 (500 条数据, 4 epoch)
    if not args.skip_sft:
        sft_cmd = f"python train_sft.py --model-name {model_name} --output-dir {sft_out} --epochs 4 --batch-size 4 --grad-accum 8"
        if not run_command(sft_cmd, "SFT 格式预热训练"):
            return

    # 2. GRPO Baseline (G=4, 无改进项)
    grpo_base_cmd = f"python train_grpo.py --model-name {model_name} --sft-path {sft_out} --output-dir {grpo_base_out} --group-size 4 --max-steps {args.steps} --batch-size 4"
    if not run_command(grpo_base_cmd, "GRPO Baseline 训练 (G=4)"):
        return

    # 3. V-GRPO (Full Combo: 长度归一/平滑退火/剪裁/过滤/多样性奖金)
    v_grpo_cmd = (
        f"python train_grpo.py --model-name {model_name} --sft-path {sft_out} "
        f"--output-dir {grpo_v_out} --group-size 4 --max-steps {args.steps} "
        f"--length-norm --reward-schedule anneal --adv-clip --filter-solvable --diversity-bonus"
    )
    if not run_command(v_grpo_cmd, "V-GRPO (全家桶优化版) 训练"):
        return

    # 4. 最终评估对照
    eval_cmd = f"python evaluate.py --base-model {model_name} --models {grpo_base_out}/grpo_G4_final {grpo_v_out}/grpo_G4_final --n-samples 50"
    run_command(eval_cmd, "模型性能对比评估")

    print("\n" + "#"*60)
    print("🏆 消融实验流水线执行成功！")
    print(f"数据日志已存至 logs/ 目录")
    print(f"模型权重已存至 saved_models/ 目录")
    print("#"*60)

if __name__ == "__main__":
    main()
