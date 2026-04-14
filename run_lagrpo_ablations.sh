#!/bin/bash
# run_lagrpo_ablations.sh — LAGRPO Iron Triangle Ablation experiments (7B Version)
# -----------------------------------------------------------------------------
# Configuration: Group Size 16, Steps 150, Qwen-7B-Instruct
# Sequential runs: B0 -> B1 -> B2 -> B3 -> B4

set -e

# === 用户配置区 ===
# 如果你的本地 7B 模型路径不同，请修改此处
MODEL="Qwen/Qwen2.5-7B-Instruct"
G=8
STEPS=150
SFT="saved_models/sft_final"
# =================

echo "=================================================="
echo "  LAGRPO Iron Triangle Ablation Suite (7B)"
echo "  $(date)"
echo "--------------------------------------------------"
echo "Model: $MODEL"
echo "Steps: $STEPS | Group Size: $G"
echo "=================================================="

# 1. B0: Baseline GRPO
echo ""
echo "── Step 1: B0 Baseline (Normal GRPO) ──"
python train_grpo.py --model-name "$MODEL" --ablation B0 --group-size $G --max-steps $STEPS --sft-path $SFT --exp-id ablation_B0 --batch-size 1 --accum-steps 4

# 2. B1: + Length-Aware Reward (Space Dimension)
echo ""
echo "── Step 2: B1 Space (Length-Aware Advantage) ──"
python train_grpo.py --model-name "$MODEL" --ablation B1 --group-size $G --max-steps $STEPS --sft-path $SFT --exp-id ablation_B1 --batch-size 1 --accum-steps 4

# 3. B2: + Step Annealing (Time Dimension)
echo ""
echo "── Step 3: B2 Time (Reward Annealing) ──"
python train_grpo.py --model-name "$MODEL" --ablation B2 --group-size $G --max-steps $STEPS --sft-path $SFT --exp-id ablation_B2 --batch-size 1 --accum-steps 4

# 4. B3: + Adv Clipping (Variance Dimension)
echo ""
echo "── Step 4: B3 Variance (Advantage Clipping) ──"
python train_grpo.py --model-name "$MODEL" --ablation B3 --group-size $G --max-steps $STEPS --sft-path $SFT --exp-id ablation_B3 --batch-size 1 --accum-steps 4

# 5. B4: Full LAGRPO
echo ""
echo "── Step 5: B4 Full LAGRPO (Iron Triangle) ──"
python train_grpo.py --model-name "$MODEL" --ablation B4 --group-size $G --max-steps $STEPS --sft-path $SFT --exp-id ablation_B4 --batch-size 1 --accum-steps 4

# Final Evaluation
echo ""
echo "── Step 6: Final Generalization Evaluation ──"
python evaluate.py --model-name "$MODEL" --n-samples 100 --batch-size 16

# Professional Plotting
echo ""
echo "── Step 7: Generating Professional Plots ──"
python plot_exploration_efficiency_v2.py

echo ""
echo "=================================================="
echo "  ✅ Ablation Suite Code Ready!"
echo "  Logs: logs/grpo_ablation_B*_G16_metrics.csv"
echo "  Plots: plots/exploration_efficiency.png"
echo "=================================================="
