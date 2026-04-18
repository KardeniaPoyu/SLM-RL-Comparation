#!/bin/bash
# ==============================================================================
# run_integrated_suite.sh — 24点强化学习综合实验流水线
# 整合内容：PPO + GRPO (G=4,8,16) + LAGRPO (B1,B2,B3,B4)
# 数据对齐：$N=3, 4, 5$ (SFT, 训练, 测试)
# ==============================================================================

set -e
export OMP_NUM_THREADS=1  # Silence libgomp warnings and optimize CPU contention

# === 用户配置区 ===
MODEL="Qwen/Qwen2.5-7B-Instruct"
G_BASE=8                      # LAGRPO 消融的基础组大小，同时也是 B0
STEPS=150                     # 每个 RL 任务的训练步数
SFT_DIR="saved_models/sft_final"
LOG_DIR="logs/integrated_exp"
PLOT_DIR="plots/integrated_exp"
# =================

# 命令行参数解析
SKIP_DATA=false
SKIP_SFT=false
SKIP_PPO=false
SKIP_GRPO_G=false
SKIP_LAGRPO=false

for arg in "$@"; do
    case $arg in
        --skip-data) SKIP_DATA=true ;;
        --skip-sft)  SKIP_SFT=true ;;
        --skip-ppo)  SKIP_PPO=true ;;
        --skip-grpo-g) SKIP_GRPO_G=true ;;
        --skip-lagrpo) SKIP_LAGRPO=true ;;
    esac
done

mkdir -p $LOG_DIR
mkdir -p $PLOT_DIR

echo "=================================================="
echo "  🚀 启动综合实验流水线 (N=3,4,5 ALIGNED)"
echo "  Model: $MODEL"
echo "  Steps: $STEPS"
echo "  Time:  $(date)"
echo "=================================================="
echo ""

# ---------------------------------------------------------
# Step 0: 检查并更新依赖 (针对 AutoDL 库版本不兼容修复)
# ---------------------------------------------------------
echo "── [0/6] 检查并修复库兼容性 (TRL/Transformers) ──"
# 强制安装稳定版本以适配现有代码 (避免 trl 1.x 的重大 Break)
pip install trl==0.12.1 transformers==4.46.3 peft==0.12.0 accelerate==0.34.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
echo "✅ 环境恢复稳定 (TRL 0.12.1 + Transformers 4.46.3)"

# ---------------------------------------------------------
# Step 1: 核心数据生成 (对齐 N=3,4,5)
# ---------------------------------------------------------
if [ "$SKIP_DATA" = false ]; then
    echo ""
    echo "── [1/6] 正在生成 N=3,4,5 对齐数据集 ──"
    python data_gen_multi.py --n 3 4 5 --sft --sft-per-n 300
fi

# ---------------------------------------------------------
# Step 2: SFT 预热训练
# ---------------------------------------------------------
if [ "$SKIP_SFT" = false ]; then
    echo ""
    echo "── [2/6] 正在执行 SFT 预热 (N=3,4,5) ──"
    python train_sft.py \
        --model-name "$MODEL" \
        --data data/sft_train.csv \
        --output-dir "$SFT_DIR" \
        --epochs 4 \
        --batch-size 2 \
        --grad-accum 8
fi

# ---------------------------------------------------------
# Step 3: PPO 基线对比
# ---------------------------------------------------------
if [ "$SKIP_PPO" = false ]; then
    echo ""
    echo "── [3/6] 正在运行 PPO 基线训练 ──"
    python train_ppo.py \
        --model-name "$MODEL" \
        --sft-path "$SFT_DIR" \
        --data-file data/train.csv \
        --lr 2e-6 \
        --max-steps "$STEPS" \
        --batch-size 16 \
        --mini-batch-size 1 \
        --grad-accum-steps 16 \
        --target-kl 0.05 \
        --save-every 50 \
        --output-dir saved_models/ppo_final \
        --log-dir "$LOG_DIR"
fi

# ---------------------------------------------------------
# Step 4: GRPO Group Size 消融 (G=4, 8, 16)
# ---------------------------------------------------------
if [ "$SKIP_GRPO_G" = false ]; then
    echo ""
    echo "── [4/6] 正在运行 GRPO 组大小消融 (G=4, 8, 16) ──"
    
    for G in 4 8 16; do
        EXP_ID="grpo_G${G}"
        echo "  → 启动 G=$G 训练..."
        python train_grpo.py \
            --model-name "$MODEL" \
            --sft-path "$SFT_DIR" \
            --group-size "$G" \
            --max-steps "$STEPS" \
            --batch-size 2 \
            --accum-steps 1 \
            --exp-id "$EXP_ID" \
            --save-every 50 \
            --log-dir "$LOG_DIR" \
            --output-dir saved_models/grpo_G${G}_final
    done
fi

# ---------------------------------------------------------
# Step 5: LAGRPO 特性消融 (基于 G=8, B1-B4)
# ---------------------------------------------------------
if [ "$SKIP_LAGRPO" = false ]; then
    echo ""
    echo "── [5/6] 正在运行 LAGRPO 特性消融 (B1 - B4, G=$G_BASE) ──"
    # 注：B0 已在 Step 4 的 G=8 中完成
    
    for ABLATION in B1 B2 B3 B4; do
        echo "  → 启动 LAGRPO $ABLATION 训练..."
        python train_grpo.py \
            --model-name "$MODEL" \
            --sft-path "$SFT_DIR" \
            --ablation "$ABLATION" \
            --group-size "$G_BASE" \
            --max-steps "$STEPS" \
            --batch-size 2 \
            --accum-steps 1 \
            --exp-id "lagrpo_${ABLATION}" \
            --save-every 50 \
            --log-dir "$LOG_DIR" \
            --output-dir saved_models/lagrpo_${ABLATION}_final
    done
fi

# ---------------------------------------------------------
# Step 6: 统一评估与作图
# ---------------------------------------------------------
echo ""
echo "── [6/6] 正在对所有模型进行对齐评估 ──"
python evaluate.py \
    --model-name "$MODEL" \
    --test-file data/test.jsonl \
    --n-samples 100 \
    --batch-size 16 \
    --output-dir "$LOG_DIR/eval"

echo ""
echo "── 生成对比图表 ──"
# 调用脚本生成 PPO vs GRPO vs LAGRPO 的综合曲线
python eval_plots.py --all --log-dir "$LOG_DIR" --output-dir "$PLOT_DIR"

echo "=================================================="
echo "  ✅ 综合实验流水线全部完成！"
echo "  📊 图表位置: $PLOT_DIR"
echo "  💾 模型位置: saved_models/"
echo "  📝 日志位置: $LOG_DIR"
echo "=================================================="
