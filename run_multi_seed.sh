#!/bin/bash
# ==============================================================================
# run_multi_seed.sh — 多随机种子自动化实验流水线 (NeurIPS/ICLR 级基准测试)
# 
# 一键运行多个 Seed 的数据生成、SFT、PPO、GRPO 训练及最终的带误差带的图表评估。
# ==============================================================================

# 设置使用的种子列表
SEEDS=(42 2026 999)
N_SAMPLES_EVAL=500  # 测试时的截断题目数量

# 核心训练参数 (适配 24G 显存)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"  # 可替换为您本地使用的模型路径
SFT_BATCH_SIZE=8
PPO_BATCH_SIZE=2
PPO_ACCUM=8
GRPO_G=16
GRPO_BATCH_SIZE=1
GRPO_ACCUM=16

echo "============================================================"
echo "🚀 启动多种子训练流水线"
echo "使用的种子: ${SEEDS[*]}"
echo "============================================================"

# 新建汇总存储各个 Seed 结果的总目录
FINAL_EVAL_MODELS_DIR="saved_models_multi"
mkdir -p $FINAL_EVAL_MODELS_DIR

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "============================================================"
    echo "🌱 开始执行 Seed: $SEED"
    echo "============================================================"
    
    # 每个随机种子都应该拥有独立的数据子集、日志和保存路径
    SEED_DATA_DIR="data_seed_$SEED"
    SEED_LOG_DIR="logs_seed_$SEED"
    
    mkdir -p $SEED_DATA_DIR
    mkdir -p $SEED_LOG_DIR
    
    # ---------------------------------------------------------
    # 1. 扩充量级的数据生成 (包含 N=6 和 大验证集)
    # ---------------------------------------------------------
    echo "[1/4] 生成 N=3,4,5,6 的数据集 (Seed $SEED) ..."
    python data_gen_multi.py \
        --n 3 4 5 6 \
        --max-per-n 1500 \
        --sft \
        --sft-per-n 300 \
        --seed $SEED \
        --test-size 500 \
        --output-dir $SEED_DATA_DIR

    # ---------------------------------------------------------
    # 2. SFT 训练基线
    # ---------------------------------------------------------
    echo "[2/4] 基于 $MODEL_ID 进行 SFT 训练 (Seed $SEED) ..."
    # 注意：此处您可能需要根据新版 Qwen2.5 调整 --model-id 路径
    python train_sft.py \
        --model-id $MODEL_ID \
        --train-file $SEED_DATA_DIR/sft_train.csv \
        --batch-size $SFT_BATCH_SIZE \
        --epochs 2 \
        --lr 5e-5 \
        --output-dir saved_models/seed_${SEED}/sft_final

    # ================= 注意 =======================
    # 为了防止单张卡上 PPO / GRPO 崩溃导致流水线停止，
    # 强烈建议在此处对 train_ppo 包装 try-catch 或 bash || true
    # 但由于这是标准自动化脚本，保留顺序运行逻辑：
    
    # ---------------------------------------------------------
    # 3. PPO 强化学习训练
    # ---------------------------------------------------------
    echo "[3/4] 运行 PPO 训练 (Seed $SEED) ..."
    python train_ppo.py \
        --model-id $MODEL_ID \
        --sft-model-path saved_models/seed_${SEED}/sft_final \
        --train-file $SEED_DATA_DIR/train.csv \
        --max-steps 100 \
        --batch-size $PPO_BATCH_SIZE \
        --grad-accum $PPO_ACCUM \
        --lr 1e-6 \
        --max-new-tokens 512 \
        --target-kl 0.05 \
        --output-dir saved_models/seed_${SEED}/ppo_final

    # ---------------------------------------------------------
    # 4. GRPO 训练对比 (使用默认 G=16)
    # ---------------------------------------------------------
    echo "[4/4] 运行 GRPO 训练 (Seed $SEED, G=$GRPO_G) ..."
    python train_grpo.py \
        --model-id $MODEL_ID \
        --sft-model-path saved_models/seed_${SEED}/sft_final \
        --train-file $SEED_DATA_DIR/train.csv \
        --max-steps 100 \
        --group-size $GRPO_G \
        --batch-size $GRPO_BATCH_SIZE \
        --grad-accum $GRPO_ACCUM \
        --lr 1e-6 \
        --max-new-tokens 512 \
        --beta 0.04 \
        --output-dir saved_models/seed_${SEED}/grpo_G${GRPO_G}_final

    echo "✅ Seed $SEED 训练阶段全部完成。"
done

echo ""
echo "============================================================"
echo "📊 多 Seed 最终评估与作图"
echo "============================================================"

# 此步骤将遍历刚才生成的所有 seed_*/ 目录
python evaluate.py \
    --test-file data_seed_${SEEDS[0]}/test.jsonl \
    --n-samples $N_SAMPLES_EVAL \
    --output-dir logs_multi_seed

# 调用学术版图表脚本绘制带 std 方差区域的 Diff-Success 曲线
python eval_plots.py \
    --all \
    --log-dir logs_multi_seed \
    --output-dir plots_multi_seed

# 同时也生成经典版供选择
python eval_plots_classic.py \
    --all \
    --log-dir logs_multi_seed \
    --output-dir plots_multi_seed

echo "============================================================"
echo "🎉 全流程完毕！"
echo "图表位置："
echo "  - plots_multi_seed/diff_success_curve.png (重头戏：难度vs成功率带误差带展示图)"
echo "  - plots_multi_seed/eval_summary.png"
echo "============================================================"
