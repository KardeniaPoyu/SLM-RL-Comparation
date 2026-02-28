#!/bin/bash
# ============================================================
# run_all.sh — 24点 PPO vs GRPO 实验一键运行脚本
# 适用于 AutoDL / 云 GPU 服务器 (A100/V100/RTX 4090 等)
# ============================================================
#
# 用法:
#   chmod +x run_all.sh
#   ./run_all.sh              # 完整流程
#   ./run_all.sh --skip-sft   # 跳过 SFT（已有预训练权重时）
#   ./run_all.sh --only-grpo  # 仅运行 GRPO 实验
#
# 预计耗时 (单 A100 40GB):
#   数据生成:  ~5 min
#   SFT:       ~15 min
#   PPO:       ~2 hrs
#   GRPO x4:   ~6 hrs (G=8,16,32,64 消融)
#
# ============================================================

set -e  # 遇到错误立即停止

# ── 默认参数 ──
SKIP_SFT=false
ONLY_GRPO=false
ONLY_PPO=false
SKIP_DATA=false

# 解析命令行参数
for arg in "$@"; do
    case $arg in
        --skip-sft)  SKIP_SFT=true ;;
        --only-grpo) ONLY_GRPO=true ;;
        --only-ppo)  ONLY_PPO=true ;;
        --skip-data) SKIP_DATA=true ;;
    esac
done

echo "=================================================="
echo "  24-Point PPO vs GRPO Experiment"
echo "  $(date)"
echo "=================================================="
echo ""

# GPU 信息
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')" 2>/dev/null || echo "No GPU detected!"
echo ""

# ========================================
# Step 0: 安装依赖
# ========================================
echo "── Step 0: 检查依赖 ──"
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -c "import numpy; v=numpy.__version__; print(f'numpy={v}'); assert v.startswith('1.26'), f'需要 numpy 1.26.x, 当前 {v}'"
echo "✅ 依赖就绪"
echo ""

# ========================================
# Step 1: 数据生成
# ========================================
if [ "$SKIP_DATA" = false ]; then
    echo "── Step 1: 数据生成 (N=3,4,5) ──"
    python data_gen_multi.py --n 3 4 5 --sft --sft-per-n 200
    echo ""
fi

# ========================================
# Step 2: SFT 预热
# ========================================
if [ "$SKIP_SFT" = false ] && [ "$ONLY_GRPO" = false ] && [ "$ONLY_PPO" = false ]; then
    echo "── Step 2: SFT 预热训练 ──"
    python train_sft.py \
        --data data/sft_train.csv \
        --epochs 4 \
        --lr 2e-4 \
        --batch-size 2
    echo ""
fi

# ========================================
# Step 3: PPO 训练
# ========================================
if [ "$ONLY_GRPO" = false ]; then
    echo "── Step 3: PPO 训练 ──"
    python train_ppo.py \
        --lr 2e-6 \
        --batch-size 16 \
        --mini-batch-size 2 \
        --grad-accum-steps 8 \
        --init-kl-coef 0.05 \
        --adaptive-kl \
        --ppo-epochs 4 \
        --max-steps 200 \
        --save-every 10 \
        --log-layer-grads
    echo ""
fi

# ========================================
# Step 4: GRPO G 消融实验
# ========================================
if [ "$ONLY_PPO" = false ]; then
    echo "── Step 4: GRPO G 消融实验 ──"

    # G=8:  bs=4, accum=2, B_eff = 4*8*2 = 64
    echo "  → G=8"
    python train_grpo.py \
        --group-size 8 --batch-size 4 --accum-steps 2 \
        --lr 2e-6 --beta 0.04 \
        --max-steps 200 \
        --save-every 10 --log-layer-grads

    # G=16: bs=2, accum=2, B_eff = 2*16*2 = 64
    echo "  → G=16"
    python train_grpo.py \
        --group-size 16 --batch-size 2 --accum-steps 2 \
        --lr 2e-6 --beta 0.04 \
        --max-steps 200 \
        --save-every 10 --log-layer-grads

    # G=32: bs=1, accum=2, B_eff = 1*32*2 = 64
    echo "  → G=32"
    python train_grpo.py \
        --group-size 32 --batch-size 1 --accum-steps 2 \
        --lr 2e-6 --beta 0.04 \
        --max-steps 200 \
        --save-every 10 --log-layer-grads

    # G=64: bs=1, accum=1, B_eff = 1*64*1 = 64
    # (GRPO内部已实现了generation和KL的chunking，bs=1是最小安全限度)
    echo "  → G=64"
    python train_grpo.py \
        --group-size 64 --batch-size 1 --accum-steps 1 \
        --lr 2e-6 --beta 0.04 \
        --max-steps 200 \
        --save-every 10 --log-layer-grads

    echo ""
fi

# ========================================
# Step 5: 最终评估 (测试集上的泛化表现)
# ========================================
echo "── Step 5: 最终评估 (100 N=3 + 100 N=4 + 100 N=5 未见题) ──"
python evaluate.py \
    --test-file data/test.csv \
    --n-samples 100 \
    --batch-size 16
echo ""

# ========================================
# Step 6: 生成论文图表
# ========================================
echo "── Step 6: 生成论文图表 ──"
python eval_plots.py --all
echo ""

echo "=================================================="
echo "  ✅ 实验完成！"
echo "  📊 图表: plots/"
echo "  📝 日志: logs/"
echo "  📝 评估: logs/eval_summary.csv"
echo "  💾 模型: saved_models/"
echo "=================================================="
