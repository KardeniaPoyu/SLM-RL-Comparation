#!/bin/bash
# run_beta_sweep.sh — LAGRPO Beta Sweep Server Entry Point
# 适用于服务器/Linux 环境，支持后台运行与 GPU 监控。

# ── 配置区 ──
STEPS=100
G=16
LOG_DIR="logs/beta_sweep"
OUTPUT_DIR="saved_models/beta_sweep"
MODEL="Qwen/Qwen2.5-0.5B-Instruct"

echo "=================================================="
echo "  LAGRPO Beta Sensitivity Sweep (Server Mode)"
echo "  Steps: $STEPS | Group Size: $G"
echo "  $(date)"
echo "=================================================="

# 检查虚拟环境
if [ -d ".venv_reasoning" ]; then
    PYTHON_EXE=".venv_reasoning/bin/python"
elif [ -d ".venv" ]; then
    PYTHON_EXE=".venv/bin/python"
else
    PYTHON_EXE="python"
fi

echo "Using Python: $(which $PYTHON_EXE)"

# 运行扫参
# 如果你需要在后台运行，请使用: setsid ./run_beta_sweep.sh > sweep.log 2>&1 &
$PYTHON_EXE run_beta_sweep.py \
    --max-steps $STEPS \
    --group-size $G \
    --model-name "$MODEL" \
    --log-dir "$LOG_DIR" \
    --output-dir "$OUTPUT_DIR"

# 实验结束后自动运行分析
echo ""
echo "── Step: Running Post-Sweep Analysis ──"
$PYTHON_EXE analyze_beta_sensitivity.py

echo ""
echo "=================================================="
echo "  ✅ Beta 消融实验全部完成！"
echo "  📊 最终曲线: plots/beta_sensitivity/beta_three_regimes.png"
echo "  📝 数据汇总: plots/beta_sensitivity/beta_summary.csv"
echo "=================================================="
