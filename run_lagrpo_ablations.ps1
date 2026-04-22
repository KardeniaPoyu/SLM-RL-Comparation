# run_lagrpo_ablations.ps1 — LAGRPO Iron Triangle Ablation experiments (7B Version)
# -----------------------------------------------------------------------------
# Configuration: Group Size 16, Steps 150, Qwen-7B-Instruct

$ErrorActionPreference = "Continue"

# === 用户配置区 ===
# 如果你的本地 7B 模型路径不同，请修改此处
$MODEL = "Qwen/Qwen2.5-7B-Instruct"
$G = 16
$STEPS = 150
$SFT = "saved_models/sft_final"
# =================

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  LAGRPO Iron Triangle Ablation Suite (7B)" -ForegroundColor Cyan
Write-Host "  $(Get-Date)" -ForegroundColor Cyan
Write-Host "--------------------------------------------------"
Write-Host "Model: $MODEL"
Write-Host "Steps: $STEPS | Group Size: $G"
Write-Host "=================================================="

# Training Loop
$Ablations = @(
    @("B0", "Baseline (Normal GRPO)"),
    @("B1", "Space (Length-Aware Advantage)"),
    @("B2", "Time (Reward Annealing)"),
    @("B3", "Variance (Advantage Clipping)"),
    @("B4", "Full LAGRPO (Iron Triangle)")
)

foreach ($Abl in $Ablations) {
    $ID = $Abl[0]
    $Name = $Abl[1]
    
    Write-Host "`n── Step $($ID): $($Name) ──" -ForegroundColor Yellow
    # 7B 模型建议增加 --accum-steps 4 以保证显存安全和梯度平稳
    python train_grpo.py --model-name "$MODEL" --ablation $ID --group-size $G --max-steps $STEPS --sft-path $SFT --exp-id "ablation_$ID" --batch-size 1 --accum-steps 4
}

# Final Evaluation
Write-Host "`n── Step Eval: Final Generalization Evaluation ──" -ForegroundColor Yellow
python evaluate.py --model-name "$MODEL" --n-samples 100 --batch-size 16

# Professional Plotting
Write-Host "`n── Step Plot: Generating Professional Plots ──" -ForegroundColor Yellow
python plot_exploration_efficiency_v2.py

Write-Host "`n==================================================" -ForegroundColor Green
Write-Host "  ✅ Ablation Suite Code Ready!" -ForegroundColor Green
Write-Host "  Logs: logs/grpo_ablation_B*_G16_metrics.csv"
Write-Host "  Plots: plots/exploration_efficiency.png"
Write-Host "=================================================="
