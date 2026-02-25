"""
eval_plots.py — 论文作图脚本
支持 PPO vs GRPO 对比 + GRPO G 消融实验可视化

用法:
    python eval_plots.py                   # PPO vs GRPO 标准对比
    python eval_plots.py --ablation        # GRPO G∈{8,16,32,64} 消融图
"""

import os
import glob
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无头模式（云服务器）
import matplotlib.pyplot as plt


def load_and_process(filepath, smooth_window=5):
    df = pd.read_csv(filepath)
    if 'success_rate' in df.columns:
        df['success_rate_smooth'] = df['success_rate'].rolling(
            window=smooth_window, min_periods=1
        ).mean()
    return df


def create_comparison_plot(ax, dfs, col, title, ylabel, use_log=False):
    """在子图 ax 上绘制多条曲线"""
    for label, df in dfs.items():
        if col in df.columns:
            ax.plot(df['step'], df[col], label=label, alpha=0.8, linewidth=1.5)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel('Update Steps', fontsize=10)
    if use_log:
        ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)


def plot_ppo_vs_grpo(log_dir='logs', output_dir='plots'):
    """PPO vs GRPO 标准对比图 (2×2 子图)"""
    os.makedirs(output_dir, exist_ok=True)

    dfs = {}
    ppo_file = os.path.join(log_dir, 'ppo_metrics.csv')
    if os.path.exists(ppo_file):
        dfs['PPO (Critic)'] = load_and_process(ppo_file)

    # 支持旧格式 grpo_metrics.csv 和新格式 grpo_G32_metrics.csv
    grpo_file = os.path.join(log_dir, 'grpo_metrics.csv')
    grpo_g32_file = os.path.join(log_dir, 'grpo_G32_metrics.csv')
    if os.path.exists(grpo_g32_file):
        dfs['GRPO (G=32)'] = load_and_process(grpo_g32_file)
    elif os.path.exists(grpo_file):
        dfs['GRPO (G=32)'] = load_and_process(grpo_file)

    if not dfs:
        print("No metrics files found in", log_dir)
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PPO vs GRPO Comparison (Arithmetic-24)', fontsize=14, fontweight='bold')

    create_comparison_plot(
        axes[0, 0], dfs, 'success_rate_smooth',
        'Sample Efficiency', 'Success Rate (MA)'
    )
    create_comparison_plot(
        axes[0, 1], dfs, 'kl_div',
        'Policy Drift (KL Divergence)', 'KL Divergence'
    )
    create_comparison_plot(
        axes[1, 0], dfs, 'grad_norm',
        'Gradient Norm Stability', 'L2 Norm', use_log=True
    )
    create_comparison_plot(
        axes[1, 1], dfs, 'grad_second_moment',
        'Gradient Second Moment (Variance Proxy)',
        'Second Moment', use_log=True
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ppo_vs_grpo.png'), dpi=300)
    plt.close()
    print(f"✅ PPO vs GRPO 对比图 → {output_dir}/ppo_vs_grpo.png")


def plot_g_ablation(log_dir='logs', output_dir='plots'):
    """GRPO G∈{8,16,32,64} 消融实验图"""
    os.makedirs(output_dir, exist_ok=True)

    dfs = {}
    for G in [8, 16, 32, 64]:
        filepath = os.path.join(log_dir, f'grpo_G{G}_metrics.csv')
        if os.path.exists(filepath):
            dfs[f'G={G}'] = load_and_process(filepath)

    if not dfs:
        # 尝试 glob 匹配
        for f in glob.glob(os.path.join(log_dir, 'grpo_G*_metrics.csv')):
            basename = os.path.basename(f)
            label = basename.replace('_metrics.csv', '').replace('grpo_', '')
            dfs[label] = load_and_process(f)

    if not dfs:
        print("No GRPO ablation files found.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GRPO Group Size Ablation (G ∈ {8, 16, 32, 64})',
                 fontsize=14, fontweight='bold')

    create_comparison_plot(
        axes[0, 0], dfs, 'success_rate_smooth',
        'Sample Efficiency vs Group Size', 'Success Rate (MA)'
    )
    create_comparison_plot(
        axes[0, 1], dfs, 'adv_std',
        'Advantage Estimation Variance', 'Reward Std'
    )
    create_comparison_plot(
        axes[1, 0], dfs, 'grad_norm',
        'Gradient Norm vs Group Size', 'L2 Norm', use_log=True
    )
    create_comparison_plot(
        axes[1, 1], dfs, 'grad_second_moment',
        'Gradient Variance vs Group Size',
        'Second Moment', use_log=True
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'grpo_g_ablation.png'), dpi=300)
    plt.close()
    print(f"✅ G 消融对比图 → {output_dir}/grpo_g_ablation.png")


def main():
    parser = argparse.ArgumentParser(description="生成论文图表")
    parser.add_argument("--ablation", action="store_true", help="生成 G 消融图")
    parser.add_argument("--all", action="store_true", help="生成所有图表")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--output-dir", type=str, default="plots")
    args = parser.parse_args()

    if args.all or not args.ablation:
        plot_ppo_vs_grpo(args.log_dir, args.output_dir)

    if args.all or args.ablation:
        plot_g_ablation(args.log_dir, args.output_dir)

    print(f"\n📊 所有图表已生成至 '{args.output_dir}/' 目录")


if __name__ == "__main__":
    main()