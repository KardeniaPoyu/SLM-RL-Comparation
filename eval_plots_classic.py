import os
import glob
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无头模式（云服务器）
import matplotlib.pyplot as plt


def load_and_process(filepath, smooth_window=5):
    df = pd.read_csv(filepath)
    
    # 清理断点续训导致的重复 step，保留最后一次（最新的）记录
    if 'step' in df.columns:
        df = df.drop_duplicates(subset=['step'], keep='last')
        df = df.sort_values('step').reset_index(drop=True)
        
    # PPO的KL在csv中叫 kl_ref，统一重命名为 kl_div 以便作图对齐
    if 'kl_ref' in df.columns and 'kl_div' not in df.columns:
        df = df.rename(columns={'kl_ref': 'kl_div'})

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


def sort_labels(dfs_dict):
    """确保图例排序符合逻辑递进 (e.g. G=4, G=8, G=16)"""
    def sort_key(k):
        nums = [int(s) for s in k if s.isdigit()]
        num = nums[0] if nums else float('inf')
        return (0 if "PPO" in k else 1, num)
    
    return {k: dfs_dict[k] for k in sorted(dfs_dict.keys(), key=sort_key)}


def plot_ppo_vs_grpo(log_dir='logs', output_dir='plots'):
    """PPO vs GRPO 标准对比图 (2×2 子图)"""
    os.makedirs(output_dir, exist_ok=True)

    dfs = {}
    ppo_file = os.path.join(log_dir, 'ppo_metrics.csv')
    if os.path.exists(ppo_file):
        dfs['PPO (Critic)'] = load_and_process(ppo_file)

    # 优先匹配 GRPO_G16_metrics.csv 作为与 PPO 的对比
    grpo_g16_file = os.path.join(log_dir, 'grpo_G16_metrics.csv')
    grpo_g8_file = os.path.join(log_dir, 'grpo_G8_metrics.csv')
    
    if os.path.exists(grpo_g16_file):
        dfs['GRPO (G=16)'] = load_and_process(grpo_g16_file)
    elif os.path.exists(grpo_g8_file):
        dfs['GRPO (G=8)'] = load_and_process(grpo_g8_file)
    else:
        # 尝试匹配任何 GRPO_G*_metrics.csv
        for f in glob.glob(os.path.join(log_dir, 'grpo_G*_metrics.csv')):
            basename = os.path.basename(f)
            label = basename.replace('_metrics.csv', '').replace('grpo_', '')
            dfs[f'GRPO ({label})'] = load_and_process(f)
            break
        
    if not dfs:
        print("No metrics files found in", log_dir)
        return
        
    dfs = sort_labels(dfs)

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle('PPO vs GRPO Comparison (Arithmetic-24)', fontsize=16, fontweight='bold')

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
    create_comparison_plot(
        axes[2, 0], dfs, 'mean_response_length',
        'Exploration Trajectory (Response Tokens Length)',
        'Avg Generated Tokens'
    )
    axes[2, 1].axis('off') # 隐藏多余的子图

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ppo_vs_grpo_classic.png'), dpi=300)
    plt.close()
    print(f"✅ PPO vs GRPO 对比图 (经典版) → {output_dir}/ppo_vs_grpo_classic.png")


def plot_g_ablation(log_dir='logs', output_dir='plots'):
    """GRPO G∈{8,16,32,64} 消融实验图"""
    os.makedirs(output_dir, exist_ok=True)

    dfs = {}
    
    for f in glob.glob(os.path.join(log_dir, 'grpo_G*_metrics.csv')):
        basename = os.path.basename(f)
        label = basename.replace('_metrics.csv', '').replace('grpo_', '')
        dfs[label] = load_and_process(f)
        
    if not dfs:
        print("No GRPO ablation files found.")
        return
        
    dfs = sort_labels(dfs)

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle('GRPO Group Size Ablation',
                 fontsize=16, fontweight='bold')

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
    create_comparison_plot(
        axes[2, 0], dfs, 'mean_response_length',
        'Exploration Trajectory vs Group Size',
        'Avg Generated Tokens'
    )
    axes[2, 1].axis('off') # 隐藏多余的子图

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'grpo_g_ablation_classic.png'), dpi=300)
    plt.close()
    print(f"✅ G 消融对比图 (经典版) → {output_dir}/grpo_g_ablation_classic.png")


def main():
    parser = argparse.ArgumentParser(description="生成经典版论文图表")
    parser.add_argument("--ablation", action="store_true", help="生成 G 消融图")
    parser.add_argument("--all", action="store_true", help="生成所有图表")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--output-dir", type=str, default="plots")
    args = parser.parse_args()

    if args.all or not args.ablation:
        plot_ppo_vs_grpo(args.log_dir, args.output_dir)

    if args.all or args.ablation:
        plot_g_ablation(args.log_dir, args.output_dir)

    print(f"\n📊 经典版所有图表已生成至 '{args.output_dir}/' 目录")


if __name__ == "__main__":
    main()