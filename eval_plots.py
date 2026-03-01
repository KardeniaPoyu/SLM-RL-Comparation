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
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无头模式（云服务器）
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 提升学术图表规范的全局设置
# ==========================================
# 1. 统一字体 (Times New Roman / Arial)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

# 2. 颜色与线型规范 (色盲友好)
sns.set_palette("colorblind")
COLORS = sns.color_palette("colorblind")
LINE_STYLES = ['-', '--', '-.', ':']

def get_style_for_label(label, all_labels):
    """为不同系列分配固定的颜色和线型组合"""
    if "PPO" in label:
        return COLORS[0], '-'  # PPO 始终为第一种颜色、实线
    
    # 获取所有的 G 值并排序，确保 G4, G8, G16 分配到固定的样式
    g_labels = [l for l in all_labels if "G=" in l or l.startswith("G")]
    # 按数值排序
    g_labels.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    try:
        idx = g_labels.index(label)
        # GRPO 变体使用第二、三、四种颜色
        return COLORS[idx + 1], LINE_STYLES[idx % len(LINE_STYLES)]
    except ValueError:
        return COLORS[-1], '-'


def load_and_process(filepath, smooth_alpha=0.2):
    """加载数据并计算指数平滑移动平均 (EMA)"""
    df = pd.read_csv(filepath)
    
    # 清理断点续训导致的重复 step，保留最后一次（最新的）记录
    if 'step' in df.columns:
        df = df.drop_duplicates(subset=['step'], keep='last')
        df = df.sort_values('step').reset_index(drop=True)
        
    # PPO的KL在csv中叫 kl_ref，统一重命名为 kl_div 以便作图对齐
    if 'kl_ref' in df.columns and 'kl_div' not in df.columns:
        df = df.rename(columns={'kl_ref': 'kl_div'})

    # 对所有数值列计算 EMA 平滑
    for col in df.select_dtypes(include=[np.number]).columns:
        if col != 'step':
            df[f'{col}_smooth'] = df[col].ewm(alpha=smooth_alpha, adjust=False).mean()
            
    return df


def create_comparison_plot(ax, dfs, col, title, ylabel, use_log=False):
    """
    双层绘制法:
    底层: 原始数据 (细线, 低透明度)
    表层: EMA 平滑曲线 (粗线, 实心)
    """
    all_labels = list(dfs.keys())
    
    for label, df in dfs.items():
        if col in df.columns:
            color, linestyle = get_style_for_label(label, all_labels)
            
            # 1. 绘制底层原始数据 (降噪处理)
            ax.plot(df['step'], df[col], color=color, alpha=0.25, linewidth=1.0)
            
            # 2. 绘制表层平滑曲线
            smooth_col = f'{col}_smooth' if f'{col}_smooth' in df.columns else col
            ax.plot(df['step'], df[smooth_col], label=label, 
                    color=color, linestyle=linestyle, linewidth=2.5)
            
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Update Steps')
    
    if use_log:
        ax.set_yscale('log')
        
    # 净化背景与坐标轴边缘 (Despining)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 弱化网格线: 仅保留水平网格线，且颜色极浅
    ax.grid(axis='y', linestyle='-', alpha=0.3, color='#E0E0E0')
    ax.grid(axis='x', visible=False)


def sort_labels(dfs_dict):
    """确保图例排序符合逻辑递进 (e.g. G=4, G=8, G=16)"""
    def sort_key(k):
        # 提取数字进行排序
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

    fig, axes = plt.subplots(3, 2, figsize=(15, 16))
    fig.suptitle('PPO vs. GRPO Comparison (Arithmetic-24)', fontweight='bold', y=0.98)

    create_comparison_plot(axes[0, 0], dfs, 'success_rate', 'Sample Efficiency', 'Success Rate')
    create_comparison_plot(axes[0, 1], dfs, 'kl_div', 'Policy Drift (KL Divergence)', 'KL Divergence')
    create_comparison_plot(axes[1, 0], dfs, 'grad_norm', 'Gradient Norm Stability', 'L2 Norm', use_log=True)
    create_comparison_plot(axes[1, 1], dfs, 'grad_second_moment', 'Gradient Second Moment', 'Second Moment', use_log=True)
    create_comparison_plot(axes[2, 0], dfs, 'mean_response_length', 'Exploration Trajectory', 'Avg Generated Tokens')
    
    # 隐藏多余的子图
    axes[2, 1].axis('off') 
    
    # 全局统一图例: 提取第一张子图的 handles 和 labels，置于大图上方居中
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94),
               ncol=len(labels), frameon=False, fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.92]) # 留出顶部空间给 title 和 legend
    plt.savefig(os.path.join(output_dir, 'ppo_vs_grpo.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ PPO vs GRPO 对比图 → {output_dir}/ppo_vs_grpo.png")


def plot_g_ablation(log_dir='logs', output_dir='plots'):
    """GRPO G∈{8,16,32,64} 消融实验图"""
    os.makedirs(output_dir, exist_ok=True)

    dfs = {}
    
    # 获取所有的消融实验数据
    for f in glob.glob(os.path.join(log_dir, 'grpo_G*_metrics.csv')):
        basename = os.path.basename(f)
        label = basename.replace('_metrics.csv', '').replace('grpo_', '')
        dfs[label] = load_and_process(f)
        
    if not dfs:
        print("No GRPO ablation files found.")
        return
        
    dfs = sort_labels(dfs)

    fig, axes = plt.subplots(3, 2, figsize=(15, 16))
    fig.suptitle('GRPO Group Size Ablation', fontweight='bold', y=0.98)

    create_comparison_plot(axes[0, 0], dfs, 'success_rate', 'Sample Efficiency vs. Group Size', 'Success Rate')
    create_comparison_plot(axes[0, 1], dfs, 'adv_std', 'Advantage Estimation Variance', 'Reward Std')
    create_comparison_plot(axes[1, 0], dfs, 'grad_norm', 'Gradient Norm vs. Group Size', 'L2 Norm', use_log=True)
    create_comparison_plot(axes[1, 1], dfs, 'grad_second_moment', 'Gradient Variance vs. Group Size', 'Second Moment', use_log=True)
    create_comparison_plot(axes[2, 0], dfs, 'mean_response_length', 'Exploration Trajectory vs. Group Size', 'Avg Generated Tokens')
    
    # 隐藏多余的子图
    axes[2, 1].axis('off') 

    # 全局统一图例
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94),
               ncol=len(labels), frameon=False, fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(os.path.join(output_dir, 'grpo_g_ablation.png'), dpi=300, bbox_inches='tight')
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