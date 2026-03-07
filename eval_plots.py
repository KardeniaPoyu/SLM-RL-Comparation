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


import json

def plot_eval_summary(log_dir='logs', output_dir='plots'):
    """最终评估测试集成功率对比柱状图"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 尝试从 jsonl 搜集结果
    eval_files = glob.glob(os.path.join(log_dir, 'eval_*.jsonl'))
    if not eval_files:
        return
        
    records = []
    # 从 eval_modelname_N.jsonl 中提取正确率
    # 格式可能如: eval_ppo_final_4.jsonl 或 eval_grpo_G16_final_all.jsonl
    for f in eval_files:
        basename = os.path.basename(f)
        # 移除 'eval_' 前缀和 '.jsonl' 后缀
        name_parts = basename.replace('eval_', '').replace('.jsonl', '').rsplit('_', 1)
        if len(name_parts) == 2:
            model_name, n_key = name_parts
        else:
            model_name = name_parts[0]
            n_key = "all"
            
        correct = 0
        total = 0
        with open(f, 'r', encoding='utf-8') as jf:
            for line in jf:
                try:
                    data = json.loads(line)
                    total += 1
                    if data.get('correct', False):
                        correct += 1
                except:
                    pass
        if total > 0:
            records.append({'model': model_name, 'n_key': n_key, 'accuracy': correct / total})

    if not records:
        return

    # 构建 DataFrame
    df_records = pd.DataFrame(records)
    
    # 重塑为 Metric, Accuracy, model 的形式以适配 grouped barplot
    df_records['Metric'] = df_records['n_key'].apply(lambda x: f"N={x}" if str(x).isdigit() else str(x).upper())
    df_records['Accuracy'] = df_records['accuracy'] * 100
    
    # 获取唯一的模型并排序
    all_models = df_records['model'].unique().tolist()
    all_models.sort(key=lambda x: (0 if 'ppo' in x.lower() else 1, x))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=df_records, x='Metric', y='Accuracy', hue='model', hue_order=all_models,
        ax=ax, palette='colorblind', edgecolor='white'
    )
    
    ax.set_title('Evaluation Success Rate on Test Set (Zero-Shot)', fontweight='bold', pad=20)
    ax.set_ylabel('Success Rate (%)')
    ax.set_xlabel('Difficulty Level')
    
    # 动态 Y 轴，如果最大值很低，不要强制到 100
    max_acc = df_records['Accuracy'].max()
    ax.set_ylim(0, max(max_acc * 1.2, 5.0))
    
    # 在柱子上添加具体数值标签
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=9, color='#333333')
    
    # 净化边缘
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='-', alpha=0.3, color='#E0E0E0')
    ax.grid(axis='x', visible=False)
    
    # 图例
    ax.legend(title='', loc='upper left', bbox_to_anchor=(1.05, 1), frameon=False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eval_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 评测成功率汇总柱状图 → {output_dir}/eval_summary.png")

def plot_difficulty_curve(log_dir='logs', output_dir='plots'):
    """评测成功率难度曲线 (带多次 Seed 误差带)"""
    os.makedirs(output_dir, exist_ok=True)
    
    eval_files = glob.glob(os.path.join(log_dir, 'eval_*.jsonl'))
    if not eval_files:
        return
        
    records = []
    for f in eval_files:
        basename = os.path.basename(f)
        # 移除前缀后缀
        raw_name = basename.replace('eval_', '').replace('.jsonl', '')
        
        # 提取 N 值 (最后的 _N 或 _all)
        name_parts = raw_name.rsplit('_', 1)
        if len(name_parts) == 2 and (name_parts[1].isdigit() or name_parts[1] == 'all'):
            model_full_name, n_key = name_parts
        else:
            model_full_name = raw_name
            n_key = "all"
            
        # 跳过 'all' 汇总，只画 N
        if n_key == "all":
            continue
            
        n_val = int(n_key)
            
        # 提取基础模型名和 seed
        # 格式可能是: ppo_final_seed_1 或 sft_final
        base_model = model_full_name
        seed = 1
        
        if "_seed_" in model_full_name:
            parts = model_full_name.split("_seed_")
            base_model = parts[0]
            try:
                seed = int(parts[1])
            except ValueError:
                pass
                
        # 提炼显示名称
        display_name = base_model.upper().replace('_FINAL', '')
        
        correct = 0
        total = 0
        with open(f, 'r', encoding='utf-8') as jf:
            for line in jf:
                try:
                    data = json.loads(line)
                    total += 1
                    if data.get('correct', False):
                        correct += 1
                except:
                    pass
                    
        if total > 0:
            records.append({
                'Model': display_name, 
                'Difficulty (N)': n_val, 
                'Seed': seed,
                'Success Rate (%)': (correct / total) * 100
            })

    if not records:
        return

    df = pd.DataFrame(records)
    
    # 按照 Difficulty 排序确保 X 轴正常
    df.sort_values('Difficulty (N)', inplace=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制折线图，Seaborn 会自动聚合具有相同 Model 和 Difficulty 的数据点，
    # 并使用 errorbar='sd' 绘制标准差阴影区域
    sns.lineplot(
        data=df, 
        x='Difficulty (N)', 
        y='Success Rate (%)', 
        hue='Model', 
        style='Model',
        markers=True, 
        dashes=False,
        errorbar='sd',  # 显示 standard deviation
        linewidth=2.5,
        markersize=10,
        palette='colorblind',
        ax=ax
    )
    
    ax.set_title('Zero-Shot Success Rate vs. Difficulty (Mean ± 1 Std)', fontweight='bold', pad=20)
    max_acc = df['Success Rate (%)'].max()
    # 让0附近的值也能清晰看出差别 (Symmetric Log Scale)
    # 比如 0, 1, 2, 10, 100 会有递进的间距
    ax.set_yscale('symlog', linthresh=1.0)
    import matplotlib.ticker as ticker
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_yticks([0, 1, 5, 10, 20, 50, 100])
    
    # 净化边缘和网格
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    ax.legend(title='', frameon=True, edgecolor='white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diff_success_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 难度-成功率曲线图(SymLog Scale) → {output_dir}/diff_success_curve.png")

def plot_success_heatmap(log_dir='logs', output_dir='plots'):
    """生成高级别论文常见的 Success Rate Heatmap (模型 vs 难度)"""
    os.makedirs(output_dir, exist_ok=True)
    
    eval_files = glob.glob(os.path.join(log_dir, 'eval_*.jsonl'))
    if not eval_files:
        return
        
    records = []
    for f in eval_files:
        basename = os.path.basename(f)
        raw_name = basename.replace('eval_', '').replace('.jsonl', '')
        
        name_parts = raw_name.rsplit('_', 1)
        if len(name_parts) == 2 and (name_parts[1].isdigit() or name_parts[1] == 'all'):
            model_full_name, n_key = name_parts
        else:
            model_full_name = raw_name
            n_key = "all"
            
        if n_key == "all":
            continue
            
        n_val = int(n_key)
        
        # 提炼显示名称 (丢弃 seed, 计算平均)
        base_model = model_full_name
        if "_seed_" in model_full_name:
            base_model = model_full_name.split("_seed_")[0]
            
        display_name = base_model.upper().replace('_FINAL', '')
        
        correct = 0
        total = 0
        with open(f, 'r', encoding='utf-8') as jf:
            for line in jf:
                try:
                    data = json.loads(line)
                    total += 1
                    if data.get('correct', False):
                        correct += 1
                except:
                    pass
                    
        if total > 0:
            records.append({
                'Model': display_name, 
                'Difficulty (N)': f"N={n_val}", 
                'Success Rate': (correct / total) * 100
            })

    if not records:
        return

    df = pd.DataFrame(records)
    # 取不同 seed 的平均值
    agg_df = df.groupby(['Model', 'Difficulty (N)'])['Success Rate'].mean().reset_index()
    
    # 转置为矩阵: 行=Model, 列=Difficulty
    pivot_df = agg_df.pivot(index='Model', columns='Difficulty (N)', values='Success Rate')
    
    # 排序
    models = list(pivot_df.index)
    models.sort(key=lambda x: (0 if 'PPO' in x else 1, x))
    pivot_df = pivot_df.reindex(models)
    
    fig, ax = plt.subplots(figsize=(8, max(4, len(models)*0.8)))
    
    # 画学术风热力图 (使用 mako 或 Blues 等渐变色)
    sns.heatmap(
        pivot_df, 
        annot=True, 
        fmt=".1f", # 1位小数
        cmap="YlGnBu", 
        cbar_kws={'label': 'Success Rate (%)'},
        linewidths=1,
        linecolor='white',
        ax=ax
    )
    
    # 将 X 轴刻度移到顶部，这在很多 Paper 中非常常见
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
    ax.set_title("Zero-Shot Success Rate Across Difficulty Levels", pad=30, fontweight='bold')
    ax.set_ylabel("Model Architecture")
    ax.set_xlabel("Task Difficulty")
    
    # 旋转Y轴文字避免默认居中导致的别扭感
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rate_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 模型难度热力图 (Heatmap) → {output_dir}/success_rate_heatmap.png")

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
        
    if args.all:
        plot_eval_summary(args.log_dir, args.output_dir)
        plot_difficulty_curve(args.log_dir, args.output_dir)
        plot_success_heatmap(args.log_dir, args.output_dir)

    print(f"\n📊 所有图表已生成至 '{args.output_dir}/' 目录")


if __name__ == "__main__":
    main()