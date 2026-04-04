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


import json

def plot_eval_summary(log_dir='logs', output_dir='plots'):
    """最终评估测试集成功率对比柱状图 (经典版)"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 尝试从 jsonl 搜集结果
    eval_files = glob.glob(os.path.join(log_dir, 'eval_*.jsonl'))
    if not eval_files:
        return
        
    records = []
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
    
    df_records['n_key'] = df_records['n_key'].apply(lambda x: f"N={x}" if str(x).isdigit() else str(x).upper())
    df_records['accuracy'] = df_records['accuracy'] * 100
    
    # pivot 使模型成为 index，N=... 成为 columns
    df_pivot = df_records.pivot(index='model', columns='n_key', values='accuracy')
    
    # 对 index 排序，PPO 放前面
    models = list(df_pivot.index)
    models.sort(key=lambda x: (0 if 'ppo' in x.lower() else 1, x))
    df_pivot = df_pivot.reindex(models)
    
    # 画图 (df_pivot.T 使得 X 轴是 Difficulty Level，不同颜色是 Model)
    ax = df_pivot.T.plot(kind='bar', figsize=(10, 6), colormap='tab10', edgecolor='black', alpha=0.85)
    
    ax.set_title('Evaluation Success Rate on Test Set (Zero-Shot) - Classic', fontsize=14, pad=15)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_xlabel('Difficulty Level', fontsize=12)
    
    # 动态 Y 轴
    max_acc = df_records['accuracy'].max()
    ax.set_ylim(0, max(max_acc * 1.2, 5.0))
    
    # 在柱子上添加具体数值标签
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=9)
    
    # 网格和图例
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 旋转 x 轴标签防止重叠
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eval_summary_classic.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 评测成功率汇总柱状图 (经典版) → {output_dir}/eval_summary_classic.png")


def plot_difficulty_curve(log_dir='logs', output_dir='plots'):
    """评测成功率难度曲线 (带多次 Seed 误差带, 经典版)"""
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
            
        base_model = model_full_name
        seed = 1
        
        if "_seed_" in model_full_name:
            parts = model_full_name.split("_seed_")
            base_model = parts[0]
            try:
                seed = int(parts[1])
            except ValueError:
                pass
                
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
    
    # 手动计算 mean 和 std 进行绘制
    agg_df = df.groupby(['Model', 'Difficulty (N)'])['Success Rate (%)'].agg(['mean', 'std']).reset_index()
    # 填充缺失值为0 (如果单个 seed std 会是 NaN)
    agg_df['std'] = agg_df['std'].fillna(0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = agg_df['Model'].unique().tolist()
    models.sort(key=lambda x: (0 if 'PPO' in x else 1, x))
    
    colors = plt.cm.tab10.colors
    
    for i, model in enumerate(models):
        m_df = agg_df[agg_df['Model'] == model].sort_values('Difficulty (N)')
        x = m_df['Difficulty (N)'].values
        y_mean = m_df['mean'].values
        y_std = m_df['std'].values
        
        color = colors[i % len(colors)]
        
        ax.plot(x, y_mean, marker='o', label=model, linewidth=2, color=color)
        if y_std.max() > 0:  # 只有包含多个 seed 有方差时才画填充带
            ax.fill_between(x, y_mean - y_std, y_mean + y_std, color=color, alpha=0.2)
    
    ax.set_title('Zero-Shot Success Rate vs. Difficulty (Mean ± 1 Std) - Classic', fontsize=14, pad=15)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_xlabel('Difficulty Level', fontsize=12)
    
    max_acc = df['Success Rate (%)'].max()
    # 让0附近的值也能清晰看出差别 (Symmetric Log Scale)
    ax.set_yscale('symlog', linthresh=1.0)
    import matplotlib.ticker as ticker
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_yticks([0, 1, 5, 10, 20, 50, 100])
    
    unique_n = sorted(df['Difficulty (N)'].unique())
    ax.set_xticks(unique_n)
    
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(title='Model', loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diff_success_curve_classic.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 难度-成功率曲线图(SymLog) (经典版) → {output_dir}/diff_success_curve_classic.png")

def plot_success_heatmap(log_dir='logs', output_dir='plots'):
    """生成高级别论文常见的 Success Rate Heatmap (模型 vs 难度) - 经典配色"""
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
    agg_df = df.groupby(['Model', 'Difficulty (N)'])['Success Rate'].mean().reset_index()
    
    pivot_df = agg_df.pivot(index='Model', columns='Difficulty (N)', values='Success Rate')
    
    models = list(pivot_df.index)
    models.sort(key=lambda x: (0 if 'PPO' in x else 1, x))
    pivot_df = pivot_df.reindex(models)
    
    fig, ax = plt.subplots(figsize=(8, max(4, len(models)*0.8)))
    
    import seaborn as sns
    sns.heatmap(
        pivot_df, 
        annot=True, 
        fmt=".1f", # 1位小数
        cmap="coolwarm", # 经典红蓝冷暖配色
        cbar_kws={'label': 'Success Rate (%)'},
        linewidths=1,
        linecolor='black',
        ax=ax
    )
    
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
    ax.set_title("Zero-Shot Success Rate Across Difficulty Levels (Classic)", pad=30, fontweight='bold', fontsize=14)
    ax.set_ylabel("Model Architecture", fontsize=12)
    ax.set_xlabel("Task Difficulty", fontsize=12)
    
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rate_heatmap_classic.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 模型难度热力图 (Heatmap) (经典版) → {output_dir}/success_rate_heatmap_classic.png")


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
        
    if args.all:
        plot_eval_summary(args.log_dir, args.output_dir)
        plot_difficulty_curve(args.log_dir, args.output_dir)
        plot_success_heatmap(args.log_dir, args.output_dir)

    print(f"\n📊 经典版所有图表已生成至 '{args.output_dir}/' 目录")


if __name__ == "__main__":
    main()