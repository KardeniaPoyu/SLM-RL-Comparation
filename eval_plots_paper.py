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

# 1.1 矢量图优化设置
plt.rcParams['pdf.fonttype'] = 42  # 确保 PDF 中的字体可嵌入且可编辑
plt.rcParams['svg.fonttype'] = 'none' # 确保 SVG 中的文字保持为文本而非路径

# ==========================================
# 2. 颜色与线型规范 (复刻 analyze_ablation_v2.py 的专业色板)
# ==========================================
ACADEMIC_COLORS = {
    "LAGRPO": "#C94040",   # 优先匹配：全功能版始终为赤红色
    "B4": "#C94040",       # 优先匹配：全功能版始终为赤红色
    "PPO": "#2C3E50",      # 深钢蓝 (Midnight Blue - 对比基准)
    "B0": "#7F7F7F",       # 中性灰 (Vanilla Base)
    "BASELINE": "#7F7F7F", 
    "G=4": "#5B8DB8",      # 莫兰迪蓝 (映射 B2)
    "B2": "#5B8DB8",       
    "G=8": "#6BAF6B",      # 鼠尾草绿 (映射 B3)
    "B3": "#6BAF6B",       
    "G=16": "#E07B39",     # 琥珀橙 (映射 B1)
    "B1": "#E07B39",       
}

LINE_STYLES = {
    "PPO": "--",          # 基准线使用虚线
    "B0": ":",            
    "DEFAULT": "-"        # 其他主曲线使用实线
}

def get_style_for_label(label, all_labels=None):
    """
    为不同系列分配全局统一的颜色。
    基于关键字匹配，确保跨文件一致性。
    """
    target_color = "#95A5A6" # Default Muted Gray
    target_ls = LINE_STYLES["DEFAULT"]
    
    label_upper = label.upper()
    
    # 1. 匹配颜色
    for key, color in ACADEMIC_COLORS.items():
        if key in label_upper:
            target_color = color
            break
            
    # 2. 匹配线型
    for key, ls in LINE_STYLES.items():
        if key in label_upper:
            target_ls = ls
            break
            
    return target_color, target_ls


def load_and_process(filepath, smooth_alpha=0.2, max_steps=None):
    """加载数据并计算指数平滑移动平均 (EMA)"""
    df = pd.read_csv(filepath)
    
    # 清理断点续训导致的重复 step，保留最后一次（最新的）记录
    if 'step' in df.columns:
        df = df.drop_duplicates(subset=['step'], keep='last')
        df = df.sort_values('step').reset_index(drop=True)
        
        # 截断步骤 (如果指定)
        if max_steps is not None:
            df = df[df['step'] <= max_steps].reset_index(drop=True)
        
    # PPO的KL在csv中叫 kl_ref，统一重命名为 kl_div 以便作图对齐
    if 'kl_ref' in df.columns and 'kl_div' not in df.columns:
        df = df.rename(columns={'kl_ref': 'kl_div'})

    # 计算探索效率 eta = (SR / Length) * 1000
    if 'success_rate' in df.columns and 'mean_response_length' in df.columns:
        df['eta'] = (df['success_rate'] / (df['mean_response_length'] + 1e-5)) * 1000

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
            
            # 2. 绘制表层平滑曲线 (采用 v2 的 alpha 与线宽)
            smooth_col = f'{col}_smooth' if f'{col}_smooth' in df.columns else col
            lw = 2.8 if ("LAGRPO" in label or "B4" in label) else 1.8
            ax.plot(df['step'], df[smooth_col], label=label, 
                    color=color, linestyle=linestyle, linewidth=lw, alpha=0.93)
            
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Update Steps')
    
    if use_log:
        ax.set_yscale('log')
        
    # 净化背景与坐标轴边缘 (Despining)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 弱化网格线: 采用 v2 版超细虚线网格
    ax.grid(True, linestyle='--', linewidth=0.45, alpha=0.55)
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
    ppo_file_new = os.path.join(log_dir, 'ppo_metrics_new.csv')
    ppo_file_old = os.path.join(log_dir, 'ppo_metrics.csv')
    
    if os.path.exists(ppo_file_new):
        print(f"Loading NEW PPO metrics (truncated at 90): {ppo_file_new}")
        dfs['PPO (Critic)'] = load_and_process(ppo_file_new, max_steps=90)
    elif os.path.exists(ppo_file_old):
        print(f"Loading OLD PPO metrics: {ppo_file_old}")
        dfs['PPO (Critic)'] = load_and_process(ppo_file_old)

    # 修改：优先匹配 G=8 作为基准，以便与 LAGRPO (G=8) 对齐对比
    grpo_g8_file = os.path.join(log_dir, 'grpo_G8_metrics.csv')
    grpo_g16_file = os.path.join(log_dir, 'grpo_G16_metrics.csv')
    
    if os.path.exists(grpo_g8_file):
        print(f"Loading GRPO (G=8) for comparison (truncated at 90)")
        dfs['GRPO (G=8)'] = load_and_process(grpo_g8_file, max_steps=90)
    elif os.path.exists(grpo_g16_file):
        print(f"Loading GRPO (G=16) for comparison (truncated at 90)")
        dfs['GRPO (G=16)'] = load_and_process(grpo_g16_file, max_steps=90)
    else:
        for f in glob.glob(os.path.join(log_dir, 'grpo_G*_metrics.csv')):
            basename = os.path.basename(f)
            label = basename.replace('_metrics.csv', '').replace('grpo_', '')
            print(f"Loading GRPO ({label}) for comparison (truncated at 90)")
            dfs[f'GRPO ({label})'] = load_and_process(f, max_steps=90)
            break

    # 显式加入 LAGRPO (G=8)
    lagrpo_file = os.path.join(log_dir, 'grpo_B4_G8_metrics.csv')
    if os.path.exists(lagrpo_file):
        print(f"Loading LAGRPO (G=8) for comparison (truncated at 90)")
        dfs['LAGRPO (G=8)'] = load_and_process(lagrpo_file, max_steps=90)
        
    if not dfs:
        print("No metrics files found in", log_dir)
        return
        
    dfs = sort_labels(dfs)

    fig, axes = plt.subplots(2, 3, figsize=(22, 11))
    fig.suptitle('PPO vs. GRPO Scaling & Efficiency Analysis (Arithmetic-24)', fontweight='bold', y=0.98, fontsize=20)

    # 第一行: 核心性能与成本
    create_comparison_plot(axes[0, 0], dfs, 'success_rate', '(a) Sample Efficiency (Success Rate)', 'Success Rate (%)')
    create_comparison_plot(axes[0, 1], dfs, 'mean_response_length', '(b) Reasoning Length Cost', 'Mean Tokens')
    create_comparison_plot(axes[0, 2], dfs, 'eta', '(c) Efficiency Index ($\eta$)', '$\eta$ Index')

    # 第二行: 稳定性与差异化
    create_comparison_plot(axes[1, 0], dfs, 'kl_div', '(d) Policy Drift (KL Divergence)', 'KL Divergence')
    create_comparison_plot(axes[1, 1], dfs, 'grad_norm', '(e) Gradient Stability (L2 Norm)', 'L2 Norm', use_log=True)
    create_comparison_plot(axes[1, 2], dfs, 'grad_second_moment', '(f) Gradient Variance', 'Second Moment', use_log=True)
    
    # 全局统一图例 (移动到标题下方)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.93),
               ncol=len(labels), frameon=True, shadow=True, fontsize=15)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(os.path.join(output_dir, 'ppo_vs_grpo_2x3_master.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'ppo_vs_grpo_2x3_master.svg'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'ppo_vs_grpo_2x3_master.pdf'), bbox_inches='tight')
    plt.close()
    print(f"DONE: PPO vs GRPO 2x3 Master Figure -> {output_dir}/ppo_vs_grpo_2x3_master.svg")


def plot_g_ablation(log_dir='logs', output_dir='plots'):
    """GRPO G∈{4,8,16,32,64} 消融实验图 (2x3 顶刊组合大图)"""
    os.makedirs(output_dir, exist_ok=True)

    dfs = {}
    
    # 获取所有的消融实验数据
    for f in glob.glob(os.path.join(log_dir, 'grpo_G*_metrics.csv')):
        basename = os.path.basename(f)
        label = basename.replace('_metrics.csv', '').replace('grpo_', '')
        dfs[f"G={label[1:] if label.startswith('G') else label}"] = load_and_process(f)

    # 显式加入 LAGRPO (G=8) 作为消融参照
    lagrpo_file = os.path.join(log_dir, 'grpo_B4_G8_metrics.csv')
    if os.path.exists(lagrpo_file):
        dfs["LAGRPO (G=8)"] = load_and_process(lagrpo_file)
        
    if not dfs:
        print("No GRPO ablation files found.")
        return
        
    dfs = sort_labels(dfs)

    fig, axes = plt.subplots(2, 3, figsize=(22, 11))
    fig.suptitle('GRPO Group Size (G) Ablation: Scaling Behaviors (EBS=16/64)', fontweight='bold', y=0.98, fontsize=20)

    # 第一行：性能与效率
    create_comparison_plot(axes[0, 0], dfs, 'success_rate', '(a) Sample Efficiency (Success Rate)', 'Success Rate (%)')
    create_comparison_plot(axes[0, 1], dfs, 'mean_response_length', '(b) Reasoning Tokens vs. G', 'Mean Tokens')
    create_comparison_plot(axes[0, 2], dfs, 'eta', '(c) Group Exploration Efficiency ($\eta$)', '$\eta$ Index')

    # 第二行：内部动态与稳定性
    create_comparison_plot(axes[1, 0], dfs, 'adv_std', '(d) Advantage Estimation Variance', 'Reward Std')
    create_comparison_plot(axes[1, 1], dfs, 'kl_div', '(e) Policy Constraint (KL Divergence)', 'KL Divergence')
    create_comparison_plot(axes[1, 2], dfs, 'grad_norm', '(f) Gradient Stability (L2 Norm)', 'L2 Norm', use_log=True)
    
    # 全局统一图例 (移动到标题下方)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.93),
               ncol=len(labels), frameon=True, shadow=True, fontsize=15)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(os.path.join(output_dir, 'grpo_g_ablation_2x3_master.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'grpo_g_ablation_2x3_master.svg'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'grpo_g_ablation_2x3_master.pdf'), bbox_inches='tight')
    plt.close()
    print(f"DONE: G Ablation 2x3 Master Figure -> {output_dir}/grpo_g_ablation_2x3_master.svg")

def plot_lagrpo_ablation(log_dir='logs', output_dir='plots'):
    """LAGRPO 铁三角机制消融实验图 (2x3 顶刊组合大图)"""
    os.makedirs(output_dir, exist_ok=True)

    dfs = {}
    files = {
        "Baseline GRPO (B0)": "grpo_ablation_B0_G8_metrics.csv",
        "+Length Aware (B1)": "grpo_ablation_B1_G8_metrics.csv",
        "+Annealing (B2)": "grpo_ablation_B2_G8_metrics.csv",
        "+Adv Clipping (B3)": "grpo_ablation_B3_G8_metrics.csv",
        "LAGRPO Full (B4)": "grpo_ablation_B4_FINAL_G8_metrics.csv"
    }

    for label, filename in files.items():
        f = os.path.join(log_dir, filename)
        if os.path.exists(f):
            dfs[label] = load_and_process(f)
        
    if not dfs:
        print("No LAGRPO ablation files found in", log_dir)
        return
        
    dfs = sort_labels(dfs)
        
    fig, axes = plt.subplots(2, 3, figsize=(22, 11))
    fig.suptitle('LAGRPO Mechanisms Ablation: Performance, Cost & Stability (EBS=64)', fontweight='bold', y=0.98, fontsize=20)

    # 第一行: 核心表现
    create_comparison_plot(axes[0, 0], dfs, 'success_rate', '(a) Success Rate Convergence', 'Success Rate (%)')
    create_comparison_plot(axes[0, 1], dfs, 'mean_response_length', '(b) Reasoning Trajectory Length', 'Mean Tokens')
    create_comparison_plot(axes[0, 2], dfs, 'eta', '(c) Exploration Efficiency ($\eta$)', '$\eta$ Index')

    # 第二行: 内部动态
    create_comparison_plot(axes[1, 0], dfs, 'mean_advantage', '(d) Advantage Distribution Bias', 'Mean Advantage')
    create_comparison_plot(axes[1, 1], dfs, 'grad_norm', '(e) Gradient Stability (L2 Norm)', 'L2 Norm', use_log=True)
    create_comparison_plot(axes[1, 2], dfs, 'grad_second_moment', '(f) Gradient Variance', 'Second Moment', use_log=True)

    # 全局统一图例 (移动到标题下方)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.93),
               ncol=len(labels), frameon=True, shadow=True, fontsize=15)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(os.path.join(output_dir, 'lagrpo_ablation_2x3_master.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'lagrpo_ablation_2x3_master.svg'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'lagrpo_ablation_2x3_master.pdf'), bbox_inches='tight')
    plt.close()
    print(f"DONE: LAGRPO 2x3 Master Figure -> {output_dir}/lagrpo_ablation_2x3_master.svg")



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
    plt.savefig(os.path.join(output_dir, 'eval_summary.svg'), bbox_inches='tight')
    plt.close()
    print(f"DONE: Evaluation Summary Bar Chart -> {output_dir}/eval_summary.svg")

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
    plt.savefig(os.path.join(output_dir, 'diff_success_curve.svg'), bbox_inches='tight')
    plt.close()
    print(f"DONE: Difficulty Success Curve -> {output_dir}/diff_success_curve.svg")

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
    plt.savefig(os.path.join(output_dir, 'success_rate_heatmap.svg'), bbox_inches='tight')
    plt.close()
    print(f"DONE: Success Rate Heatmap -> {output_dir}/success_rate_heatmap.svg")

def main():
    parser = argparse.ArgumentParser(description="生成论文图表")
    parser.add_argument("--ablation", action="store_true", help="生成 G 消融图")
    parser.add_argument("--lagrpo", action="store_true", help="生成 LAGRPO 消融图")
    parser.add_argument("--all", action="store_true", help="生成所有图表")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--output-dir", type=str, default="plots")
    args = parser.parse_args()

    if args.all or (not args.ablation and not args.lagrpo):
        plot_ppo_vs_grpo(args.log_dir, args.output_dir)

    if args.all or args.ablation:
        plot_g_ablation(args.log_dir, args.output_dir)
        
    if args.all or args.lagrpo:
        plot_lagrpo_ablation(args.log_dir, args.output_dir)
        
    if args.all:
        plot_eval_summary(args.log_dir, args.output_dir)
        plot_difficulty_curve(args.log_dir, args.output_dir)
        plot_success_heatmap(args.log_dir, args.output_dir)

    print(f"\nAll plots have been generated to '{args.output_dir}/' directory")

if __name__ == "__main__":
    main()