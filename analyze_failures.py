import os
import glob
import json
import re
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict

def extract_numbers_from_expr(expr):
    """提取表达式中的所有数字"""
    nums = []
    # 使用正则匹配整数或小数
    for match in re.finditer(r'\b\d+(?:\.\d+)?\b', expr):
        try:
            val = float(match.group())
            # 如果是整数，转换为 int
            if val.is_integer():
                nums.append(int(val))
            else:
                nums.append(val)
        except:
            pass
    return sorted(nums)

def safe_eval(expr):
    """安全地评估算数表达式"""
    try:
        # 只允许基础运算
        allowed_names = {}
        code = compile(expr, "<string>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                raise NameError(f"Use of {name} not allowed")
        return eval(expr, {"__builtins__": {}}, allowed_names)
    except Exception:
        return None

def analyze_failure(nums, response):
    """
    分析单个失败的回复，返回由高到低优先级的错误原因分类。
    """
    N = len(nums)
    target_sorted = sorted([float(x) for x in nums])
    
    # 1. Format Error
    if "</think>" not in response:
        return "Format Error (Missing tag)"
    
    parts = response.split("</think>")
    think_block = parts[0]
    answer_block = parts[1].strip() if len(parts) > 1 else ""
    
    if not answer_block:
        return "Format Error (Empty Answer)"
        
    # 2. CoT Hallucination
    remain_matches = re.finditer(r'(?:剩余|还剩下|剩?下?的?数?是?:?)\s*[:：]?\s*\[(.*?)\]', think_block)
    for match in remain_matches:
        arr_str = match.group(1)
        arr_nums = extract_numbers_from_expr(arr_str)
        if len(arr_nums) >= N:
            return "CoT Hallucination (Invalid State)"
            
    # 接著分析最終表達式
    used_nums = extract_numbers_from_expr(answer_block)
    
    # 3. Number Count Mismatch
    if len(used_nums) != N:
        if len(used_nums) == 4 and N != 4:
            return "Overfitting to N=4 (Count=4)"
        return "Number Count Mismatch"
        
    # 4. Number Value Mismatch
    used_nums_sorted = sorted([float(x) for x in used_nums])
    if used_nums_sorted != target_sorted:
        return "Number Value Hallucination"
        
    # 5. Invalid Expression / Math Error
    result = safe_eval(answer_block)
    if result is None:
        return "Invalid Math Expression"
        
    # 6. Calculation Error
    if abs(result - 24) > 1e-5:
        return "Calculation Error (Result != 24)"
        
    return "Unknown Error / Tolerance Issue"

def main():
    log_dir = 'logs'
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)
    
    eval_files = glob.glob(os.path.join(log_dir, 'eval_*.jsonl'))
    if not eval_files:
        print("没有找到评估数据。")
        return
        
    all_failures = []
    
    for f in eval_files:
        basename = os.path.basename(f)
        raw_name = basename.replace('eval_', '').replace('.jsonl', '')
        name_parts = raw_name.rsplit('_', 1)
        if len(name_parts) == 2 and (name_parts[1].isdigit() or name_parts[1] == 'all'):
            model = name_parts[0]
            n_val = name_parts[1]
        else:
            model = raw_name
            n_val = 'all'
            
        if n_val == 'all':
            continue
            
        with open(f, 'r', encoding='utf-8') as jf:
            for line in jf:
                try:
                    data = json.loads(line)
                    if not data.get('correct', False):
                        nums_list = data.get('nums', [])
                        if isinstance(nums_list, str):
                            nums_list = [int(x.strip()) for x in nums_list.split(',')]
                            
                        reason = analyze_failure(nums_list, data.get('response', ''))
                        all_failures.append({
                            'Model': model,
                            'Difficulty': f"N={n_val}",
                            'Reason': reason
                        })
                except Exception as e:
                    pass

    if not all_failures:
        print("神奇，居然没有一次失败？")
        return
        
    df = pd.DataFrame(all_failures)
    
    # 学术绘图风格设置
    plt.style.use('seaborn-v0_8-paper')
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DeJavu Serif']
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['figure.dpi'] = 300
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # 透视表：行是模型/难度，列是失败原因
    pivot_df = pd.crosstab(
        index=[df['Model'], df['Difficulty']], 
        columns=df['Reason'], 
        normalize='index' # 归一化为百分比
    ) * 100
    
    # 按照 Model 排序，让 PPO 靠前
    pivot_df.sort_index(level=0, key=lambda idx: [0 if 'ppo' in x.lower() else 1 for x in idx], inplace=True)
    
    # 获取精美的调色板
    colors = sns.color_palette("Set2", n_colors=len(pivot_df.columns))
    
    # 画堆积柱状图
    pivot_df.plot(
        kind='bar', 
        stacked=True, 
        ax=ax,
        color=colors,
        edgecolor='black',
        linewidth=0.5,
        width=0.75
    )
    
    ax.set_title('Failure Mode Analysis Across Models and Difficulty Levels', pad=20, fontweight='bold')
    ax.set_xlabel('Model & Difficulty')
    ax.set_ylabel('Proportion of Failure Types (%)')
    ax.set_ylim(0, 100)
    
    # 优化X轴刻度标签
    labels = [f"{idx[0]} ({idx[1]})" for idx in pivot_df.index]
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=11)
    
    # 精简边框和网格
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.grid(axis='x', visible=False)
    
    # 图例
    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], legend_labels[::-1], title='Failure Category', 
              bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, edgecolor='black')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'failure_modes_analysis.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 失败原因分析图表已生成 -> {out_path}")
    
    # 打印一些具体的例子，帮助诊断
    print("\n🔍 典型的失败案例示例 (抽样):")
    for reason in df['Reason'].unique():
        sample = df[df['Reason'] == reason].head(1)
        if not sample.empty:
            print(f"- {reason} (出自: {sample.iloc[0]['Model']} {sample.iloc[0]['Difficulty']})")

if __name__ == '__main__':
    main()
