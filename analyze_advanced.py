"""
高级: Response 与 Metrics 关联分析
"""

import os
import csv
import re
from collections import defaultdict

def load_metrics(csv_file):
    """加载训练指标"""
    metrics = {}
    
    if not os.path.exists(csv_file):
        print(f"⚠️  文件未找到: {csv_file}")
        return metrics
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row.get('step', 0))
            metrics[step] = row
    
    return metrics

def parse_responses_with_metrics(response_file, metrics_file=None):
    """
    加载 response 并关联对应的 metrics
    返回: {step: {'responses': [...], 'metrics': {...}}}
    """
    data = defaultdict(lambda: {'responses': [], 'metrics': None})
    
    if not os.path.exists(response_file):
        print(f"⚠️  文件未找到: {response_file}")
        return data
    
    # 先加载 metrics
    metrics = {}
    if metrics_file and os.path.exists(metrics_file):
        metrics = load_metrics(metrics_file)
    
    # 解析 response
    with open(response_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pattern = r"(Step|Update) (\d+) - Sample (\d+):\n(.+?)(?=\n-{80})"
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        step_num = int(match.group(2))
        response_text = match.group(4).strip()
        data[step_num]['responses'].append(response_text)
        
        # 关联对应的 metrics
        if step_num in metrics:
            data[step_num]['metrics'] = metrics[step_num]
    
    return data

def analyze_with_metrics(method_name, response_file, metrics_file):
    """
    分析 response 与对应的训练指标
    """
    print(f"\n{'='*70}")
    print(f"📊 {method_name} 详细分析")
    print(f"{'='*70}")
    
    data = parse_responses_with_metrics(response_file, metrics_file)
    
    if not data:
        print("❌ 没有找到数据")
        return
    
    print(f"\n✓ 总共处理了 {len(data)} 个 Step/Update")
    
    # 分析成功率与生成长度的关系
    success_length_pairs = []
    
    for step_num in sorted(data.keys()):
        step_data = data[step_num]
        
        # 获取该 step 的成功率（如果有）
        success_rate = None
        if step_data['metrics']:
            success_rate = float(step_data['metrics'].get('success_rate', 0))
        
        # 计算该 step 所有 response 的平均长度
        avg_length = sum(len(r) for r in step_data['responses']) / len(step_data['responses']) if step_data['responses'] else 0
        
        if success_rate is not None:
            success_length_pairs.append((step_num, success_rate, avg_length))
    
    # 按成功率分组统计
    if success_length_pairs:
        print(f"\n📈 成功率与生成长度关系 (前10个Step):")
        print(f"{'Step':<8} {'Success Rate':<15} {'Avg Length':<15} {'Observations':<20}")
        print("-" * 60)
        
        for step, succ_rate, avg_len in success_length_pairs[:10]:
            # 检查是否有思考过程
            has_thinking = sum(1 for r in data[step]['responses'] if '<think>' in r)
            obs = f"含思考: {has_thinking}/{len(data[step]['responses'])}"
            print(f"{step:<8} {succ_rate:<15.2%} {avg_len:<15.0f} {obs:<20}")
    
    # 整体统计
    print(f"\n📝 生成内容统计:")
    
    all_lengths = []
    all_thinking = 0
    all_responses = 0
    
    for step_data in data.values():
        for resp in step_data['responses']:
            all_lengths.append(len(resp))
            if '<think>' in resp or '</think>' in resp:
                all_thinking += 1
            all_responses += 1
    
    if all_lengths:
        avg_length = sum(all_lengths) / len(all_lengths)
        print(f"  • 平均长度: {avg_length:.0f} 字符")
        print(f"  • 最大长度: {max(all_lengths)} 字符")
        print(f"  • 最小长度: {min(all_lengths)} 字符")
        print(f"  • 包含思考标签: {all_thinking}/{all_responses} ({100*all_thinking/all_responses:.1f}%)")

def compare_success_correlation(grpo_response, grpo_metrics, ppo_response, ppo_metrics):
    """
    对比 GRPO 和 PPO：成功率与生成长度的相关性
    """
    print(f"\n{'='*70}")
    print(f"🔍 成功率与生成长度相关性分析")
    print(f"{'='*70}")
    
    # 获取两个方法的数据
    grpo_data = parse_responses_with_metrics(grpo_response, grpo_metrics)
    ppo_data = parse_responses_with_metrics(ppo_response, ppo_metrics)
    
    # 计算相关性指标
    def calc_correlation(data):
        pairs = []
        for step_data in data.values():
            if step_data['metrics']:
                success_rate = float(step_data['metrics'].get('success_rate', 0))
                avg_length = sum(len(r) for r in step_data['responses']) / len(step_data['responses'])
                pairs.append((success_rate, avg_length))
        
        if len(pairs) < 2:
            return None
        
        # 计算皮尔逊相关系数的简单版本
        mean_sr = sum(p[0] for p in pairs) / len(pairs)
        mean_len = sum(p[1] for p in pairs) / len(pairs)
        
        numerator = sum((p[0] - mean_sr) * (p[1] - mean_len) for p in pairs)
        denom_sr = sum((p[0] - mean_sr) ** 2 for p in pairs)
        denom_len = sum((p[1] - mean_len) ** 2 for p in pairs)
        
        if denom_sr > 0 and denom_len > 0:
            correlation = numerator / (denom_sr * denom_len) ** 0.5
            return correlation, pairs
        return None
    
    grpo_corr = calc_correlation(grpo_data)
    ppo_corr = calc_correlation(ppo_data)
    
    if grpo_corr:
        corr_val, pairs = grpo_corr
        avg_sr = sum(p[0] for p in pairs) / len(pairs)
        print(f"\n✅ GRPO:")
        print(f"  • 成功率-长度相关系数: {corr_val:.3f}")
        print(f"  • 平均成功率: {avg_sr:.2%}")
    
    if ppo_corr:
        corr_val, pairs = ppo_corr
        avg_sr = sum(p[0] for p in pairs) / len(pairs)
        print(f"\n✅ PPO:")
        print(f"  • 成功率-长度相关系数: {corr_val:.3f}")
        print(f"  • 平均成功率: {avg_sr:.2%}")
    
    # 解读
    if grpo_corr and ppo_corr:
        print(f"\n💡 解读:")
        gr_sign = "正" if grpo_corr[0] > 0 else "负"
        pr_sign = "正" if ppo_corr[0] > 0 else "负"
        print(f"  • GRPO: 生成更长的回复通常会带来{gr_sign}的成功率变化")
        print(f"  • PPO: 生成更长的回复通常会带来{pr_sign}的成功率变化")

def export_high_quality_responses(response_file, metrics_file, output_file, threshold=0.8):
    """
    导出高质量 response（成功率 > threshold）
    """
    data = parse_responses_with_metrics(response_file, metrics_file)
    
    high_quality = []
    
    for step_num in sorted(data.keys()):
        step_data = data[step_num]
        if step_data['metrics']:
            success_rate = float(step_data['metrics'].get('success_rate', 0))
            if success_rate >= threshold:
                for resp in step_data['responses']:
                    high_quality.append((step_num, success_rate, resp))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"高质量 Response (成功率 >= {threshold:.0%})\n")
        f.write("=" * 80 + "\n\n")
        
        for step_num, success_rate, resp in high_quality:
            f.write(f"Step {step_num} (成功率: {success_rate:.2%}):\n")
            f.write(resp + "\n")
            f.write("-" * 80 + "\n\n")
    
    print(f"✓ 已导出 {len(high_quality)} 个高质量 response 至: {output_file}")

if __name__ == "__main__":
    print("🚀 开始高级分析...\n")
    
    grpo_response = 'logs/grpo_responses.txt'
    grpo_metrics = 'logs/grpo_metrics.csv'
    ppo_response = 'logs/ppo_responses.txt'
    ppo_metrics = 'logs/ppo_metrics.csv'
    
    # 分析 GRPO
    analyze_with_metrics("GRPO", grpo_response, grpo_metrics)
    
    # 分析 PPO
    analyze_with_metrics("PPO", ppo_response, ppo_metrics)
    
    # 对比相关性
    compare_success_correlation(grpo_response, grpo_metrics, ppo_response, ppo_metrics)
    
    # 导出高质量 response
    if os.path.exists(grpo_metrics):
        export_high_quality_responses(grpo_response, grpo_metrics, 'logs/grpo_high_quality.txt', threshold=0.75)
    if os.path.exists(ppo_metrics):
        export_high_quality_responses(ppo_response, ppo_metrics, 'logs/ppo_high_quality.txt', threshold=0.75)
    
    print("\n✅ 高级分析完成！")
