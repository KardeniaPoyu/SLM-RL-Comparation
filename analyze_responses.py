"""
分析和对比 GRPO 与 PPO 的 Response 日志
"""

import os
import re
from collections import defaultdict

def parse_response_file(file_path):
    """
    解析 response 日志文件
    返回: {step: [responses]}
    """
    responses = defaultdict(list)
    
    if not os.path.exists(file_path):
        print(f"文件未找到: {file_path}")
        return responses
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取每个 Step/Update 的所有 response
    pattern = r"(Step|Update) (\d+) - Sample (\d+):\n(.+?)(?=\n-{80})"
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        step_num = int(match.group(2))
        response_text = match.group(4).strip()
        responses[step_num].append(response_text)
    
    return responses

def analyze_responses(responses, method_name):
    """分析 response 的统计信息"""
    print(f"\n{'='*60}")
    print(f"📊 {method_name} Response 分析")
    print(f"{'='*60}")
    
    if not responses:
        print("❌ 没有找到任何 response")
        return
    
    total_steps = len(responses)
    total_responses = sum(len(rs) for rs in responses.values())
    
    print(f"✓ 总 Step 数: {total_steps}")
    print(f"✓ 总 Response 数: {total_responses}")
    
    # 计算字符统计
    all_lengths = []
    all_tokens = 0
    
    for step_responses in responses.values():
        for resp in step_responses:
            all_lengths.append(len(resp))
            # 简单 token 估计 (实际应该用 tokenizer)
            all_tokens += len(resp.split())
    
    avg_length = sum(all_lengths) / len(all_lengths) if all_lengths else 0
    max_length = max(all_lengths) if all_lengths else 0
    min_length = min(all_lengths) if all_lengths else 0
    
    print(f"\n📝 Response 长度统计:")
    print(f"  • 平均长度: {avg_length:.0f} 字符")
    print(f"  • 最大长度: {max_length} 字符")
    print(f"  • 最小长度: {min_length} 字符")
    print(f"  • 估计总 Token 数: {all_tokens:,}")
    
    # 检查 response 质量指标
    think_tags = sum(1 for rs in responses.values() for r in rs if '<think>' in r or '</think>' in r)
    print(f"\n🎯 质量指标:")
    print(f"  • 包含 <think> 标签的 Response: {think_tags} / {total_responses}")
    
    # 显示前 5 个 step 的 sample
    print(f"\n📌 前 5 个 Step 的 Sample 1:")
    for step in sorted(responses.keys())[:5]:
        if responses[step]:
            sample = responses[step][0][:100].replace('\n', ' ') + "..."
            print(f"  Step {step}: {sample}")

def compare_methods(grpo_path, ppo_path):
    """对比 GRPO 和 PPO 的 response"""
    print("\n" + "="*60)
    print("🔄 GRPO 与 PPO 对比")
    print("="*60)
    
    grpo_responses = parse_response_file(grpo_path)
    ppo_responses = parse_response_file(ppo_path)
    
    if grpo_responses and ppo_responses:
        # 比较共同 step
        common_steps = set(grpo_responses.keys()) & set(ppo_responses.keys())
        print(f"\n✓ 共有 {len(common_steps)} 个共同的 Step")
        
        # 样本对比
        if common_steps:
            first_common_step = min(common_steps)
            print(f"\n💭 Step {first_common_step} 的 Sample 1 对比:")
            
            grpo_sample = grpo_responses[first_common_step][0] if grpo_responses[first_common_step] else "N/A"
            ppo_sample = ppo_responses[first_common_step][0] if ppo_responses[first_common_step] else "N/A"
            
            print(f"\n【GRPO】")
            print(grpo_sample[:200] + "..." if len(grpo_sample) > 200 else grpo_sample)
            
            print(f"\n【PPO】")
            print(ppo_sample[:200] + "..." if len(ppo_sample) > 200 else ppo_sample)

def create_summary_report(grpo_path, ppo_path, output_path='logs/response_summary.txt'):
    """创建汇总报告"""
    grpo_responses = parse_response_file(grpo_path)
    ppo_responses = parse_response_file(ppo_path)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("GRPO vs PPO Response 对比分析\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("【GRPO 统计】\n")
        f.write(f"  Step 数: {len(grpo_responses)}\n")
        f.write(f"  总 Response: {sum(len(rs) for rs in grpo_responses.values())}\n")
        
        f.write("\n【PPO 统计】\n")
        f.write(f"  Step 数: {len(ppo_responses)}\n")
        f.write(f"  总 Response: {sum(len(rs) for rs in ppo_responses.values())}\n")
        
        # 样本展示
        f.write("\n" + "=" * 80 + "\n")
        f.write("样本对比 (Step 0)\n")
        f.write("=" * 80 + "\n\n")
        
        if 0 in grpo_responses and grpo_responses[0]:
            f.write("【GRPO Step 0 Sample 1】\n")
            f.write(grpo_responses[0][0] + "\n\n")
        
        if 0 in ppo_responses and ppo_responses[0]:
            f.write("【PPO Update 0 Sample 1】\n")
            f.write(ppo_responses[0][0] + "\n\n")
    
    print(f"\n✓ 汇总报告已保存至: {output_path}")

if __name__ == "__main__":
    grpo_log = 'logs/grpo_responses.txt'
    ppo_log = 'logs/ppo_responses.txt'
    
    print("🚀 开始分析 Response 日志...\n")
    
    # 分析 GRPO
    grpo_responses = parse_response_file(grpo_log)
    analyze_responses(grpo_responses, "GRPO")
    
    # 分析 PPO
    ppo_responses = parse_response_file(ppo_log)
    analyze_responses(ppo_responses, "PPO")
    
    # 对比
    compare_methods(grpo_log, ppo_log)
    
    # 生成汇总报告
    create_summary_report(grpo_log, ppo_log)
    
    print("\n✅ 分析完成！")
