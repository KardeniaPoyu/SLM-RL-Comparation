#!/usr/bin/env python
"""
Response 日志查看器 - 快速浏览生成的 response
"""

import os
import sys

def view_response_file(file_path, limit=None):
    """查看 response 文件的内容"""
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return
    
    print(f"\n{'='*70}")
    print(f"📄 {os.path.basename(file_path)}")
    print(f"{'='*70}\n")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if limit:
        lines = lines[:limit]
        print(f"(显示前 {limit} 行)\n")
    
    for line in lines:
        print(line.rstrip())
    
    total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))
    print(f"\n{'='*70}")
    print(f"总行数: {total_lines}")
    print(f"文件大小: {os.path.getsize(file_path) / 1024:.1f} KB")

def compare_first_responses():
    """对比 GRPO 和 PPO 的第一个 response"""
    
    grpo_file = 'logs/grpo_responses.txt'
    ppo_file = 'logs/ppo_responses.txt'
    
    grpo_exists = os.path.exists(grpo_file)
    ppo_exists = os.path.exists(ppo_file)
    
    if not grpo_exists and not ppo_exists:
        print("❌ GRPO 和 PPO 的 response 日志都不存在")
        return
    
    print(f"\n{'='*70}")
    print("🔍 第一个 Response 对比")
    print(f"{'='*70}\n")
    
    # 提取第一个 response
    def get_first_response(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 跳过头部注释
        start = None
        for i, line in enumerate(lines):
            if "Step" in line or "Update" in line:
                start = i
                break
        
        if start is None:
            return None
        
        # 收集内容直到分隔线
        content = []
        for i in range(start + 1, len(lines)):
            if lines[i].startswith('-'):
                break
            content.append(lines[i])
        
        return ''.join(content).strip()
    
    if grpo_exists:
        grpo_resp = get_first_response(grpo_file)
        if grpo_resp:
            print("【GRPO 第一个生成的 Response】")
            print(grpo_resp)
            print("\n" + "-"*70 + "\n")
    
    if ppo_exists:
        ppo_resp = get_first_response(ppo_file)
        if ppo_resp:
            print("【PPO 第一个生成的 Response】")
            print(ppo_resp)
    
    print("\n" + "="*70)

def main():
    """主菜单"""
    
    while True:
        print("\n" + "="*70)
        print("👀 Response 日志查看器")
        print("="*70)
        print("\n请选择:")
        print("  1. 查看 GRPO Response 日志 (完整)")
        print("  2. 查看 GRPO Response 日志 (前50行)")
        print("  3. 查看 PPO Response 日志 (完整)")
        print("  4. 查看 PPO Response 日志 (前50行)")
        print("  5. 对比 GRPO 和 PPO 的第一个 Response")
        print("  6. 查看 Response 分析摘要")
        print("  0. 退出")
        
        choice = input("\n请输入选择: ").strip()
        
        if choice == '1':
            view_response_file('logs/grpo_responses.txt')
        elif choice == '2':
            view_response_file('logs/grpo_responses.txt', limit=50)
        elif choice == '3':
            view_response_file('logs/ppo_responses.txt')
        elif choice == '4':
            view_response_file('logs/ppo_responses.txt', limit=50)
        elif choice == '5':
            compare_first_responses()
        elif choice == '6':
            view_response_file('logs/response_summary.txt', limit=100)
        elif choice == '0':
            print("\n👋 再见！")
            break
        else:
            print("❌ 无效选择")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ 已退出")
