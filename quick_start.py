#!/usr/bin/env python
"""
快速开始脚本：一键运行训练和分析
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_grpo_training():
    """运行 GRPO 训练"""
    print("\n" + "="*70)
    print("🚀 启动 GRPO 训练...")
    print("="*70)
    
    try:
        result = subprocess.run(
            ["python", "train_grpo.py"],
            capture_output=False
        )
        if result.returncode == 0:
            print("✅ GRPO 训练完成")
            return True
        else:
            print("❌ GRPO 训练失败")
            return False
    except Exception as e:
        print(f"❌ 运行 GRPO 失败: {e}")
        return False

def run_ppo_training():
    """运行 PPO 训练"""
    print("\n" + "="*70)
    print("🚀 启动 PPO 训练...")
    print("="*70)
    
    try:
        result = subprocess.run(
            ["python", "train_ppo.py"],
            capture_output=False
        )
        if result.returncode == 0:
            print("✅ PPO 训练完成")
            return True
        else:
            print("❌ PPO 训练失败")
            return False
    except Exception as e:
        print(f"❌ 运行 PPO 失败: {e}")
        return False

def analyze_responses():
    """分析 response 日志"""
    print("\n" + "="*70)
    print("📊 分析 Response 日志...")
    print("="*70)
    
    try:
        result = subprocess.run(
            ["python", "analyze_responses.py"],
            capture_output=False
        )
        if result.returncode == 0:
            print("✅ Response 分析完成")
            return True
        else:
            print("❌ Response 分析失败")
            return False
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return False

def analyze_advanced():
    """高级分析"""
    print("\n" + "="*70)
    print("🔬 执行高级分析...")
    print("="*70)
    
    try:
        result = subprocess.run(
            ["python", "analyze_advanced.py"],
            capture_output=False
        )
        if result.returncode == 0:
            print("✅ 高级分析完成")
            return True
        else:
            print("❌ 高级分析失败")
            return False
    except Exception as e:
        print(f"❌ 高级分析失败: {e}")
        return False

def check_logs():
    """检查生成的日志文件"""
    print("\n" + "="*70)
    print("📁 日志文件检查")
    print("="*70)
    
    log_files = [
        'logs/grpo_metrics.csv',
        'logs/grpo_responses.txt',
        'logs/ppo_metrics.csv',
        'logs/ppo_responses.txt',
        'logs/response_summary.txt',
        'logs/grpo_high_quality.txt',
        'logs/ppo_high_quality.txt',
    ]
    
    print("\n已生成的文件:")
    for log_file in log_files:
        if os.path.exists(log_file):
            size = os.path.getsize(log_file) / 1024  # KB
            print(f"  ✅ {log_file:<35} ({size:>8.1f} KB)")
        else:
            print(f"  ⚠️  {log_file:<35} (未生成)")

def show_menu():
    """显示菜单"""
    print("\n" + "="*70)
    print("🎯 GRPO vs PPO 训练与分析工具")
    print("="*70)
    print("\n请选择操作:")
    print("  1. 运行 GRPO 训练")
    print("  2. 运行 PPO 训练")
    print("  3. 运行 GRPO + PPO 对比训练")
    print("  4. 分析 Response 日志")
    print("  5. 执行高级分析")
    print("  6. 检查日志文件")
    print("  7. 查看说明文档")
    print("  0. 退出")
    print("  q. 退出")
    print("\n" + "-"*70)

def main():
    """主函数"""
    print("\n🎬 欢迎使用 GRPO vs PPO 训练与分析工具\n")
    
    while True:
        show_menu()
        choice = input("请输入选择 (0-7): ").strip().lower()
        
        if choice == '1':
            run_grpo_training()
        elif choice == '2':
            run_ppo_training()
        elif choice == '3':
            print("\n开始对比训练: 先运行 GRPO，再运行 PPO")
            grpo_ok = run_grpo_training()
            time.sleep(2)
            if grpo_ok:
                ppo_ok = run_ppo_training()
                if ppo_ok:
                    print("\n✅ 两个训练都完成，开始分析...")
                    time.sleep(2)
                    analyze_responses()
        elif choice == '4':
            analyze_responses()
        elif choice == '5':
            analyze_advanced()
        elif choice == '6':
            check_logs()
        elif choice == '7':
            print("\n📖 查看 OPTIMIZATION.md 获取详细说明\n")
            if os.path.exists('OPTIMIZATION.md'):
                with open('OPTIMIZATION.md', 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(content)
            else:
                print("说明文档不存在")
        elif choice in ['0', 'q']:
            print("\n👋 再见！")
            break
        else:
            print("\n❌ 无效的选择，请重新输入")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  已退出")
        sys.exit(0)
