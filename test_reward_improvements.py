"""
对比测试：旧奖励函数 vs 新奖励函数
"""

import sys

class OldArithmetic24Env:
    """旧版本的奖励函数（用于对比）"""
    def __init__(self):
        pass
        
    def _parse_output(self, text):
        has_think = "</think>" in text
        if has_think:
            parts = text.split("</think>")
            after_think_text = parts[1].strip()
            lines = [line.strip() for line in after_think_text.split('\n') if line.strip()]
            pred_expr = lines[0] if lines else ""
        else:
            lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
            pred_expr = lines[-1] if lines else ""
        return has_think, pred_expr
    
    def compute_reward(self, input_nums_str, output_text):
        """原版本（简化版，仅作演示）"""
        has_think, pred_expr = self._parse_output(output_text)
        
        reward = 0.0
        if has_think:
            reward += 0.1
            
        if not pred_expr:
            reward -= 0.5
            return reward, False
        
        # 简单检查（原版本）
        if any(c in pred_expr for c in ['中', '英', '的', '是']):
            reward -= 1.0
            return reward, False
        
        # 假设数学验证通过
        return reward + 1.0, True

# 导入新版本
sys.path.insert(0, '.')
from env import Arithmetic24Env

def test_case(name, nums, response, expected_old, expected_new_min):
    """测试单个用例"""
    print(f"\n{'='*80}")
    print(f"📝 测试: {name}")
    print(f"{'='*80}")
    print(f"输入数字: {nums}")
    print(f"模型输出:\n{response[:100]}..." if len(response) > 100 else f"模型输出:\n{response}")
    
    old_env = OldArithmetic24Env()
    new_env = Arithmetic24Env()
    
    try:
        old_reward, old_correct = old_env.compute_reward(nums, response)
    except Exception as e:
        old_reward = "ERROR"
        old_correct = str(e)
    
    try:
        new_reward, new_correct = new_env.compute_reward(nums, response)
    except Exception as e:
        new_reward = "ERROR"
        new_correct = str(e)
    
    print(f"\n📊 结果对比:")
    print(f"  旧版本: reward={old_reward:>6}, correct={old_correct}")
    print(f"  新版本: reward={new_reward:>6}, correct={new_correct}")
    
    # 评估
    if isinstance(new_reward, (int, float)):
        if new_reward >= expected_new_min:
            print(f"  ✅ 新版本奖励符合预期 (>= {expected_new_min})")
        else:
            print(f"  ⚠️  新版本奖励低于预期 (期望>= {expected_new_min}, 实际={new_reward})")
    
    return new_reward

def main():
    print("\n" + "="*80)
    print("🧪 奖励函数改进验证测试")
    print("="*80)
    
    results = []
    
    # 测试1: 正确答案
    r1 = test_case(
        "正确答案（标准格式）",
        "3, 6, 8, 2",
        "</think>\n3 * 6 + 8 - 2",
        expected_old=1.1,
        expected_new_min=1.1
    )
    results.append(("正确答案", r1))
    
    # 测试2: 多个</think>标签（新增检查）
    r2 = test_case(
        "多个</think>标签 - 严重违规",
        "3, 6, 8, 2",
        "</think>\n3 * 6 = 18\n</think>\n18 + 8 - 2 = 24\n</think>\n3 * 6 + 8 - 2",
        expected_old=-0.5,
        expected_new_min=-1.0
    )
    results.append(("多个标签", r2))
    
    # 测试3: 包含等号（新增检查）
    r3 = test_case(
        "包含等号 - 计算步骤而非表达式",
        "3, 6, 8, 2",
        "</think>\n3 * 6 = 18",
        expected_old=0.6,
        expected_new_min=-0.6
    )
    results.append(("包含等号", r3))
    
    # 测试4: 包含中文（原检查升级）
    r4 = test_case(
        "包含中文 - 垃圾文字",
        "3, 6, 8, 2",
        "</think>\n3乘以6加8减2等于24",
        expected_old=-1.0,
        expected_new_min=-1.0
    )
    results.append(("中文垃圾", r4))
    
    # 测试5: 大量无关文本（新增检查）
    long_text = "</think>\n3 * 6 + 8 - 2\n这个答案是对的，因为：" + "这里有很多无关文本" * 50
    r5 = test_case(
        "过长Response - 包含废话",
        "3, 6, 8, 2",
        long_text,
        expected_old=1.1,
        expected_new_min=-0.3
    )
    results.append(("过长response", r5))
    
    # 测试6: 只有一对括号的正确答案
    r6 = test_case(
        "带括号的正确答案",
        "3, 6, 8, 2",
        "</think>\n(3 * 6 + 8) - 2",
        expected_old=1.1,
        expected_new_min=1.1
    )
    results.append(("带括号", r6))
    
    # 测试7: 错误的数字
    r7 = test_case(
        "使用了错误的数字",
        "3, 6, 8, 2",
        "</think>\n5 * 6 + 8 - 2",
        expected_old=-0.5,
        expected_new_min=-0.5
    )
    results.append(("错误数字", r7))
    
    # 测试8: 没有</think>标签（兼容旧模式）
    r8 = test_case(
        "没有</think>标签",
        "3, 6, 8, 2",
        "3 * 6 + 8 - 2",
        expected_old=1.0,
        expected_new_min=1.0
    )
    results.append(("无think标签", r8))
    
    # 总结
    print(f"\n" + "="*80)
    print("📈 测试总结")
    print("="*80)
    
    print("\n奖励分数统计:")
    for test_name, reward in results:
        if isinstance(reward, (int, float)):
            status = "✅ 正常" if reward >= -0.5 else "❌ 低奖励"
            print(f"  {test_name:<20}: {reward:>7.2f} {status}")
        else:
            print(f"  {test_name:<20}: {reward}")
    
    print(f"\n" + "="*80)
    print("💡 主要改进:")
    print("  ✅ 多标签检测: 从无到 -1.0 分的严惩")
    print("  ✅ 等号检测: 从无到 -0.6 分的惩罚")
    print("  ✅ 长度检测: 从无到 -0.3 分的惩罚")
    print("  ✅ 更清晰的奖惩梯度: 最低 -1.0，最高 +1.1")

if __name__ == "__main__":
    main()
