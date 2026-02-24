import re
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

class Arithmetic24Env:
    def __init__(self):
        self.transformations = standard_transformations + (implicit_multiplication_application,)
        
    def get_prompt(self, nums_str):
        return f"""你是一个解决24点游戏的顶级数学专家。
请使用给定的四个数字，通过加(+)、减(-)、乘(*)、除(/)和括号()，计算出24。每个数字必须且只能使用一次。

【严格执行规则】
1. 思考过程：必须在 <think></think> 标签内简短地写出寻找24的逻辑推导。提示：除法可能会产生分数作为中间结果（例如 8/3），这是允许且常见的解题技巧。
2. 最终答案：在 </think> 之后，仅输出一行纯数学表达式。
3. 字符白名单：最终的表达式中【只能】包含数字、加减乘除符号和括号，严禁出现等号（=）、汉字、英文字母或任何解释性标点。

正确格式示例：
<think>
目标是凑出24。给定的数字是 3, 6, 8, 2。
我可以尝试先用 3 * 6 得到 18。
剩下的数字是 8 和 2。如果用 18 + 8 = 26。
最后 26 - 2 正好等于 24。
四个数字都用了一次。最终表达式：3 * 6 + 8 - 2。
</think>
3 * 6 + 8 - 2

输入：{nums_str}
输出：
<think>
"""
        
    def _parse_output(self, text):
        has_think = "</think>" in text
        think_count = text.count("</think>")
        
        if has_think:
            # 【核心修复】：切分后取索引 [1]，即第一个 </think> 紧跟的内容
            # 坚决不能用 [-1]，防止模型胡言乱语生成多个标签
            parts = text.split("</think>")
            after_think_text = parts[1].strip()
            
            # 过滤掉空行，只取第一行有效内容作为表达式
            lines = [line.strip() for line in after_think_text.split('\n') if line.strip()]
            pred_expr = lines[0] if lines else ""
        else:
            # 没输出 </think>，取最后一行兜底
            lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
            pred_expr = lines[-1] if lines else ""
        
        # 【新增】返回</think>的出现次数，用于检测多标签问题
        return has_think, pred_expr, think_count

    def _verify_expression(self, expr_str, target_nums):
        if not expr_str:
            return False, "Empty expression"
        
        expr_str = expr_str.split('=')[0].strip()
        
        # 1. 严格数字匹配：检查数字的“多重集”(Multiset)
        digits = re.findall(r'\d+', expr_str)
        try:
            used_nums = sorted([int(d) for d in digits])
            target_nums_int = sorted([int(n) for n in target_nums])
        except ValueError:
            return False, "Parse error"

        if used_nums != target_nums_int:
            return False, "Used wrong numbers"
        
        # ==========================================
        # 【新增的终极安全防线】：防止模型乱造符号卡死评测器
        # 1. 字符白名单：只允许数字、加减乘除、括号、空格、小数点
        if not re.fullmatch(r'[\d\+\-\*\/\(\)\s\.]+', expr_str):
            return False, "Invalid characters"
        
        # 2. 禁用指数运算：防止模型生成类似 8**9999 导致 SymPy 内存溢出
        if "**" in expr_str:
            return False, "Exponentiation not allowed"
        # ==========================================

        # 2. 严谨的数学求值
        try:
            # 允许 SymPy 处理 / 为有理数除法
            parsed = parse_expr(expr_str, transformations=self.transformations, evaluate=True)
            
            # 检查是否包含变量（防止模型输出类似 8x/3 + 24 - 8x）
            if hasattr(parsed, 'free_symbols') and parsed.free_symbols:
                return False, "Contains variables"

            # 【性能优化】放弃极度缓慢的 sympy.simplify，直接使用 float 计算
            # RL 训练中环境反馈必须极速。由于前面的字符白名单已经排除了非常规操作，转 float 是安全的，速度提升百倍
            if abs(float(parsed) - 24.0) < 1e-5:
                return True, "Correct"
            
            return False, "Wrong value"
        except Exception:
            return False, "Math error"

    def compute_reward(self, input_nums_str, output_text):
        target_nums = [n.strip() for n in input_nums_str.split(',')]
        has_think, pred_expr, think_count = self._parse_output(output_text)
        
        reward = 0.0
        is_correct = False
    
        # 阶段1：严重违规直接大扣并早退
        if think_count > 1:
            reward -= 1.0
            return reward, False
        
        if re.search(r'[\u4e00-\u9fa5a-zA-Z]', pred_expr):  # 废话/字母
            reward -= 0.8
            return reward, False
    
        # 阶段2：基础格式奖励（小幅正向）
        if has_think:
            reward += 0.2          # 从 0.1 提到 0.2，鼓励用 think
    
        # 响应长度惩罚（软化，防止过度短输出）
        text_len = len(output_text.strip())
        if text_len > 500:
            reward -= 0.4
        elif text_len < 50:
            reward -= 0.2          # 太短也轻罚，防止 collapse 到空
    
        # 阶段3：表达式初步检查
        if not pred_expr:
            reward -= 0.4
            return reward, False
    
        equals_count = pred_expr.count('=')
        if equals_count > 0:
            reward -= 0.4          # 从 -0.6 降到 -0.4，允许轻微多步残留
    
        expr_len = len(pred_expr)
        if expr_len > 120:
            reward -= 0.3
        elif expr_len < 10:
            reward -= 0.3
    
        # 阶段4：数学验证 + 奖励放大
        is_correct, reason = self._verify_expression(pred_expr, target_nums)
        if is_correct:
            reward += 4.0                  # 核心正向信号
            
            # 弱化形状奖励：只在正确且有一定复杂性时加一点点
            operators = sum(1 for c in pred_expr if c in '+-*/')
            if operators >= 3 or '(' in pred_expr:
                reward += 0.3              # 幅度控制在 7.5% 左右，不容易被 hack
        else:
            # 负反馈保持小而均匀
            if reason in ["Math error", "Parse error"]:
                reward -= 0.2
            elif reason in ["Used wrong numbers", "Invalid characters"]:
                reward -= 0.3
            elif reason == "Exponentiation not allowed":
                reward -= 0.4
            else:
                reward -= 0.1
            
        # 防止极端负值
        reward = max(reward, -1.5)
    
        return reward, is_correct

if __name__ == "__main__":
    env = Arithmetic24Env()
    prompt = env.get_prompt("3, 3, 8, 8")
    print("Prompt Preview:\n", prompt)
    print("\n" + "="*80)
    print("测试改进的奖励函数")
    print("="*80)
    
    # 测试1：正确答案
    sample_out = "</think>\n8 / (3 - 8/3)"
    reward, is_correct = env.compute_reward("3, 3, 8, 8", sample_out)
    print(f"\n✅ 正确答案: '8 / (3 - 8/3)'")
    print(f"   奖励: {reward:.2f}, 正确: {is_correct}")

    # 测试2：错误的数字
    sample_out_bad = "</think>\n8 * 3"
    reward, is_correct = env.compute_reward("3, 3, 8, 8", sample_out_bad)
    print(f"\n❌ 错误数字: '8 * 3' (应该用3, 3, 8, 8)")
    print(f"   奖励: {reward:.2f}, 正确: {is_correct}")
    
    # 测试3：包含等号（不规范格式）
    sample_out_with_equals = "</think>\n8 / (3 - 8/3) = 24"
    reward, is_correct = env.compute_reward("3, 3, 8, 8", sample_out_with_equals)
    print(f"\n⚠️  包含等号: '8 / (3 - 8/3) = 24'")
    print(f"   奖励: {reward:.2f}, 正确: {is_correct}")
    
    # 测试4：多个</think>标签
    sample_out_multiple_tags = "</think>\n计算过程</think>\n8 / (3 - 8/3)"
    reward, is_correct = env.compute_reward("3, 3, 8, 8", sample_out_multiple_tags)
    print(f"\n❌ 多个标签: '......</think>......</think>......'")
    print(f"   奖励: {reward:.2f}, 正确: {is_correct}")