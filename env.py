import re
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

class Arithmetic24Env:
    def __init__(self):
        self.transformations = standard_transformations + (implicit_multiplication_application,)
        
    def get_prompt(self, nums_str):
        # 【修复1】使用极简版示范，逼迫模型用最少字数完成思考，防止 Token 被截断
        return f"""You are a math expert solving the 24-point game.
Use exactly these four numbers to make 24 using +, -, *, /, and parentheses.
Keep your thoughts extremely short inside <think></think>, then output the math expression.

Example:
Input: 2, 3, 6, 8
Output:
<think>
3 * 6 = 18
18 + 8 = 26
26 - 2 = 24
</think>
3 * 6 + 8 - 2

Input: {nums_str}
Output:
<think>
"""
        
    def _parse_output(self, text):
        has_think = "</think>" in text
        
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
            
        return has_think, pred_expr

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

            # 使用 SymPy 的简化功能检查是否等于 24
            if sympy.simplify(parsed - 24) == 0:
                return True, "Correct"
            
            # 备选：处理浮点数情况
            if abs(float(parsed) - 24.0) < 1e-6:
                return True, "Correct"
            
            return False, "Wrong value"
        except Exception:
            return False, "Math error"

    def compute_reward(self, input_nums_str, output_text):
        target_nums = [n.strip() for n in input_nums_str.split(',')]
        has_think, pred_expr = self._parse_output(output_text)
        
        reward = 0.0
        
        # 格式分
        if has_think:
            reward += 0.1
            
        # 【新增】：严厉惩罚“话痨”行为，逼迫模型学会输出 EOS
        if output_text.count("</think>") > 1 or "Explanation:" in output_text:
            reward -= 0.5 
            
        if not pred_expr:
            reward -= 0.5
            return reward, False
            
        # 逻辑分验证
        is_correct, reason = self._verify_expression(pred_expr, target_nums)
        if is_correct:
            reward += 1.0
        else:
            if reason in ["Math error", "Parse error", "Used wrong numbers", "Empty expression"]:
                reward -= 0.5
                
        return reward, is_correct

if __name__ == "__main__":
    env = Arithmetic24Env()
    prompt = env.get_prompt("3, 3, 8, 8")
    print("Prompt Preview:\n", prompt)
    
    # 模拟大模型实际的输出行为：开头不带 <think>，因为 Prompt 已经替它写了
    sample_out = "\n8 / (3 - 8/3) = 8 / (1/3) = 24\n</think>\n8 / (3 - 8/3)"
    reward, is_correct = env.compute_reward("3, 3, 8, 8", sample_out)
    print(f"\nSample OK Output -> Reward: {reward}, Correct: {is_correct}")

    sample_out_bad = "\n8 * 3 = 24\n</think>\n8 * 3"
    reward, is_correct = env.compute_reward("3, 3, 8, 8", sample_out_bad)
    print(f"Sample Bad Numbers Output -> Reward: {reward}, Correct: {is_correct}")