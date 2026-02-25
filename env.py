"""
env.py — 24点游戏环境 & 复合奖励函数 (RLVR)
支持 N=3~6 任意数量数字的验证
"""

import re
import sympy
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application
)


class Arithmetic24Env:
    def __init__(self):
        self.transformations = standard_transformations + (implicit_multiplication_application,)

    def get_prompt(self, nums_str):
        return f"""计算24点。
使用数字 {nums_str}，每个只能用一次。
先在 <think></think> 内写简短步骤。
</think> 后只输出最终公式。
<think>
"""

    def _parse_output(self, text):
        """
        解析模型输出，提取表达式和格式信息。
        返回: (has_think_open, has_think_close, pred_expr, think_close_count)
        """
        has_think_close = "</think>" in text
        has_think_open = "<think>" in text
        think_close_count = text.count("</think>")

        if has_think_close:
            parts = text.split("</think>")
            after_think_text = parts[1].strip()
            lines = [line.strip() for line in after_think_text.split('\n') if line.strip()]
            pred_expr = lines[0] if lines else ""
        else:
            lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
            pred_expr = lines[-1] if lines else ""

        return has_think_open, has_think_close, pred_expr, think_close_count

    def _verify_expression(self, expr_str, target_nums):
        """验证表达式是否合法且等于24。统一返回 (bool, str)。"""
        if not expr_str:
            return False, "Empty expression"

        expr_str = expr_str.split('=')[0].strip()

        # 1. 严格数字匹配：多重集 Multiset
        digits = re.findall(r'\d+', expr_str)
        try:
            used_nums = sorted([int(d) for d in digits])
            target_nums_int = sorted([int(n) for n in target_nums])
        except ValueError:
            return False, "Parse error"

        if used_nums != target_nums_int:
            return False, "Used wrong numbers"

        # 2. 字符白名单
        if not re.fullmatch(r'[\d\+\-\*\/\(\)\s\.]+', expr_str):
            return False, "Invalid characters"

        # 3. 禁用指数运算
        if "**" in expr_str:
            return False, "Exponentiation not allowed"

        # 4. 数学求值
        try:
            parsed = parse_expr(expr_str, transformations=self.transformations, evaluate=True)

            if hasattr(parsed, 'free_symbols') and parsed.free_symbols:
                return False, "Contains variables"

            if abs(float(parsed) - 24.0) < 1e-5:
                return True, "Correct"

            return False, "Wrong value"
        except Exception:
            return False, "Math error"

    def compute_reward(self, input_nums_str, output_text):
        """
        复合奖励函数 (RLVR)。
        分4个阶段：严重违规 -> 格式奖励 -> 表达式检查 -> 数学验证。
        """
        target_nums = [n.strip() for n in input_nums_str.split(',')]
        has_think_open, has_think_close, pred_expr, think_close_count = self._parse_output(output_text)

        reward = 0.0
        is_correct = False

        # ── 阶段1：严重违规 → 扣分并早退 ──
        if think_close_count > 1:
            reward -= 1.0
            return reward, False

        if re.search(r'[\u4e00-\u9fa5a-zA-Z]', pred_expr):
            reward -= 0.8
            return reward, False

        # ── 阶段2：格式奖励 ──
        # 完整 <think>...</think> 结构给满分，仅有闭合标签给一半
        if has_think_open and has_think_close:
            reward += 0.2
        elif has_think_close:
            reward += 0.1

        # 长度惩罚
        text_len = len(output_text.strip())
        if text_len > 500:
            reward -= 0.4
        elif text_len < 50:
            reward -= 0.2

        # ── 阶段3：表达式初步检查 ──
        if not pred_expr:
            reward -= 0.4
            return reward, False

        equals_count = pred_expr.count('=')
        if equals_count > 0:
            reward -= 0.4

        expr_len = len(pred_expr)
        if expr_len > 120:
            reward -= 0.3
        elif expr_len < 10:
            reward -= 0.3

        # ── 阶段4：数学验证 ──
        is_correct, reason = self._verify_expression(pred_expr, target_nums)
        if is_correct:
            reward += 4.0

            operators = sum(1 for c in pred_expr if c in '+-*/')
            if operators >= 3 or '(' in pred_expr:
                reward += 0.3
        else:
            if reason in ["Math error", "Parse error"]:
                reward -= 0.2
            elif reason in ["Used wrong numbers", "Invalid characters"]:
                reward -= 0.3
            elif reason == "Exponentiation not allowed":
                reward -= 0.4
            else:
                reward -= 0.1

        reward = max(reward, -1.5)
        return reward, is_correct


if __name__ == "__main__":
    env = Arithmetic24Env()
    prompt = env.get_prompt("3, 3, 8, 8")
    print("Prompt Preview:\n", prompt)

    tests = [
        ("3, 3, 8, 8", "<think>步骤</think>\n8 / (3 - 8/3)", "完整think结构+正确"),
        ("3, 3, 8, 8", "</think>\n8 / (3 - 8/3)", "仅闭合标签+正确"),
        ("3, 3, 8, 8", "</think>\n8 * 3", "错误数字"),
        ("3, 3, 8, 8", "</think>\n计算过程</think>\n8 / (3 - 8/3)", "多标签"),
        ("3, 5, 8", "</think>\n(5 + 3) * 8", "N=3 错误值(=64)"),
    ]

    print("\n" + "=" * 60)
    for nums, output, desc in tests:
        r, c = env.compute_reward(nums, output)
        status = "✅" if c else "❌"
        print(f"  {status} {desc:30s} → reward={r:+.2f}, correct={c}")