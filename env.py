"""
env.py — 24点游戏环境 & 复合奖励函数 (RLVR)
极速版：用 Python eval() 替代 SymPy，预编译所有正则，奖励计算速度提升 ~50x
"""

import re

# ── 预编译正则（避免每次调用重新编译）──
_RE_DIGITS = re.compile(r'\d+')
_RE_WHITELIST = re.compile(r'[\d\+\-\*\/\(\)\s\.]+')
_RE_GARBAGE = re.compile(r'[\u4e00-\u9fa5a-zA-Z]')


class Arithmetic24Env:
    def __init__(self):
        pass

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
        """
        验证表达式是否合法且等于24。
        用 Python eval() 替代 SymPy parse_expr，速度提升 ~50x。
        安全性由前置的字符白名单保证。
        """
        if not expr_str:
            return False, "Empty expression"

        expr_str = expr_str.split('=')[0].strip()

        # 1. 严格数字匹配：多重集 Multiset
        digits = _RE_DIGITS.findall(expr_str)
        try:
            used_nums = sorted([int(d) for d in digits])
            target_nums_int = sorted([int(n) for n in target_nums])
        except ValueError:
            return False, "Parse error"

        if used_nums != target_nums_int:
            return False, "Used wrong numbers"

        # 2. 字符白名单（这一步保证了 eval() 的安全性）
        if not _RE_WHITELIST.fullmatch(expr_str):
            return False, "Invalid characters"

        # 3. 禁用指数运算
        if "**" in expr_str:
            return False, "Exponentiation not allowed"

        # 4. 极速数学求值：用 Python eval() 替代 SymPy
        try:
            result = eval(expr_str, {"__builtins__": {}}, {})
            if abs(float(result) - 24.0) < 1e-5:
                return True, "Correct"
            return False, "Wrong value"
        except ZeroDivisionError:
            return False, "Math error"
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
            return -1.0, False

        if _RE_GARBAGE.search(pred_expr):
            return -0.8, False

        # ── 阶段2：格式奖励 ──
        if has_think_open and has_think_close:
            reward += 0.2
        elif has_think_close:
            reward += 0.1

        text_len = len(output_text.strip())
        if text_len > 500:
            reward -= 0.4
        elif text_len < 50:
            reward -= 0.2

        # ── 阶段3：表达式初步检查 ──
        if not pred_expr:
            return max(reward - 0.4, -1.5), False

        if '=' in pred_expr:
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
            if reason in ("Math error", "Parse error"):
                reward -= 0.2
            elif reason in ("Used wrong numbers", "Invalid characters"):
                reward -= 0.3
            elif reason == "Exponentiation not allowed":
                reward -= 0.4
            else:
                reward -= 0.1

        return max(reward, -1.5), is_correct

# ── 模块级辅助函数（ProcessPoolExecutor 需要顶层可 pickle 的函数）──
_global_env = None

def _compute_single_reward(args):
    """Worker function for parallel reward computation."""
    global _global_env
    if _global_env is None:
        _global_env = Arithmetic24Env()
    nums, resp = args
    return _global_env.compute_reward(nums, resp)


# ── 进程池（懒初始化，复用）──
_reward_pool = None

def _get_reward_pool():
    global _reward_pool
    if _reward_pool is None:
        import os
        from concurrent.futures import ProcessPoolExecutor
        n_workers = min(os.cpu_count() or 4, 8)
        _reward_pool = ProcessPoolExecutor(max_workers=n_workers)
    return _reward_pool


def compute_rewards_parallel(nums_list, responses):
    """
    并行计算奖励（利用多核 CPU 加速字符串解析）。
    返回: (rewards_list, corrects_count)
    """
    try:
        pool = _get_reward_pool()
        results = list(pool.map(_compute_single_reward,
                                zip(nums_list, responses),
                                chunksize=max(1, len(nums_list) // 8)))
    except Exception:
        # fallback 串行
        env = Arithmetic24Env()
        results = [env.compute_reward(n, r) for n, r in zip(nums_list, responses)]

    rewards = [r for r, _ in results]
    corrects = sum(1 for _, c in results if c)
    return rewards, corrects


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

    # 速度基准测试
    import time
    test_expr = "</think>\n8 / (3 - 8/3)"
    start = time.perf_counter()
    for _ in range(1000):
        env.compute_reward("3, 3, 8, 8", test_expr)
    elapsed = time.perf_counter() - start
    print(f"\n⚡ 1000 次奖励计算耗时: {elapsed:.3f}s ({elapsed*1000:.1f}ms/call → {1000/elapsed:.0f} calls/s)")