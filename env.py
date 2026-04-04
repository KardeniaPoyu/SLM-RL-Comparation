"""
env.py — 24点游戏环境 & 复合奖励函数 (RLVR)
极速版：用 Python eval() 替代 SymPy，预编译所有正则，奖励计算速度提升 ~50x
"""

import re

# ── 预编译正则（避免每次调用重新编译）──
_RE_DIGITS = re.compile(r'\d+')
_RE_WHITELIST = re.compile(r'[\d\+\-\*\/\(\)\s\.]+')
_RE_GARBAGE = re.compile(r'[\u4e00-\u9fa5a-zA-Z]')
_RE_EQUATION = re.compile(r'([\d\.\s\+\-\*\/\(\)]+)=([\s\-\d\.]+)')


class Arithmetic24Env:
    def __init__(self, simple_mode=True):
        """
        simple_mode=True:    分层连续递进奖励 (含距离衰减)，RL 前期推荐
        simple_mode=False:   复合多阶段奖励 (保留原有逻辑)，RL 后期可切换
        simple_mode='binary': 严格二元奖励 (correct=1, wrong=0, garbage=-0.5)
                              用于 Dual-Phase Reward Schedule 的 Phase 2
        """
        self.simple_mode = simple_mode

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
        返回: (has_think_open, has_think_close, pred_expr, think_close_count, think_content)
        """
        has_think_close = "</think>" in text
        
        # 因为 return_prompt=False，模型输出一般不会再包含原始 prompt 的 <think>
        # 所以我们只需要验证模型是否老老实实输出了 </think> 即可，或者人为给它加上
        has_think_open = True
        
        think_close_count = text.count("</think>")

        if has_think_close:
            parts = text.split("</think>")
            think_content = parts[0].strip()
            after_think_text = parts[1].strip()
            lines = [line.strip() for line in after_think_text.split('\n') if line.strip()]
            pred_expr = lines[0] if lines else ""
        else:
            think_content = text.strip()
            lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
            pred_expr = lines[-1] if lines else ""

        return has_think_open, has_think_close, pred_expr, think_close_count, think_content

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
            # 安全增强：禁止用小数或者负数来拼凑原始数字
            # 原始题目一定是整数形式的，所以我们在处理 target_nums 时，检查有没有数字粘连（比如把 3 和 8 拼成了 38）
            # 虽然多重集检查了一次，但有可能 1和0拼成了 10，而题目里其实是单独的1和0，多重集不会查出来数字被重排（例如[1,0] 和 [10] 被正则当做不同的字符串，但这在正则步骤已经拦截了）。
            
            # 为了防止 1/0 计算得到无穷大等导致奇怪的 float 对比，严格检查异常
            result = eval(expr_str, {"__builtins__": {}}, {})
            
            if result is None:
                return False, "Math error", None
                
            if abs(float(result) - 24.0) < 1e-5:
                # 增加了返回 float 结果的能力
                return True, "Correct", float(result)
            return False, "Wrong value", float(result)
        except ZeroDivisionError:
            return False, "Math error (Division by zero)", None
        except Exception:
            return False, "Math error", None

    def _evaluate_intermediate_steps(self, think_content):
        """
        [PRM] 验证状态追踪格式中的中间公式是否合法。
        支持的新格式： 8 + 10 = 18。剩余: [13, 3, 18]
        返回: (正确等式数量, 是否存在瞎编等式)
        """
        # 注意：这里的正则 _RE_EQUATION 在上方已被修改
        equations = _RE_EQUATION.findall(think_content)
        correct_count = 0
        has_hallucination = False
        
        for left_expr, right_val in equations:
            left_expr = left_expr.strip()
            # 过滤掉句号和多余空白
            right_val = right_val.replace('。', '').strip()
            if not left_expr or not right_val:
                continue
                
            # 清理无用字符
            if not _RE_WHITELIST.fullmatch(left_expr):
                continue
                
            try:
                # 评估左边
                left_result = eval(left_expr, {"__builtins__": {}}, {})
                # 评估右边
                right_result = float(right_val)
                
                # 避免一些极小浮点数差异
                if abs(float(left_result) - right_result) < 1e-4:
                    correct_count += 1
                else:
                    # 左边算出来不等于它声明的右边，这是严重的幻觉作弊
                    has_hallucination = True
                    break
            except Exception:
                has_hallucination = True
                break
                
        return correct_count, has_hallucination

    def compute_reward(self, input_nums_str, output_text):
        """
        奖励函数。
        simple_mode=True: 分层连续递进奖励，包含距离衰减 (0.3 -> ~0.8) 与防作弊惩罚 (-0.5)
        simple_mode=False: 复合多阶段奖励 (保留原有逻辑，含负奖励)
        """
        import math
        
        target_nums = [n.strip() for n in input_nums_str.split(',')]
        N_expected = len(target_nums)
        has_think_open, has_think_close, pred_expr, think_close_count, think_content = self._parse_output(output_text)

        # ── Dual-Phase Binary 模式 (Phase 2: Anti-Reward-Hacking) ──
        if self.simple_mode == 'binary':
            # 最简洁的信号: 正确=1, 格式错误=-0.5, 其他错误=0
            if think_close_count > 1 or not pred_expr:
                return -0.5, False
            if _RE_GARBAGE.search(pred_expr):
                return -0.5, False

            is_correct, reason, eval_result = self._verify_expression(pred_expr, target_nums)
            if is_correct:
                return 1.0, True
            else:
                return 0.0, False

        # ── 连续化模式 (Distance Reward & Anti-Hallucination) ──
        if self.simple_mode:
            # Tier 0: 垃圾输出惩罚
            if think_close_count > 1 or not pred_expr:
                return -0.5, False  # 修改为严厉惩罚，避免乱用标签
            if _RE_GARBAGE.search(pred_expr):
                return -0.5, False

            # Anti-Hallucination: 思维链捏造检查。像 "剩余: [6, 12, 6]" 这样没越算越少的
            remain_matches = re.finditer(r'(?:剩余|还剩下|剩?下?的?数?是?:?)\s*[:：]?\s*\[(.*?)\]', think_content)
            for match in remain_matches:
                arr_str = match.group(1)
                # 使用已有逻辑提取
                digits = _RE_DIGITS.findall(arr_str)
                if len(digits) >= N_expected:
                    # 抓到造假
                    return -0.5, False

            # 尝试验证表达式
            is_correct, reason, eval_result = self._verify_expression(pred_expr, target_nums)

            # Number Count Mismatch - 极其核心的防 N=4 过拟合查杀点
            # 如果没凑对但使用了错误的数字量，不管是不是报错，统统罚款
            try:
                digits = _RE_DIGITS.findall(pred_expr.split('=')[0])
                if len(digits) > 0 and len(digits) != N_expected:
                    return -0.5, False
            except Exception:
                pass


            if is_correct:
                # Tier 3/4: 正确
                reward = 1.0
                if has_think_close and len(think_content) > 10:
                    reward = 1.5  # Tier 4: 优秀 — 有推理过程 (提高上线增加探索吸引力)
                return reward, True
            else:
                # "距离连续判定" - 使用了正确的数字并计算出结果，但不是24
                if reason == "Wrong value" and eval_result is not None:
                    # 例如，算出22，距离是 2。exp(-0.2) ≈ 0.81 * 0.8 ≈ 0.65
                    # 例如，算出100，距离是 76。exp(-7.6) 趋近于 0
                    dist = abs(eval_result - 24.0)
                    continuous_r = 0.8 * math.exp(-dist / 10.0) 
                    
                    # 避免给了合法错误组合和直接写0.1差不多，稍微给点甜头
                    reward = 0.1 + max(0.0, continuous_r)
                    
                    if has_think_close:
                        reward += 0.05
                    return min(0.95, reward), False  # 封顶不到 1.0 (防止抢走真正24的风头)
                    
                elif reason in ("Used wrong numbers", "Invalid characters", "Exponentiation not allowed"):
                    # Tier 1: 至少有个表达式，但数字用错了。由于前面有 Number Count 检测，这里拦截的是篡改数值的幻觉
                    reward = -0.2  # 不是0.1，用错了数字其实就是幻觉
                    return reward, False
                else:
                    # Tier 1: 数学错误 / 解析错误 (如 1/0)
                    reward = 0.1 if has_think_close else 0.0
                    return reward, False

        # ── 复合多阶段模式 (保留原有逻辑) ──
        reward = 0.0
        is_correct = False

        # ── 阶段1：严重违规 → 扣分并早退 ──
        if think_close_count > 1:
            return -1.0, False

        if _RE_GARBAGE.search(pred_expr):
            return -0.8, False

        # ── 阶段2：格式与思维链 (CoT) 奖励 ──
        if has_think_close:
            reward += 0.2
            think_len = len(think_content)
            if think_len < 3:
                reward -= 0.5
            elif think_len > 10:
                reward += 0.2
            correct_steps, has_hallucination = self._evaluate_intermediate_steps(think_content)
            if has_hallucination:
                reward -= 1.0
            else:
                reward += min(correct_steps * 0.1, 0.5)
        else:
            reward -= 0.5

        text_len = len(output_text.strip())
        if text_len > 600:
            reward -= 0.4

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
        is_correct, reason, eval_result = self._verify_expression(pred_expr, target_nums)
        if is_correct:
            reward += 4.0
            operators = sum(1 for c in pred_expr if c in '+-*/')
            if operators >= 3 or '(' in pred_expr:
                reward += 0.3
        else:
            if reason in ("Math error", "Parse error", "Math error (Division by zero)", "Empty expression"):
                reward -= 0.6
            elif reason in ("Used wrong numbers", "Invalid characters"):
                reward -= 0.6
            elif reason == "Exponentiation not allowed":
                reward -= 0.4
            else:
                reward -= 0.2

        return max(reward, -1.5), is_correct

# ── 模块级辅助函数（ProcessPoolExecutor 需要顶层可 pickle 的函数）──
_global_env = None
_global_simple_mode = True

def _compute_single_reward(args):
    """Worker function for parallel reward computation."""
    global _global_env, _global_simple_mode
    if _global_env is None or _global_env.simple_mode != _global_simple_mode:
        _global_env = Arithmetic24Env(simple_mode=_global_simple_mode)
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


def compute_rewards_parallel(nums_list, responses, simple_mode=True):
    """
    并行计算奖励（利用多核 CPU 加速字符串解析）。
    simple_mode=True:     分层连续递进奖励，RL 前期及 fixed schedule 推荐
    simple_mode='binary':  严格二元奖励，用于 Dual-Phase 的 Phase 2
    simple_mode=False:    复合多阶段奖励
    返回: (rewards_list, corrects_count)
    """
    global _global_simple_mode
    _global_simple_mode = simple_mode
    try:
        pool = _get_reward_pool()
        results = list(pool.map(_compute_single_reward,
                                zip(nums_list, responses),
                                chunksize=max(1, len(nums_list) // 8)))
    except Exception:
        # fallback 串行
        env = Arithmetic24Env(simple_mode=simple_mode)
        results = [env.compute_reward(n, r) for n, r in zip(nums_list, responses)]

    rewards = [r for r, _ in results]
    corrects = sum(1 for _, c in results if c)
    return rewards, corrects


if __name__ == "__main__":
    for mode_name, simple in [("分层递进模式 (simple)", True), ("复合模式 (complex)", False)]:
        env = Arithmetic24Env(simple_mode=simple)
        print(f"\n{'='*60}")
        print(f"  {mode_name}")
        print(f"{'='*60}")

        tests = [
            # Tier 4: 正确 + 推理过程
            ("3, 3, 8, 8", "<think>8/(3-8/3) 先算 8/3≈2.67，3-2.67=0.33，8/0.33=24</think>\n8 / (3 - 8/3)", "Tier4: 正确+推理"),
            # Tier 3: 正确但没推理
            ("3, 3, 8, 8", "</think>\n8 / (3 - 8/3)", "Tier3: 正确无推理"),
            # Tier 2: 正确数字但值≠24
            ("3, 3, 8, 8", "</think>\n3 + 3 + 8 + 8", "Tier2: 对数字错值"),
            # Tier 1: 用错数字
            ("3, 3, 8, 8", "</think>\n8 * 3", "Tier1: 错数字"),
            # Tier 0: 垃圾
            ("3, 3, 8, 8", "</think>\n计算</think>\n8/(3-8/3)", "Tier0: 多标签"),
            ("3, 3, 8, 8", "没有任何格式", "Tier0: 无表达式"),
        ]

        for nums, output, desc in tests:
            r, c = env.compute_reward(nums, output)
            status = "✅" if c else "❌"
            print(f"  {status} {desc:20s} → reward={r:+.2f}, correct={c}")

    # 速度基准测试
    import time
    test_expr = "</think>\n8 / (3 - 8/3)"
    start = time.perf_counter()
    for _ in range(1000):
        env.compute_reward("3, 3, 8, 8", test_expr)
    elapsed = time.perf_counter() - start
    print(f"\n⚡ 1000 次奖励计算耗时: {elapsed:.3f}s ({elapsed*1000:.1f}ms/call → {1000/elapsed:.0f} calls/s)")