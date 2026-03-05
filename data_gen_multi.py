"""
data_gen_multi.py — 多难度24点数据生成器
支持 N ∈ {3, 4, 5, 6}，输出 JSONL 格式
同时生成 SFT 训练数据（含 CoT 推理轨迹）

用法:
    python data_gen_multi.py                    # 生成全部难度
    python data_gen_multi.py --n 3 4            # 只生成 N=3 和 N=4
    python data_gen_multi.py --sft              # 同时生成 SFT 数据
    python data_gen_multi.py --max-per-n 500    # 每个 N 最多 500 条
"""

import itertools
import json
import os
import random
import re
import argparse
import ast
from fractions import Fraction
from collections import defaultdict


# ============================================================
# 1. 通用 24 点求解器（支持任意 N）
# ============================================================

def can_make_24(nums):
    """
    递归穷举判断 nums 能否通过四则运算得到 24。
    nums: list of Fraction
    """
    if len(nums) == 1:
        return abs(nums[0] - 24) < 1e-9

    for i in range(len(nums)):
        for j in range(len(nums)):
            if i == j:
                continue
            a, b = nums[i], nums[j]
            rest = [nums[k] for k in range(len(nums)) if k != i and k != j]

            # 尝试所有二元运算
            candidates = [a + b, a - b, a * b]
            if b != 0:
                candidates.append(Fraction(a, b))

            for c in candidates:
                if can_make_24(rest + [c]):
                    return True
    return False


def find_24_expression(nums):
    """
    递归搜索一个合法表达式（字符串），使 nums 通过四则运算得到 24。
    返回表达式字符串或 None。
    nums: list of int
    """
    if len(nums) == 1:
        return None  # 不应直接调用单个数字

    # 基于表达式字符串的递归搜索
    return _search_expr([str(n) for n in nums], [Fraction(n) for n in nums])


def _search_expr(exprs, vals):
    """
    exprs: list of str  — 当前子表达式字符串
    vals:  list of Fraction — 对应的数值
    """
    if len(vals) == 1:
        if abs(vals[0] - 24) < 1e-9:
            return exprs[0]
        return None

    for i in range(len(vals)):
        for j in range(len(vals)):
            if i == j:
                continue
            a_expr, a_val = exprs[i], vals[i]
            b_expr, b_val = exprs[j], vals[j]

            rest_exprs = [exprs[k] for k in range(len(vals)) if k != i and k != j]
            rest_vals = [vals[k] for k in range(len(vals)) if k != i and k != j]

            ops = [
                ('+', a_val + b_val),
                ('-', a_val - b_val),
                ('*', a_val * b_val),
            ]
            if b_val != 0:
                ops.append(('/', Fraction(a_val, b_val)))

            for op_sym, result_val in ops:
                # 构造表达式字符串（加括号避免歧义）
                new_expr = f"({a_expr} {op_sym} {b_expr})"
                res = _search_expr(
                    rest_exprs + [new_expr],
                    rest_vals + [result_val]
                )
                if res is not None:
                    return res
    return None


def _simplify_expr(expr_str):
    """
    简化表达式：去掉最外层多余括号。
    例如 "((3 + 5) * (8 - 5))" -> "(3 + 5) * (8 - 5)"
    """
    s = expr_str.strip()
    # 去掉最外层括号（如果匹配）
    if s.startswith('(') and s.endswith(')'):
        depth = 0
        safe = True
        for i, c in enumerate(s):
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            if depth == 0 and i < len(s) - 1:
                safe = False
                break
        if safe:
            s = s[1:-1].strip()
    return s


# ============================================================
# 2. 数据枚举 & 生成
# ============================================================

DIFFICULTY_MAP = {
    3: "easy",
    4: "medium",
    5: "hard",
    6: "expert"
}

NUM_RANGE = range(1, 14)  # 1~13


def enumerate_valid_combinations_exact(n, max_count=None):
    """
    穷举枚举组合能算出 24 的组合。适合 N<=5。
    """
    valid = []
    total_checked = 0

    for comb in itertools.combinations_with_replacement(NUM_RANGE, n):
        total_checked += 1
        if total_checked % 5000 == 0:
            print(f"  [N={n}] 已检查 {total_checked} 个组合，找到 {len(valid)} 个有效...")

        if can_make_24([Fraction(x) for x in comb]):
            valid.append(comb)
            if max_count and len(valid) >= max_count:
                print(f"  [N={n}] 达到上限 {max_count}，停止枚举。")
                break

    print(f"  [N={n}] 穷举完成：共检查 {total_checked} 个，有效 {len(valid)} 个")
    return valid

def enumerate_valid_combinations_sampled(n, max_count=1000):
    """
    拒绝采样（Rejection Sampling），适合 N>=6，避免枚举空间爆炸。
    随机生成 n 个数字的组合，验证是否等于 24，直到收集满 max_count。
    """
    valid = set()
    total_checked = 0

    print(f"  [N={n}] 使用拒绝采样快速生成数据，目标数量: {max_count} ...")
    
    # 为了避免死循环，设置最大尝试次数
    max_attempts = max_count * 500 
    
    while len(valid) < max_count and total_checked < max_attempts:
        total_checked += 1
        
        # 随机组合需要排序后放入 set 去重
        comb = tuple(sorted(random.choices(NUM_RANGE, k=n)))
        
        if comb in valid:
            continue
            
        if can_make_24([Fraction(x) for x in comb]):
            valid.add(comb)
            if len(valid) % 100 == 0:
                print(f"  [N={n}] 已采到 {len(valid)} / {max_count} 个有效组合...")

    print(f"  [N={n}] 采样完成：尝试 {total_checked} 次，有效 {len(valid)} 个")
    return list(valid)

def get_valid_combinations(n, max_count=None):
    if n >= 6:
        # N=6 穷举空间极大 (C(18,6)=18564)，虽然在可接受边缘，但递归解法会非常慢。
        # 因此 N>=6 固定使用拒绝采样，快速保底获取测试集。
        target_count = max_count if max_count else 2000
        return enumerate_valid_combinations_sampled(n, max_count=target_count)
    else:
        return enumerate_valid_combinations_exact(n, max_count=max_count)


def _get_random_failed_paths(digits, target_expr):
    """
    生成带状态追踪(State-Tracking)的错误尝试路径（回溯）。
    """
    ops = [('+', lambda a,b: a+b), ('-', lambda a,b: a-b), ('*', lambda a,b: a*b), ('/', lambda a,b: a/b if b!=0 else float('inf'))]
    paths = []
    
    # 尝试生成 1~2 条错误路径
    num_attempts = random.randint(1, 2)
    
    for _ in range(num_attempts):
        if len(digits) < 2: break
        
        # 复制当前可用数字
        current_digits = list(digits)
        idx1, idx2 = random.sample(range(len(current_digits)), 2)
        d1, d2 = current_digits[idx1], current_digits[idx2]
        op_sym, op_fn = random.choice(ops)
        
        try:
            val1, val2 = float(d1), float(d2)
            res = op_fn(val1, val2)
        except:
            continue
            
        def fmt(v):
            if isinstance(v, float) and v.is_integer(): return str(int(v))
            if isinstance(v, float): return f"{v:.2f}"
            return str(v)
            
        res_str = fmt(res)
        d1_str = fmt(val1)
        d2_str = fmt(val2)
        
        # 从列表中移除用掉的数字，加入新结果
        # 为了保证移除正确，按值移除一次
        current_digits.remove(d1)
        current_digits.remove(d2)
        current_digits.append(res_str)
        
        remain_str = f"[{', '.join(current_digits)}]"
        
        reason = ""
        if res == float('inf') or res < 0:
            reason = "失败（路线无效）。回溯。"
        elif isinstance(res, float) and not res.is_integer() and '/' not in target_expr:
            reason = "失败（产生无法消除的分数）。回溯。"
        elif res > 100:
            reason = "失败（结果过大）。回溯。"
        elif res == 24 and len(current_digits) > 1:
            reason = "失败（未用完所有数字）。回溯。"
        else:
            reason = "失败（无法凑出24）。回溯。"
            
        path_text = f"尝试：\n{d1_str} {op_sym} {d2_str} = {res_str}。剩余: {remain_str}\n{reason}\n\n"
        paths.append(path_text)
        
    return paths

def generate_cot_from_expr(expr_str, provided_digits=None):
    """
    生成状态追踪(State-Tracking)风格的 Long-CoT。
    强制模型输出带有回溯的严谨深度优先搜索格式。
    """
    clean = _simplify_expr(expr_str)
    
    # 获取此题目的所有使用到的数字
    if provided_digits is not None:
        original_digits = provided_digits
    else:
        original_digits = re.findall(r'\d+', clean)
        
    nums_str = ", ".join(original_digits)
    
    cot_parts = []
    cot_parts.append(f"目标：使用 {nums_str} 计算 24。\n\n")
    
    # 1. 插入伪造的回溯/失败尝试
    failed_attempts = _get_random_failed_paths(original_digits, clean)
    for attempt in failed_attempts:
        cot_parts.append(attempt)
        
    cot_parts.append("尝试：\n")
    
    # 2. 真实求解路径 (利用 AST 解析)
    try:
        tree = ast.parse(clean, mode='eval')
    except Exception:
        return f"直接推导：{clean} = 24。"

    steps = []
    op_map = {
        ast.Add: '+', ast.Sub: '-',
        ast.Mult: '*', ast.Div: '/'
    }

    def evaluate_node(node):
        if isinstance(node, ast.Constant):
            return node.value, str(node.value)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            val, _ = evaluate_node(node.operand)
            return -val, str(-val)
        elif isinstance(node, ast.BinOp):
            left_val, left_str = evaluate_node(node.left)
            right_val, right_str = evaluate_node(node.right)
            op = type(node.op)
            op_sym = op_map.get(op, '?')

            if op == ast.Add:
                res = left_val + right_val
            elif op == ast.Sub:
                res = left_val - right_val
            elif op == ast.Mult:
                res = left_val * right_val
            elif op == ast.Div:
                res = left_val / right_val if right_val != 0 else float('inf')
            else:
                res = 0

            def fmt(v):
                if isinstance(v, float) and v == int(v): return str(int(v))
                elif isinstance(v, float): return f"{v:.3f}".rstrip('0').rstrip('.')
                return str(v)

            l_fmt, r_fmt, res_fmt = fmt(left_val), fmt(right_val), fmt(res)
            steps.append((l_fmt, op_sym, r_fmt, res_fmt))
            return res, res_fmt
        return 0, "0"

    evaluate_node(tree.body)

    if not steps:
        return f"通过计算 {clean} 得到 24。"
        
    # 按照严格的状态格式输出真实步骤
    current_state = list(original_digits)
    for idx, (l, op, r, res) in enumerate(steps):
        # 从状态列表中移除用掉的数字（如果有的话，处理多重集）
        if l in current_state: current_state.remove(l)
        if r in current_state: current_state.remove(r)
        current_state.append(res)
        
        remain_str = f"[{', '.join(current_state)}]"
        
        if idx == len(steps) - 1:
            cot_parts.append(f"{l} {op} {r} = {res}。成功！\n")
        else:
            cot_parts.append(f"{l} {op} {r} = {res}。剩余: {remain_str}\n")

    return "".join(cot_parts)


# ============================================================
# 3. 主函数：生成 JSONL 数据
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="多难度24点数据生成器")
    parser.add_argument("--n", nargs='+', type=int, default=[3, 4, 5, 6],
                        help="要生成的 N 值列表（默认: 3 4 5 6）")
    parser.add_argument("--max-per-n", type=int, default=3000,
                        help="每个 N 最多生成多少条（默认: 3000 测试集极大化）")
    parser.add_argument("--sft", action="store_true",
                        help="同时生成 SFT 训练数据（含 CoT 推理轨迹）")
    parser.add_argument("--sft-per-n", type=int, default=200,
                        help="每个 N 生成的 SFT 样本数（默认: 200）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（默认: 42）")
    parser.add_argument("--test-size", type=int, default=500,
                        help="强制划分为测试集的数量（默认: 每个难度 500 道）")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="输出目录（默认: data）")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    all_train = []
    all_test = []
    sft_data = []
    stats = defaultdict(lambda: {"total": 0, "train": 0, "test": 0, "sft": 0})

    for n in args.n:
        print(f"\n{'='*60}")
        print(f"生成 N={n} ({DIFFICULTY_MAP.get(n, 'unknown')}) 的数据...")
        print(f"{'='*60}")

        valid = get_valid_combinations(n, max_count=args.max_per_n)

        if not valid:
            print(f"  ⚠️ N={n} 没有找到任何有效组合！")
            continue

        random.shuffle(valid)
        
        # 保证尽量提取 args.test_size 的验证数据
        actual_test_size = args.test_size
        if len(valid) <= args.test_size:
            actual_test_size = max(1, int(len(valid) * 0.5))
        elif len(valid) - args.test_size < 10:
            actual_test_size = len(valid) - 10

        test_combs = valid[:actual_test_size]
        train_combs = valid[actual_test_size:]

        difficulty = DIFFICULTY_MAP.get(n, f"n{n}")

        for comb in train_combs:
            record = {
                "nums": ", ".join(map(str, comb)),
                "n": n,
                "difficulty": difficulty
            }
            all_train.append(record)

        for comb in test_combs:
            record = {
                "nums": ", ".join(map(str, comb)),
                "n": n,
                "difficulty": difficulty
            }
            all_test.append(record)

        stats[n]["total"] = len(valid)
        stats[n]["train"] = len(train_combs)
        stats[n]["test"] = len(test_combs)

        # SFT 数据生成
        if args.sft:
            sft_count = 0
            sft_target = min(args.sft_per_n, len(train_combs))
            print(f"  正在为 N={n} 生成 SFT 数据（目标: {sft_target} 条）...")

            for comb in train_combs:
                if sft_count >= sft_target:
                    break

                expr = find_24_expression(list(comb))
                if expr is None:
                    continue

                clean_expr = _simplify_expr(expr)
                cot_text = generate_cot_from_expr(expr, provided_digits=[str(c) for c in comb])

                nums_str = ", ".join(map(str, comb))
                # 与 env.py get_prompt() 格式保持一致
                prompt = f"计算24点。\n使用数字 {nums_str}，每个只能用一次。\n先在 <think></think> 内写简短步骤。\n</think> 后只输出最终公式。\n<think>\n"
                response = f"{cot_text}</think>\n{clean_expr}"

                sft_data.append({
                    "nums": nums_str,
                    "n": n,
                    "difficulty": difficulty,
                    "prompt": prompt,
                    "response": response,
                    "expression": clean_expr
                })
                sft_count += 1

            stats[n]["sft"] = sft_count
            print(f"  ✅ N={n} SFT 数据: {sft_count} 条")

    # 写入 JSONL
    train_path = os.path.join(args.output_dir, "train.jsonl")
    test_path = os.path.join(args.output_dir, "test.jsonl")

    # 同时生成向后兼容的 CSV（供现有 train_grpo.py / train_ppo.py 使用）
    train_csv_path = os.path.join(args.output_dir, "train.csv")
    test_csv_path = os.path.join(args.output_dir, "test.csv")

    random.shuffle(all_train)
    random.shuffle(all_test)

    with open(train_path, 'w', encoding='utf-8') as f:
        for record in all_train:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    with open(test_path, 'w', encoding='utf-8') as f:
        for record in all_test:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # 向后兼容 CSV（只含 nums 列）
    import csv
    with open(train_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['nums'])
        for record in all_train:
            writer.writerow([record['nums']])

    with open(test_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['nums'])
        for record in all_test:
            writer.writerow([record['nums']])

    # SFT 数据
    if args.sft and sft_data:
        random.shuffle(sft_data)
        sft_jsonl_path = os.path.join(args.output_dir, "sft_train.jsonl")
        sft_csv_path = os.path.join(args.output_dir, "sft_train.csv")

        with open(sft_jsonl_path, 'w', encoding='utf-8') as f:
            for record in sft_data:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        # 向后兼容 CSV
        with open(sft_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["nums", "prompt", "response"])
            writer.writeheader()
            for record in sft_data:
                writer.writerow({
                    "nums": record["nums"],
                    "prompt": record["prompt"],
                    "response": record["response"]
                })

        print(f"\n📝 SFT 数据: {sft_jsonl_path} ({len(sft_data)} 条)")
        print(f"   CSV 兼容: {sft_csv_path}")

    # 统计摘要
    print(f"\n{'='*60}")
    print("📊 数据生成统计摘要")
    print(f"{'='*60}")
    print(f"{'N':>4} | {'难度':<8} | {'总数':>6} | {'训练':>6} | {'测试':>6} | {'SFT':>6}")
    print("-" * 60)
    total_all = {"total": 0, "train": 0, "test": 0, "sft": 0}
    for n in sorted(stats.keys()):
        s = stats[n]
        d = DIFFICULTY_MAP.get(n, "?")
        print(f"{n:>4} | {d:<8} | {s['total']:>6} | {s['train']:>6} | {s['test']:>6} | {s['sft']:>6}")
        for k in total_all:
            total_all[k] += s[k]
    print("-" * 60)
    print(f"{'合计':>4} | {'':8} | {total_all['total']:>6} | {total_all['train']:>6} | {total_all['test']:>6} | {total_all['sft']:>6}")

    print(f"\n✅ 训练集: {train_path} ({len(all_train)} 条)")
    print(f"✅ 测试集: {test_path} ({len(all_test)} 条)")
    print(f"   CSV 兼容: {train_csv_path}, {test_csv_path}")


if __name__ == "__main__":
    main()
