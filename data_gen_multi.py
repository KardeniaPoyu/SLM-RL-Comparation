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


def enumerate_valid_combinations(n, max_count=None):
    """
    枚举所有可重复组合 C(13+n-1, n) 中能算出 24 的组合。
    返回 list of tuple。
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

    print(f"  [N={n}] 完成：共检查 {total_checked} 个，有效 {len(valid)} 个")
    return valid


def generate_cot_from_expr(expr_str):
    """
    将带括号的表达式解析为 AST，并生成逐步计算的中文推理链。
    例如: "(3 + 5) * (8 - 5)" ->
         "首先计算 3 + 5 = 8。然后计算 8 - 5 = 3。最后计算 8 * 3 = 24。"
    """
    clean = _simplify_expr(expr_str)
    try:
        tree = ast.parse(clean, mode='eval')
    except Exception:
        return f"通过计算 {clean} 得到 24。"

    steps = []
    op_map = {
        ast.Add: '+', ast.Sub: '-',
        ast.Mult: '*', ast.Div: '/'
    }

    def evaluate_node(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -evaluate_node(node.operand)
        elif isinstance(node, ast.BinOp):
            left_val = evaluate_node(node.left)
            right_val = evaluate_node(node.right)
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

            # 美化数字显示
            def fmt(v):
                if isinstance(v, float) and v == int(v):
                    return str(int(v))
                elif isinstance(v, float):
                    return f"{v:.2f}"
                return str(v)

            steps.append(f"计算 {fmt(left_val)} {op_sym} {fmt(right_val)} = {fmt(res)}")
            return res
        return 0

    evaluate_node(tree.body)

    if not steps:
        return f"通过计算 {clean} 得到 24。"

    cot_parts = []
    for idx, step in enumerate(steps):
        if idx == 0:
            cot_parts.append(f"首先{step}。")
        elif idx == len(steps) - 1:
            cot_parts.append(f"最后{step}。")
        else:
            cot_parts.append(f"然后{step}。")

    return "".join(cot_parts)


# ============================================================
# 3. 主函数：生成 JSONL 数据
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="多难度24点数据生成器")
    parser.add_argument("--n", nargs='+', type=int, default=[3, 4, 5, 6],
                        help="要生成的 N 值列表（默认: 3 4 5 6）")
    parser.add_argument("--max-per-n", type=int, default=None,
                        help="每个 N 最多生成多少条（默认: 不限制）")
    parser.add_argument("--sft", action="store_true",
                        help="同时生成 SFT 训练数据（含 CoT 推理轨迹）")
    parser.add_argument("--sft-per-n", type=int, default=200,
                        help="每个 N 生成的 SFT 样本数（默认: 200）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（默认: 42）")
    parser.add_argument("--split-ratio", type=float, default=0.9,
                        help="训练集占比（默认: 0.9）")
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

        valid = enumerate_valid_combinations(n, max_count=args.max_per_n)

        if not valid:
            print(f"  ⚠️ N={n} 没有找到任何有效组合！")
            continue

        random.shuffle(valid)
        split_idx = int(len(valid) * args.split_ratio)
        train_combs = valid[:split_idx]
        test_combs = valid[split_idx:]

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
                cot_text = generate_cot_from_expr(expr)

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
