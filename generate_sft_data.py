import csv
import itertools
import sympy

def solve_24(nums):
    """Find a valid expression for 24 using 4 numbers."""
    nums = [int(x) for x in nums]
    
    # helper for binary ops
    ops = [
        ('+', lambda a,b: a+b),
        ('-', lambda a,b: a-b),
        ('*', lambda a,b: a*b),
        ('/', lambda a,b: a/b if b != 0 else None)
    ]
    
    templates = [
        "({} {} {}) {} ({} {} {})",
        "(({} {} {}) {} {}) {} {}",
        "({} {} ({} {} {})) {} {}",
        "{} {} (({} {} {}) {} {})",
        "{} {} ({} {} ({} {} {}))"
    ]
    
    for perm in itertools.permutations(nums):
        for op1, op2, op3 in itertools.product([op[0] for op in ops], repeat=3):
            for tmpl in templates:
                expr = tmpl.format(perm[0], op1, perm[1], op2, perm[2], op3, perm[3])
                try:
                    val = eval(expr)
                    if abs(val - 24.0) < 1e-5:
                        return expr
                except:
                    pass
    return None

def is_hard(expr):
    """
    判断题目是否为“困难”：包含除法，并且产生了中间分数（例如不能整除的情况），
    或者含有比较复杂的嵌套组合如 (a)/(b/c) 或 (a-b/c)*d
    简单的近似判断：包含除法且不仅仅是类似 24/1 或 8*3=24
    """
    if not expr:
        return False
    # 这里通过判断表达式结构或特定组合来近似
    # 有除法说明有一定几率是分数题
    if "/" in expr:
        # 给一定概率视为 hard
        return True
    return False

import ast

def generate_cot(expr_str):
    """
    解析四则运算表达式的AST，并生成一步步的自然语言计算过程。
    例如：输入 "(1 + 7) * (11 - 8)"
    输出："首先计算 1 + 7 = 8。然后计算 11 - 8 = 3。最后计算 8 * 3 = 24。"
    """
    try:
        tree = ast.parse(expr_str, mode='eval')
    except Exception:
        return f"通过尝试不同的组合：{expr_str} 可以计算出 24。"
    
    steps = []
    
    # 定义操作符映射
    op_map = {
        ast.Add: '+',
        ast.Sub: '-',
        ast.Mult: '*',
        ast.Div: '/'
    }
    
    def evaluate_node(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left_val = evaluate_node(node.left)
            right_val = evaluate_node(node.right)
            op = type(node.op)
            op_sym = op_map.get(op, '?')
            
            # 计算当前步骤的结果
            if op == ast.Add:
                res = left_val + right_val
            elif op == ast.Sub:
                res = left_val - right_val
            elif op == ast.Mult:
                res = left_val * right_val
            elif op == ast.Div:
                res = left_val / right_val if right_val != 0 else float('inf')
            
            # 格式化一下浮点数，比如 8/3 或者整除的情况
            # 为了易读，如果接近整数就用整数，否则保留两位小数或分数（这里简化为两位小数或整数）
            if isinstance(left_val, float) and left_val.is_integer(): left_val = int(left_val)
            if isinstance(right_val, float) and right_val.is_integer(): right_val = int(right_val)
            if isinstance(res, float) and res.is_integer(): res = int(res)
            
            if isinstance(res, float):
                step_str = f"计算 {left_val} {op_sym} {right_val} ≈ {res:.2f}"
            else:
                step_str = f"计算 {left_val} {op_sym} {right_val} = {res}"
            
            steps.append(step_str)
            return res
        return 0

    evaluate_node(tree.body)
    
    if not steps:
        return f"通过尝试不同的组合：{expr_str} 可以计算出 24。"
        
    cot_texts = []
    for idx, step in enumerate(steps):
        if idx == 0:
            cot_texts.append(f"首先{step}。")
        elif idx == len(steps) - 1:
            cot_texts.append(f"最后{step}。")
        else:
            cot_texts.append(f"然后{step}。")
            
    return "".join(cot_texts)
    
def main():
    with open('data/train.csv', 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    out_rows = []
    
    # 我们需要 500 条数据，其中 20% 是 hard，即 100 条 hard
    target_total = 500
    target_hard = int(target_total * 0.20)
    
    hard_count = 0
    normal_count = 0
    
    for row in rows:
        if hard_count >= target_hard and normal_count >= (target_total - target_hard):
            break
            
        nums_str = row['nums']
        nums = [n.strip() for n in nums_str.split(',')]
        
        expr = solve_24(nums)
        if expr:
            # 判断难度
            hard_flag = is_hard(expr)
            
            # 控制比例
            if hard_flag and hard_count >= target_hard:
                continue
            if not hard_flag and normal_count >= (target_total - target_hard):
                continue
                
            if hard_flag:
                hard_count += 1
            else:
                normal_count += 1

            prompt = f"{nums_str} 算24\n<think>简短推理</think>后只输出表达式\n<think>\n"
            
            # 生成真实逻辑推导
            cot_text = generate_cot(expr)
            ans = f"{cot_text}</think>\n{expr}"
            
            out_rows.append({"nums": nums_str, "prompt": prompt, "response": ans})
            
    with open('data/sft_train.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["nums", "prompt", "response"])
        writer.writeheader()
        writer.writerows(out_rows)
        
    print(f"Generated {len(out_rows)} SFT examples saved to data/sft_train.csv")
    print(f"- Hard examples (contains '/'): {hard_count}")
    print(f"- Normal examples: {normal_count}")

if __name__ == "__main__":
    main()
