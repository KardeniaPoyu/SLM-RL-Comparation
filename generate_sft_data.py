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

import sys
sys.path.insert(0, '.')
from env import Arithmetic24Env

def parse_and_format_expr(expr):
    # just return formatted
    return expr

def main():
    env = Arithmetic24Env()
    
    with open('data/train.csv', 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    out_rows = []
    
    # We just need some samples for SFT, let's say 500
    count = 0
    for row in rows:
        if count >= 500:
            break
        nums_str = row['nums']
        nums = [n.strip() for n in nums_str.split(',')]
        
        expr = solve_24(nums)
        if expr:
            # Generate the expected format
            prompt = env.get_prompt(nums_str)
            
            # Remove redundant parentheses from eval logic (optional but better)
            # Actually simplest logic: 
            ans = f"""目标是凑出24。给定的数字是 {nums_str}。
通过尝试不同的组合，我找到了以下有效表达式：
{expr} 可以计算出 24。
四个数字都用了一次，逻辑正确。
</think>
{expr}"""
            out_rows.append({"nums": nums_str, "prompt": prompt, "response": ans})
            count += 1
            
    with open('data/sft_train.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["nums", "prompt", "response"])
        writer.writeheader()
        writer.writerows(out_rows)
        
    print(f"Generated {len(out_rows)} SFT examples saved to data/sft_train.csv")

if __name__ == "__main__":
    main()
