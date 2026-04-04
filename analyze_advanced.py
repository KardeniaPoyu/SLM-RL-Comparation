import os
import json
import re
import pandas as pd
import math
import numpy as np

# RegEx for parsing equations in the thinking block
_RE_EQUATION = re.compile(r'([\d\.\s\+\-\*\/\(\)]+)=([\s\-\d\.]+)')
_RE_WHITELIST = re.compile(r'[\d\+\-\*\/\(\)\s\.]+')

def evaluate_intermediate_steps(think_content):
    """
    Evaluates intermediate equations like "A * B = C".
    Returns (num_equations, num_correct, has_hallucination)
    """
    equations = _RE_EQUATION.findall(think_content)
    num_equations = len(equations)
    correct_count = 0
    has_hallucination = False
    
    for left_expr, right_val in equations:
        left_expr = left_expr.strip()
        right_val = right_val.replace('。', '').strip()
        if not left_expr or not right_val:
            continue
            
        if not _RE_WHITELIST.fullmatch(left_expr):
            continue
            
        try:
            left_result = eval(left_expr, {"__builtins__": {}}, {})
            right_result = float(right_val)
            
            if abs(float(left_result) - right_result) < 1e-4:
                correct_count += 1
            else:
                has_hallucination = True
        except Exception:
            has_hallucination = True
            
    return num_equations, correct_count, has_hallucination

def analyze_logic_density(file_path):
    """
    Analyzes a given eval_XXX.jsonl file for logic density and hallucinations.
    """
    if not os.path.exists(file_path):
        return None
        
    stats = {
        'total_samples': 0,
        'correct_samples': 0,
        'total_tokens_approx': 0,
        'total_equations': 0,
        'total_correct_equations': 0,
        'hallucination_samples': 0,
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            stats['total_samples'] += 1
            if data.get('correct', False):
                stats['correct_samples'] += 1
                
            response = data.get('response', '')
            
            # Extract thinking part
            think_content = response
            if '</think>' in response:
                think_content = response.split('</think>')[0]
                
            # Token approx (chars / 2)
            stats['total_tokens_approx'] += len(response) / 2.0
            
            num_eq, corr_eq, has_hal = evaluate_intermediate_steps(think_content)
            stats['total_equations'] += num_eq
            stats['total_correct_equations'] += corr_eq
            if has_hal:
                stats['hallucination_samples'] += 1
                
    if stats['total_samples'] == 0:
        return None
        
    # Calculate derived metrics
    res = {
        'Success Rate (%)': (stats['correct_samples'] / stats['total_samples']) * 100,
        'Mean Length (chars)': (stats['total_tokens_approx'] * 2) / stats['total_samples'],
        'Logic Density (eqs/1k chars)': (stats['total_correct_equations'] / max(1, stats['total_tokens_approx'] * 2)) * 1000,
        'Hallucination Rate (%)': (stats['hallucination_samples'] / stats['total_samples']) * 100,
        'Eq Accuracy (%)': (stats['total_correct_equations'] / max(1, stats['total_equations'])) * 100
    }
    return res

def main():
    log_dir = "logs"
    print(f"{'='*80}")
    print(f"{'Micro-Level Analysis: Logic Density & Hallucination Rate':^80}")
    print(f"{'='*80}")
    
    # Define models and N values
    models = ['sft', 'ppo', 'grpo_G4', 'grpo_G8', 'grpo_G16']
    Ns = [3, 4, 5, 6]
    
    results = []
    
    for model in models:
        for n in Ns:
            # Construct filename like eval_sft_final_3.jsonl
            # For PPO it's eval_ppo_final_3.jsonl
            # For GRPO it's eval_grpo_G16_final_3.jsonl
            file_name = f"eval_{model}_final_{n}.jsonl"
            file_path = os.path.join(log_dir, file_name)
            
            stats = analyze_logic_density(file_path)
            if stats:
                stats['Model'] = model
                stats['N'] = n
                results.append(stats)
                
    if not results:
        print("No evaluation files found!")
        return
        
    df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['Model', 'N', 'Success Rate (%)', 'Mean Length (chars)', 'Logic Density (eqs/1k chars)', 'Hallucination Rate (%)', 'Eq Accuracy (%)']
    df = df[cols]
    
    # Group by model to show average across all Ns (or just show the pivot)
    print("\n[ Full Breakdown by Model and N ]")
    print(df.to_string(index=False, float_format="%.2f"))
    
    print("\n[ Summary: Macros Across N=3,4,5,6 ]")
    summary = df.groupby('Model').agg({
        'Success Rate (%)': 'mean',
        'Mean Length (chars)': 'mean',
        'Logic Density (eqs/1k chars)': 'mean',
        'Hallucination Rate (%)': 'mean'
    }).reset_index()
    
    # Sort for better reading
    summary['sort_key'] = summary['Model'].map({'sft':0, 'ppo':1, 'grpo_G4':2, 'grpo_G8':3, 'grpo_G16':4})
    summary = summary.sort_values('sort_key').drop('sort_key', axis=1)
    print(summary.to_string(index=False, float_format="%.2f"))

    # Save to CSV for the report
    df.to_csv(os.path.join(log_dir, 'logic_density_metrics.csv'), index=False)
    print(f"\nSaved metrics to {os.path.join(log_dir, 'logic_density_metrics.csv')}")

if __name__ == "__main__":
    main()
