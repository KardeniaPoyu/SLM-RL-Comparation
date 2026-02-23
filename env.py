import re
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

class Arithmetic24Env:
    def __init__(self):
        self.transformations = standard_transformations + (implicit_multiplication_application,)
        
    def get_prompt(self, nums_str):
        return f"Input: {nums_str}.\nOutput: <think>\n"
        
    def _parse_output(self, text):
        # Match <think>...</think> and extract the expression after it
        think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        has_think = think_match is not None
        
        if has_think:
            pred_expr = text[think_match.end():].strip()
            # In case the model continues generation, just take the first line
            pred_expr = pred_expr.split('\n')[0].strip()
        else:
            # If no </think> token, extract whatever we can
            pred_expr = text.strip().split('\n')[-1].strip()
            
        return has_think, pred_expr

    def _verify_expression(self, expr_str, target_nums):
        # Strip trailing '=' or '24' things that the model might generate
        expr_str = expr_str.split('=')[0].strip()
        
        digits = re.findall(r'\d+', expr_str)
        try:
            used_nums = sorted([int(d) for d in digits])
        except ValueError:
            return False, "Parse error"
            
        sorted_target = sorted([int(n) for n in target_nums])
        if used_nums != sorted_target:
            return False, "Used wrong numbers"
            
        try:
            parsed = parse_expr(expr_str, transformations=self.transformations, evaluate=True)
            if abs(float(parsed) - 24.0) < 1e-6:
                return True, "Correct"
            else:
                return False, "Wrong value"
        except Exception:
            return False, "Math error"

    def compute_reward(self, input_nums_str, output_text):
        target_nums = [n.strip() for n in input_nums_str.split(',')]
        
        has_think, pred_expr = self._parse_output(output_text)
        
        reward = 0.0
        
        if has_think:
            reward += 0.1
            
        if not pred_expr:
            reward -= 0.5
            return reward, False
            
        is_correct, reason = self._verify_expression(pred_expr, target_nums)
        if is_correct:
            reward += 1.0
        else:
            if reason in ["Math error", "Parse error", "Used wrong numbers"]:
                reward -= 0.5
                
        return reward, is_correct

if __name__ == "__main__":
    env = Arithmetic24Env()
    prompt = env.get_prompt("3, 3, 8, 8")
    print("Prompt:", prompt)
    
    sample_out = "<think>\n8 / (3 - 8/3) = 8 / (1/3) = 24\n</think>\n8 / (3 - 8/3)"
    reward, is_correct = env.compute_reward("3, 3, 8, 8", sample_out)
    print(f"Sample OK Output -> Reward: {reward}, Correct: {is_correct}")

    sample_out_bad = "<think>\n8 * 3 = 24\n</think>\n8 * 3"
    reward, is_correct = env.compute_reward("3, 3, 8, 8", sample_out_bad)
    print(f"Sample Bad Numbers Output -> Reward: {reward}, Correct: {is_correct}")
