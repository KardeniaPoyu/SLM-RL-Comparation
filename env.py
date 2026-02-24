import re
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

class Arithmetic24Env:
    def __init__(self):
        self.transformations = standard_transformations + (implicit_multiplication_application,)
        
    def get_prompt(self, nums_str):
        return f"""你是一个解决24点游戏的顶级数学专家。
请使用给定的四个数字，通过加(+)、减(-)、乘(*)、除(/)和括号()，计算出24。每个数字必须且只能使用一次。

【严格执行规则】
1. 思考过程：必须在 <think></think> 标签内简短地写出寻找24的逻辑推导。提示：除法可能会产生分数作为中间结果（例如 8/3），这是允许且常见的解题技巧。
2. 最终答案：在 </think> 之后，仅输出一行纯数学表达式。
3. 字符白名单：最终的表达式中【只能】包含数字、加减乘除符号和括号，严禁出现等号（=）、汉字、英文字母或任何解释性标点。

正确格式示例：
<think>
目标是凑出24。给定的数字是 3, 6, 8, 2。
我可以尝试先用 3 * 6 得到 18。
剩下的数字是 8 和 2。如果用 18 + 8 = 26。
最后 26 - 2 正好等于 24。
四个数字都用了一次。最终表达式：3 * 6 + 8 - 2。
</think>
3 * 6 + 8 - 2

输入：{nums_str}
输出：
<think>
"""
        
    def _parse_output(self, text):
        has_think = "</think>" in text
        think_count = text.count("</think>")
        
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
        
        # 【新增】返回</think>的出现次数，用于检测多标签问题
        return has_think, pred_expr, think_count

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
        has_think, pred_expr, think_count = self._parse_output(output_text)
        
        reward = 0.0
        
        # ==========================================
        # 【阶段1】：格式检查（严格）
        # ==========================================
        
        # 1.1 检查是否有多个</think>标签（严重违规）
        if think_count > 1:
            # 模型生成了多个</think>标签，说明没理解独一无二的要求
            reward -= 1.0
            return reward, False
        
        # 1.2 基础格式奖励
        if has_think:
            reward += 0.1
        
        # 1.3 检查响应总长度（防止模型废话过多）
        # 正常格式：<think>推理(~50-100字)</think> + 表达式(~30字) = ~150字左右
        # 如果超过300字，说明模型生成了过多无关内容
        if len(output_text.strip()) > 400:
            reward -= 0.3
        
        # ==========================================
        # 【阶段2】：表达式提取和初步检查
        # ==========================================
        
        # 2.1 判空拦截
        if not pred_expr:
            reward -= 0.5
            return reward, False
        
        # 2.2 检查表达式中是否包含多个等号（多步骤计算的症状）
        # 规则规定只能输出单个表达式，如"3 * 6 + 8 - 2"，不能是"3 * 6 = 18; 18 + 8 = 26"
        equals_count = pred_expr.count('=')
        if equals_count > 0:
            # 只要有等号，说明模型输出的是多步计算而不是单个表达式
            reward -= 0.6
            return reward, False
        
        # 2.3 【核心增强】：全文字拦截器
        # 检查提取到的表达式中是否包含汉字 (\u4e00-\u9fa5) 或英文字母 (a-zA-Z)
        if re.search(r'[\u4e00-\u9fa5a-zA-Z]', pred_expr):
            # 只要包含任何废话，直接扣大分，且不进行后续逻辑校验
            reward -= 1.0 
            return reward, False
        
        # 2.4 检查表达式长度是否合理
        # 正常的24点表达式应该在20-80字符之间
        if len(pred_expr) > 100:
            # 表达式过长，可能包含了多步计算或其他垃圾内容
            reward -= 0.4
            # 继续检查数学逻辑
        elif len(pred_expr) < 3:
            # 表达式过短，肯定不对
            reward -= 0.3
            return reward, False
        
        # ==========================================
        # 【阶段3】：逻辑分验证
        # ==========================================
        
        is_correct, reason = self._verify_expression(pred_expr, target_nums)
        if is_correct:
            reward += 1.0
        else:
            # 只要进到这里，说明格式是对的（纯公式），但数学逻辑错了
            if reason == "Math error":
                reward -= 0.5
            elif reason == "Parse error":
                reward -= 0.4
            elif reason == "Used wrong numbers":
                reward -= 0.5
            elif reason == "Empty expression":
                reward -= 0.5
            elif reason == "Invalid characters":
                reward -= 0.6
            elif reason == "Exponentiation not allowed":
                reward -= 0.6
            else:
                reward -= 0.3
                
        return reward, is_correct

if __name__ == "__main__":
    env = Arithmetic24Env()
    prompt = env.get_prompt("3, 3, 8, 8")
    print("Prompt Preview:\n", prompt)
    print("\n" + "="*80)
    print("测试改进的奖励函数")
    print("="*80)
    
    # 测试1：正确答案
    sample_out = "</think>\n8 / (3 - 8/3)"
    reward, is_correct = env.compute_reward("3, 3, 8, 8", sample_out)
    print(f"\n✅ 正确答案: '8 / (3 - 8/3)'")
    print(f"   奖励: {reward:.2f}, 正确: {is_correct}")

    # 测试2：错误的数字
    sample_out_bad = "</think>\n8 * 3"
    reward, is_correct = env.compute_reward("3, 3, 8, 8", sample_out_bad)
    print(f"\n❌ 错误数字: '8 * 3' (应该用3, 3, 8, 8)")
    print(f"   奖励: {reward:.2f}, 正确: {is_correct}")
    
    # 测试3：包含等号（不规范格式）
    sample_out_with_equals = "</think>\n8 / (3 - 8/3) = 24"
    reward, is_correct = env.compute_reward("3, 3, 8, 8", sample_out_with_equals)
    print(f"\n⚠️  包含等号: '8 / (3 - 8/3) = 24'")
    print(f"   奖励: {reward:.2f}, 正确: {is_correct}")
    
    # 测试4：多个</think>标签
    sample_out_multiple_tags = "</think>\n计算过程</think>\n8 / (3 - 8/3)"
    reward, is_correct = env.compute_reward("3, 3, 8, 8", sample_out_multiple_tags)
    print(f"\n❌ 多个标签: '......</think>......</think>......'")
    print(f"   奖励: {reward:.2f}, 正确: {is_correct}")