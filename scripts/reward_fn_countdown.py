import re
import ast
import operator
import os
import random

def evaluate_expression(expr):
    allowed_ops = {
        ast.Add: operator.add, ast.Sub: operator.sub, 
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg
    }
    try:
        expr = expr.replace('[', '(').replace(']', ')')
        tree = ast.parse(expr, mode='eval')
        def _eval(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.Num):
                return node.n
            elif isinstance(node, ast.BinOp):
                op_type = type(node.op)
                if op_type not in allowed_ops:
                    raise TypeError(f"Operator {op_type} not allowed")
                return allowed_ops[op_type](_eval(node.left), _eval(node.right))
            elif isinstance(node, ast.UnaryOp):
                op_type = type(node.op)
                if op_type not in allowed_ops:
                     raise TypeError(f"Operator {op_type} not allowed")
                return allowed_ops[op_type](_eval(node.operand))
            else:
                raise TypeError(f"Node type {type(node)} not allowed")
        return _eval(tree.body)
    except Exception:
        return None

def log_completion(solution_str, target, nums):
    """Log model completions to a file at a 10% sampling frequency."""
    log_path = os.environ.get('COMPLETION_LOG_PATH')
    if log_path and random.random() < 0.1:
        try:
            with open(log_path, 'a') as f:
                f.write(f"Target: {target} | Nums: {nums}\n")
                f.write(f"Solution:\n{solution_str}\n")
                f.write("-" * 40 + "\n")
        except Exception:
            pass

def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1., **kwargs):
    target = None
    nums = None
    if isinstance(ground_truth, dict):
        target = ground_truth.get('target')
        nums = ground_truth.get('numbers')
    
    if target is None or nums is None:
        return 0.0

    # Sampling logging
    log_completion(solution_str, target, nums)

    try:
        target = float(target)
        nums = [int(x) for x in nums]
    except:
        return 0.0

    reward = 0.0
    
    # 1. Format: closed think tag
    if "</think>" in solution_str:
        reward += 0.1
    
    # 2. Format: answer tags
    answer_match = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    if answer_match:
        reward += 0.1
        answer_content = answer_match.group(1).strip()
    else:
        # If no answer tag, we can't easily verify the result
        return reward

    # 3. Constraint: Use correct numbers
    try:
        used_nums = [int(n) for n in re.findall(r"\d+", answer_content)]
        if sorted(used_nums) == sorted(nums):
            reward += 0.3
        else:
            # If numbers are wrong, we still might check result but usually 
            # countdown requires using exact numbers. 
            # For signal, we stop here.
            return reward
            
        # 4. Correctness: Result matches target
        result = evaluate_expression(answer_content)
        if result is not None and abs(float(result) - target) < 1e-5:
            reward += 0.5
            
    except Exception:
        pass
        
    return reward
