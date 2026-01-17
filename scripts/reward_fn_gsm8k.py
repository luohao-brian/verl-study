import re
from verl.utils.reward_score import gsm8k

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Optimized Reward function for GSM8K based on official verl implementation.
    Includes:
    1. Last-300-char clipping (Official optimization).
    2. Last-match priority (Official logic).
    3. Multi-level format incentive (Our convergence optimization).
    """
    if data_source != 'gsm8k':
        return 0.0

    # 1. 使用官方内置逻辑计算核心得分
    # format_score=0.1: 格式正确(有数字)但答案错，给0.1分
    # score=1.1: 格式正确且答案对，给1.1分
    # 官方函数内部会自动进行最后300字符截断和取最后一个匹配的操作
    score = gsm8k.compute_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        method='strict',
        format_score=0.1,
        score=1.1
    )
    
    # 2. 兜底逻辑：如果官方逻辑返回0 (说明####后面没数字或没####)
    # 我们额外检查是否至少出现了 "####" 这个符号，给个更小的鼓励分
    if score == 0:
        if "####" in solution_str:
            return 0.05
            
    return score
