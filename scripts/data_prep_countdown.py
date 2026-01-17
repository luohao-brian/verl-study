"""
Preprocess dataset for countdown task - given a target number and N numbers, generate equations to reach target
"""

import os
from datasets import load_dataset
import argparse

def make_content(dp):
    target = dp['target']
    numbers = dp['nums']
    return f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='artifacts/data/countdown')
    parser.add_argument('--train_size', type=int, default=327680)
    parser.add_argument('--test_size', type=int, default=1024)

    args = parser.parse_args()

    data_source = 'countdown'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    raw_dataset = load_dataset('Jiayi-Pan/Countdown-Tasks-3to4', split='train')

    assert len(raw_dataset) > TRAIN_SIZE + TEST_SIZE
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))

    def make_map_fn(split):
        def process_fn(example, idx):
            user_content = make_content(example)
            system_content = "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
            
            # Ensure numbers are a clean list for JSON/Parquet compatibility
            nums = example['nums']
            if hasattr(nums, 'tolist'):
                nums = nums.tolist()
            else:
                nums = list(nums)
                
            solution = {
                "target": float(example['target']),
                "numbers": nums
            }
            
            data = {
                "data_source": data_source,
                "prompt": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))