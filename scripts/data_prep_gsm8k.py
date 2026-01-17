import datasets
import pandas as pd
import os
import re

def extract_answer(answer_str):
    # Extract number after ####
    if '####' in answer_str:
        return answer_str.split('####')[-1].strip()
    return answer_str

def process_dataset(split_name, dataset):
    data_list = []
    for example in dataset:
        question = example['question']
        answer = example['answer']
        ground_truth = extract_answer(answer)
        
        # Format as chat list
        # prompt = [{"role": "user", "content": "Question: " + question + "\nLet's think step by step."}]
        # Actually verl legacy_data might expect a string if return_raw_chat is False, 
        # or list if it applies template.
        # Let's provide the chat list format which is safer for modern tokenizers.
        
        # For base models, sometimes we just want the raw string.
        # But Qwen supports chat templates.
        
        data_item = {
            "prompt": [
                {"role": "system", "content": "You are a math assistant. For every question, you must think step by step and provide the final numerical answer after '#### ' at the end."},
                {"role": "user", "content": f"Question: {question}\nAnswer:"}
            ],
            "data_source": "gsm8k",
            "reward_model": {"ground_truth": ground_truth},
            "extra_info": {"original_answer": answer}
        }
        data_list.append(data_item)
    return pd.DataFrame(data_list)

def prep_data():
    print("Loading GSM8K...")
    ds = datasets.load_dataset("gsm8k", "main")
    
    print("Processing Train...")
    train_df = process_dataset("train", ds['train'])
    
    print("Processing Test...")
    test_df = process_dataset("test", ds['test'])
    
    os.makedirs('artifacts/data/gsm8k', exist_ok=True)
    print("Saving to parquet...")
    train_df.to_parquet('artifacts/data/gsm8k/train.parquet')
    test_df.to_parquet('artifacts/data/gsm8k/test.parquet')
    print("Done.")

if __name__ == "__main__":
    prep_data()
