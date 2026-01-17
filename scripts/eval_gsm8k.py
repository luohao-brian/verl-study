import argparse
import pandas as pd
import numpy as np
import os
import math
import time
import gc
from datasets import load_dataset
from vllm import LLM, SamplingParams
from verl.utils.reward_score import gsm8k

class Metrics:
    @staticmethod
    def calculate_entropy(p):
        if p <= 0 or p >= 1:
            return 0
        return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

    @classmethod
    def get_group_stats(cls, scores):
        mean = np.mean(scores)
        var = np.var(scores)
        entropy = cls.calculate_entropy(mean)
        return mean, var, entropy

class GSM8KTask:
    @staticmethod
    def get_prompt(question):
        return (
            "<|im_start|>system\n"
            "You are a math assistant. For every question, you must think step by step and provide the final numerical answer after '#### ' at the end.\n"
            "<|im_end|>\n"
            f"<|im_start|>user\nQuestion: {question}\nAnswer:<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    @staticmethod
    def extract_answer(text):
        if "####" in text:
            return text.split("####")[-1].strip()
        return ""

    @classmethod
    def score(cls, generated_text, ground_truth):
        gt_val = cls.extract_answer(ground_truth)
        score = gsm8k.compute_score(
            solution_str=generated_text,
            ground_truth=gt_val,
            method='strict',
            format_score=0.1,
            score=1.0
        )
        return 1.0 if score >= 1.0 else 0.0

class Evaluator:
    def __init__(self, model_path, tp=1):
        self.model_path = model_path
        self.tp = tp
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            print(f"Initializing vLLM Engine (TP={self.tp})...")
            self._llm = LLM(model=self.model_path, tensor_parallel_size=self.tp, trust_remote_code=True)
        return self._llm

    def cleanup(self):
        if self._llm:
            print("Cleaning up vLLM Engine...")
            del self._llm
            self._llm = None
            gc.collect()
            time.sleep(2)

    def load_data(self, limit=None, seed=42):
        print("Loading Dataset...")
        ds = load_dataset("gsm8k", "main", split="test")
        if seed is not None:
            ds = ds.shuffle(seed=seed)
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))
        return ds

    def run_reward(self, limit, seed, output_path):
        ds = self.load_data(limit, seed)
        prompts = [GSM8KTask.get_prompt(ex['question']) for ex in ds]
        
        sampling_params = SamplingParams(temperature=0, max_tokens=1024, stop=["<|im_end|>"])
        outputs = self.llm.generate(prompts, sampling_params)
        
        results = []
        correct = 0
        for i, output in enumerate(outputs):
            text = output.outputs[0].text
            is_correct = GSM8KTask.score(text, ds[i]['answer'])
            correct += is_correct
            results.append({
                "question": ds[i]['question'],
                "response": text,
                "ground_truth": ds[i]['answer'],
                "correct": bool(is_correct)
            })
        
        print(f"\n--- Reward Accuracy: {correct/len(ds):.2%} ---")
        self._save(results, output_path)
        self.cleanup()

    def run_analyse(self, rollout_n, temperature, limit, seed, output_path):
        ds = self.load_data(limit, seed)
        prompts = [GSM8KTask.get_prompt(ex['question']) for ex in ds]
        
        sampling_params = SamplingParams(
            n=rollout_n, 
            temperature=temperature, 
            max_tokens=1024, 
            stop=["<|im_end|>"]
        )
        outputs = self.llm.generate(prompts, sampling_params)
        
        results = []
        group_stats = []
        for i, output in enumerate(outputs):
            scores = [GSM8KTask.score(c.text, ds[i]['answer']) for c in output.outputs]
            m, v, e = Metrics.get_group_stats(scores)
            group_stats.append((m, v, e))
            results.append({
                "question": ds[i]['question'],
                "mean_reward": m,
                "reward_var": v,
                "entropy": e,
                "success_count": sum(scores),
                "total_count": rollout_n
            })
            
        print(f"\n--- GRPO Analysis (rollout_n={rollout_n}, temp={temperature}) ---")
        print(f"Avg Accuracy: {np.mean([x[0] for x in group_stats]):.2%}")
        print(f"Avg Variance: {np.mean([x[1] for x in group_stats]):.4f}")
        print(f"Avg Entropy:  {np.mean([x[2] for x in group_stats]):.4f}")
        self._save(results, output_path)
        self.cleanup()

    @staticmethod
    def run_stats(input_path, var_min=0.05, entropy_min=0.2, limit=10):
        if not os.path.exists(input_path):
            print(f"Error: Analysis file {input_path} not found. Run 'analyse' first.")
            return
        
        df = pd.read_csv(input_path)
        col_map = {'group_reward_var': 'reward_var', 'group_entropy': 'entropy', 'group_mean_reward': 'mean_reward'}
        df = df.rename(columns=col_map)

        if 'reward_var' not in df.columns or 'entropy' not in df.columns:
            print(f"Error: Required columns not found. Available: {df.columns.tolist()}")
            return
        
        high_value = df[(df['reward_var'] >= var_min) & (df['entropy'] >= entropy_min)]
        high_value = high_value.sort_values(by='reward_var', ascending=False)
        
        print(f"\n--- High-Value Training Samples (Top {limit}) ---")
        print(f"Found {len(high_value)} samples with Var >= {var_min} and Entropy >= {entropy_min}\n")
        
        for i, row in high_value.head(limit).iterrows():
            print(f"[{i}] Var: {row['reward_var']:.3f} | Entropy: {row['entropy']:.3f} | Acc: {row['mean_reward']:.2%}")
            print(f"Question: {row['question'][:150]}...\n")

    def _save(self, data, path):
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        pd.DataFrame(data).to_csv(path, index=False)
        print(f"Results saved to {path}")

def main():
    parser = argparse.ArgumentParser(description="GSM8K Evaluation & Analysis Tool")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("reward")
    
    analyse_parser = subparsers.add_parser("analyse")
    analyse_parser.add_argument("--rollout_n", type=int, default=16)
    analyse_parser.add_argument("--temperature", type=float, default=1.0)
    
    stats_parser = subparsers.add_parser("stats")
    stats_parser.add_argument("--input", type=str, default="artifacts/results/analyse_eval.csv")
    stats_parser.add_argument("--var_min", type=float, default=0.05)
    stats_parser.add_argument("--entropy_min", type=float, default=0.2)
    stats_parser.add_argument("--limit", type=int, default=10)
    
    args = parser.parse_args()
    
    if args.command == "stats":
        Evaluator.run_stats(args.input, args.var_min, args.entropy_min, args.limit)
        return

    evaluator = Evaluator(args.model_path, args.tp)
    try:
        if args.command == "reward":
            output = args.output or "artifacts/results/reward_eval.csv"
            evaluator.run_reward(args.limit, args.seed, output)
        elif args.command == "analyse":
            output = args.output or "artifacts/results/analyse_eval.csv"
            evaluator.run_analyse(args.rollout_n, args.temperature, args.limit, args.seed, output)
    finally:
        evaluator.cleanup()

if __name__ == "__main__":
    main()