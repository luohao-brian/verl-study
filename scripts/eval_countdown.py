import argparse
import pandas as pd
import numpy as np
import os
import gc
import ray
import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import reward_fn_countdown as reward_fn

# Suppress Ray and NCCL errors
os.environ["RAY_IGNORE_UNHANDLED_ERRORS"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

class CountdownEvaluator:
    def __init__(self, model_path, tp=8):
        print(f"Initializing Tokenizer and vLLM Engine (TP={tp})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.llm = LLM(model=model_path, tensor_parallel_size=tp, trust_remote_code=True)

    def evaluate(self, data_path, rollout_n=1, temperature=0.7, limit=50):
        print(f"Reading evaluation data: {data_path}")
        df = pd.read_parquet(data_path)
        if limit > 0:
            df = df.head(limit)
        
        # Prepare prompts
        prompts = df['prompt'].tolist()
        formatted_prompts = []
        for p in prompts:
            # Apply standard chat template
            text = self.tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            formatted_prompts.append(text)
        
        sampling_params = SamplingParams(
            n=rollout_n,
            temperature=temperature if rollout_n > 1 else 0,
            max_tokens=1024,
            stop=["</answer>"], # Stop after answer tag closure
            include_stop_str_in_output=True
        )
        
        print(f"Generating {rollout_n} samples for {len(df)} inputs...")
        start_time = time.time()
        outputs = self.llm.generate(formatted_prompts, sampling_params)
        print(f"Generation took {time.time() - start_time:.2f}s")
        
        results = []
        log_file = "logs/completion_countdown_eval.log"
        success_log_file = "logs/completion_countdown_eval_success.log"
        os.makedirs("logs", exist_ok=True)
        
        # Clear previous logs
        with open(log_file, "w") as f: f.write("--- Eval Group Failures (Group Score = 0) ---\n\n")
        with open(success_log_file, "w") as f: f.write("--- Eval Group Successes (Group Score > 0) ---\n\n")
        
        with open(log_file, "a") as f_fail, open(success_log_file, "a") as f_success:
            for i, output in enumerate(outputs):
                # Extract ground truth from the new format
                reward_model_data = df.iloc[i]['reward_model']
                ground_truth = reward_model_data.get('ground_truth')
                
                scores = []
                completions = []
                
                for completion in output.outputs:
                    # Pass the ground_truth dict to the updated reward function
                    # Note: completion.text typically DOES NOT include the prompt/pre-fill.
                    # So it starts after <think>. 
                    # We need to ensure reward_fn handles this if it expects <think> tag.
                    # Our current reward_fn gives points for <think> existence.
                    # If vLLM generation *continues*, it won't repeat <think>.
                    # We might need to prepend the pre-fill to the completion for scoring, 
                    # OR update reward_fn to not be strict about <think> if we are pre-filling it.
                    
                    # Let's check reward_fn. It gives 0.125 for </think> or <answer>.
                    # It doesn't strictly require <think> at start.
                    # But verifying: completion.text will likely start with content inside think or just after.
                    
                    # To be safe and consistent with "Show your work in <think>...", 
                    # let's prepend the missing start tag if needed for scoring analysis,
                    # or just pass the completion. 
                    
                    # Actually, if we force "<think>", the model generates the *content* of think.
                    # It might generate "</think>" later.
                    
                    score = reward_fn.compute_score(completion.text, ground_truth=ground_truth)
                    scores.append(score)
                    completions.append(completion.text)
                
                mean_s = np.mean(scores)
                var_s = np.var(scores)
                max_s = np.max(scores)
                
                norm_mean = min(max(mean_s, 0), 1)
                if norm_mean <= 0 or norm_mean >= 1:
                    entropy = 0.0
                else:
                    entropy = -(norm_mean * np.log2(norm_mean) + (1 - norm_mean) * np.log2(1 - norm_mean))

                # Extract user prompt content for logging
                # p is a list of dicts
                prompt_content = df.iloc[i]['prompt'][1]['content'] 
                
                target = ground_truth.get('target')
                nums = ground_truth.get('numbers')

                if max_s == 0:
                    f_fail.write(f"PROMPT: {prompt_content}\n")
                    f_fail.write(f"TARGET: {target} | NUMS: {nums}\n")
                    f_fail.write(f"BEST SAMPLE: {output.outputs[0].text[:500].replace('\n', ' ')}...\n")
                    f_fail.write("-" * 50 + "\n")
                else:
                    f_success.write(f"PROMPT: {prompt_content}\n")
                    f_success.write(f"TARGET: {target} | NUMS: {nums}\n")
                    best_idx = np.argmax(scores)
                    f_success.write(f"BEST SAMPLE (Score: {scores[best_idx]}): {completions[best_idx][:4000]}\n")
                    f_success.write("-" * 50 + "\n")

                results.append({
                    "mean_score": mean_s,
                    "max_score": max_s,
                    "var": var_s,
                    "entropy": entropy
                })
            
        res_df = pd.DataFrame(results)
        print(f"\n" + "="*40)
        print(f"STRICT ALIGNED EVALUATION REPORT (n={rollout_n})")
        print(f"-"*40)
        print(f"Mean Accuracy (Signal): {res_df['mean_score'].mean():.2%}")
        pass_at_n = res_df[res_df['max_score'] >= 1.0].shape[0] / len(res_df)
        print(f"Max Accuracy (Pass@{rollout_n}): {pass_at_n:.2%}")
        print(f"Avg Group Variance: {res_df['var'].mean():.4f}")
        print(f"Avg Group Entropy: {res_df['entropy'].mean():.4f}")
        print("="*40 + "\n")
        print(f"Failures saved to: {log_file}")
        print(f"Successes saved to: {success_log_file}")
        
    def cleanup(self):
        print("Starting graceful shutdown...")
        if hasattr(self, 'llm'):
            del self.llm
        gc.collect()
        if ray.is_initialized():
            ray.shutdown()
        time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tp", type=int, default=1) 
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--rollout-n", type=int, default=1, dest="rollout_n")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    
    if "COMPLETION_LOG_PATH" in os.environ: del os.environ["COMPLETION_LOG_PATH"]
    
    evaluator = CountdownEvaluator(args.model_path, args.tp)
    try:
        evaluator.evaluate("artifacts/data/countdown/test.parquet", 
                          rollout_n=args.rollout_n, 
                          temperature=args.temperature, 
                          limit=args.limit)
    finally:
        evaluator.cleanup()
