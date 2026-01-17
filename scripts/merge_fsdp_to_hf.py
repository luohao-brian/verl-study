import torch
import os
import argparse
from transformers import AutoModelForCausalLM, AutoConfig
from tqdm import tqdm

def merge_fsdp_to_hf(fsdp_path, output_path):
    world_size = 8
    print(f"Loading FSDP shards from {fsdp_path} (World Size: {world_size})...")
    
    # 1. 加载配置
    hf_path = os.path.join(fsdp_path, 'huggingface')
    config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
    
    # 2. 预加载所有分片
    shards = []
    print("Step 1/3: Loading Shards")
    for rank in tqdm(range(world_size), desc="Loading ranks"):
        shard_path = os.path.join(fsdp_path, f'model_world_size_8_rank_{rank}.pt')
        if not os.path.exists(shard_path):
            print(f"Warning: Shard {shard_path} not found, skipping.")
            continue
        shard = torch.load(shard_path, map_location='cpu', weights_only=False)
        # 移除 _orig_mod. 前缀
        shard = {k.replace("_orig_mod.", ""): v for k, v in shard.items()}
        shards.append(shard)
    
    if not shards:
        print("Error: No shards loaded!")
        return

    # 3. 合并逻辑
    print("\nStep 2/3: Merging Shards")
    full_state_dict = {}
    all_keys = list(shards[0].keys())
    
    for key in tqdm(all_keys, desc="Merging parameters"):
        tensors = [s[key] for s in shards]
        # 处理可能存在的 DTensor
        tensors = [t.to_local() if hasattr(t, 'to_local') else t for t in tensors]
        
        # FSDP 默认在第 0 维切分
        if tensors[0].dim() > 0 and tensors[0].size(0) * world_size >= 1:
             try:
                full_state_dict[key] = torch.cat(tensors, dim=0)
             except Exception:
                # 如果 cat 失败，说明该参数可能在所有 rank 上都是全量的
                full_state_dict[key] = tensors[0]
        else:
            full_state_dict[key] = tensors[0]

    # 释放分片内存
    del shards

    # 4. 初始化模型并保存
    print("\nStep 3/3: Saving Model")
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16, trust_remote_code=True)
    
    # 使用 strict=False 允许少量的词表裁剪差异
    model.load_state_dict(full_state_dict, strict=False)
    
    print(f"Saving to {output_path}...")
    model.save_pretrained(output_path)
    
    # 拷贝分词器等核心文件
    import shutil
    for item in os.listdir(hf_path):
        s = os.path.join(hf_path, item)
        d = os.path.join(output_path, item)
        if os.path.isfile(s) and not os.path.exists(d):
            shutil.copy2(s, d)
            
    print("\nSuccess! Merge complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsdp_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    merge_fsdp_to_hf(args.fsdp_path, args.output_path)