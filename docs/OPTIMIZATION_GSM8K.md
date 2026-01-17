# GRPO 训练优化指南: GSM8K 任务

本文档详细说明了在 8x A100 环境下，使用 GRPO (Group Relative Policy Optimization) 训练 Qwen3-4B 进行 GSM8K 数学推理任务的超参数配置与设计理念。

## 1. 数据与 Batch 配置 (Data & Batch Configuration)

### 训练数据集 (`data.train_files`)
-   **设置:** `gsm8k/train.parquet`
-   **设计:** 使用标准的 GSM8K 训练集，包含高质量的小学数学应用题，旨在训练模型的多步推理能力。

### 训练 Batch Size (`data.train_batch_size`)
-   **设置:** `32`
-   **原理:** 每个训练步（Step）处理的**原始问题（Prompts）**数量。
-   **设计:** 
    *   设置为 `32`，配合 `rollout.n=16`，每步总共生成 $32 \times 16 = 512$ 个序列。
    *   较小的 Batch Size 有助于模型更频繁地更新权重，对奖励信号保持高度灵敏。

### PPO Mini-Batch Size (`actor.ppo_mini_batch_size`)
-   **设置:** `32`
-   **原理:** PPO 内循环中单次 SGD 更新的数据量。
-   **设计:** 
    *   $M=32$。由于 GRPO 需要基于一组采样（$G=16$）计算 Advantage，建议 $M$ 是 $G$ 的整数倍。
    *   当前配置下，每次梯度更新平均涵盖 2 个完整的原始问题（$32/16=2$），既保证了梯度的有效性，又避免了样本过于碎片化。

### Micro Batch Size (`ppo_micro_batch_size_per_gpu`)
-   **设置:** `4`
-   **原理:** 单张 GPU 在前向/反向传播时处理的数据块大小，受显存限制。
-   **设计:** 针对 A100 (80GB) 优化，最大化显存利用率以提升吞吐量。

### Rollout 采样数 (`rollout.n`)
-   **设置:** `16`
-   **原理:** 针对每个问题生成的回答数量，用于估计 GRPO 的 Baseline。
-   **设计:** 
    *   对于 GSM8K 这种逻辑推理任务，`n=16` 提供了足够的样本多样性来稳定优势函数（Advantage）的估计。
    *   如果显存允许，增加此数值通常能进一步降低梯度方差。

--- 

## 2. 优化与训练动态 (Optimization & Training Dynamics)

### 学习率设计 (`actor.optim.lr`)
-   **设置:** `1e-6` (配合 Cosine Decay)
-   **原理:** 控制权重更新的步长。
-   **设计:** 
    *   **保守策略**: 相比于 Countdown 任务 (`2e-6`)，GSM8K 的语义更复杂，模型容易发生“灾难性遗忘”或语言能力退化。因此选择了更低的 `1e-6`。
    *   **稳定性**: 实验表明，该学习率能有效维持 `grad_norm` 在健康范围 (0.05 - 0.15)，避免梯度爆炸。

### KL 散度与系数 (`algorithm.kl_ctrl.kl_coef`)
-   **设置:** `0.01`
-   **原理:** 防止强化学习后的策略 $\pi_\theta$ 偏离参考策略 $\pi_{ref}$ 太远。
-   **现象解释 (Why KL is 0?):**
    *   在训练日志中，你可能会观察到 `actor/kl_loss` 或 `actor/ppo_kl` 一直为 `0`。
    *   **原因**: 这是 GRPO 在 `verl` 中的实现特性。GRPO 通常将 KL 散度作为一个**惩罚项（Penalty）直接扣除在 Reward（奖励）中**（即 Token-level KL Penalty），而不是作为 Loss 函数中的显式项。
    *   因此，KL 约束实际上正在生效（通过降低偏离样本的 Reward），只是没有体现在 PPO 传统的 Loss 指标上。

### 温度参数 (`rollout.temperature`)
-   **设置:** `1.0` (默认)
-   **设计:** 保持较高的温度以鼓励**探索 (Exploration)**。在 GRPO 中，如果模型生成的 `n` 个回答一模一样，Advantage 将会归零，训练就会停滞。

--- 

## 3. 基础设施与日志 (Infrastructure & Logging)

### 并行策略 (`strategy`)
-   **Actor:** `fsdp` (Fully Sharded Data Parallel)
    *   将模型参数、梯度和优化器状态切分到 8 张 GPU 上，极大降低单卡显存压力。
-   **Rollout:** `vLLM` (Tensor Parallel = 1)
    *   利用 vLLM 的高性能推理引擎进行数据生成。由于 4B 模型较小，单卡即可承载，因此不使用张量并行 (TP)，而是让 8 张卡并行独立生成 (Data Parallel)。

### 日志监控
-   **WandB**: 实时监控 `acc` (准确率), `entropy` (熵), `grad_norm` (梯度范数)。
-   **本地日志**: 所有 Shell 输出重定向至 `logs/` 目录。
-   **Completion Logs**: 专门记录模型生成的 `<think>` 过程，用于人工审计推理质量。

--- 

## 4. 序列长度 (Sequence Lengths)

### 最大 Prompt 长度
-   **设置:** `512`
-   **设计:** 覆盖 GSM8K 的问题描述及 Few-shot 示例（如果使用）。

### 最大响应长度
-   **设置:** `1024`
-   **设计:** 预留足够的 Token 数供模型进行 Chain-of-Thought (CoT) 推理。GSM8K 的解答通常需要多步计算，截断会导致错误的奖励信号（未完成的推理通常被判错）。