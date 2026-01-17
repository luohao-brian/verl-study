# GRPO 训练优化指南: Countdown 任务

本文档详细说明了在 8x A100 环境下，使用 GRPO (Group Relative Policy Optimization) 训练 Qwen3-4B 进行 Countdown (数字游戏) 任务的超参数配置与设计理念。

## 1. 数据与 Batch 配置 (Data & Batch Configuration)

### 训练数据集 (`data.train_files`)
-   **设置:** `countdown/train.parquet` (4096 样本)
-   **原理:** 控制模型每轮次 (Epoch) 见到的题目多样性。
-   **设计:** 增加样本量至 4096，以确保模型学习的是通用的数字凑整逻辑，而不是死记硬背特定的数字组合。

### 训练 Batch Size (`data.train_batch_size`)
-   **设置:** `64`
-   **原理:** 每个训练步（Step）处理的原始问题数量。
-   **设计:** 
    *   设置为 `64`，启用**梯度累积 (Gradient Accumulation)**。
    *   计算: `Batch Size (64) / (Num GPUs (8) * Micro Batch (4)) = 2 accumulation steps`。
    *   **优势:** 相比于 GSM8K 的 32，更大的 Batch Size 能平滑梯度噪声，使训练更加稳定。

### PPO Mini-Batch Size (`actor.ppo_mini_batch_size`)
-   **设置:** `64`
-   **原理:** 单次参数更新的数据量。
-   **设计:** 与 `train_batch_size` 保持一致，确保每次更新都利用了当前采集的所有数据的统计信息。

### Micro Batch Size (`ppo_micro_batch_size_per_gpu`)
-   **设置:** `4`
-   **设计:** 针对 A100 (80GB) 的显存优化配置。

### Rollout 采样数 (`rollout.n`)
-   **设置:** `16`
-   **原理:** GRPO 算法的核心参数，用于估计 Baseline。
-   **设计:** 16 是数学推理任务的平衡点。更高的 $N$ 能减少方差，但线性增加计算成本。

---

## 2. 优化与训练动态 (Optimization & Training Dynamics)

### 学习率设计 (`actor.optim.lr`)
-   **设置:** `2e-6`
-   **原理:** 步长控制。
-   **设计:** 
    *   **激进策略**: Countdown 任务逻辑相对单一（纯数学运算），模型主要需要学习输出格式 (`<think>`) 和运算规则。
    *   **实验数据**: 早期实验表明 `1e-6` 导致收敛过慢，且容易陷入局部最优（Entropy 极低）；提升至 `2e-6` 后，模型能更快地跳出错误模式，且 `grad_norm` 依然保持稳定。

### KL 散度与系数 (`algorithm.kl_ctrl.kl_coef`)
-   **设置:** `0.01`
-   **原理:** 限制策略漂移。
-   **现象解释 (Why KL is 0?):**
    *   **观察**: 训练日志中 `actor/kl_loss` 常年为 0。
    *   **原因**: 这是 GRPO 实现的特性。KL 惩罚被计算并**从 Reward 中扣除**（作为 Advantage 计算的一部分），而非作为 Loss 的独立项。
    *   **作用**: 尽管日志为 0，KL 仍在幕后防止模型生成完全偏离预训练语言分布的“乱码”解。

### 温度参数 (`rollout.temperature`)
-   **设置:** `1.0`
-   **设计:** 必须保持高温度。Countdown 任务容易导致模型“背题”，高温度强迫模型在采样时尝试不同的数字组合和运算顺序，这对发现新的解法至关重要。

---

## 3. 基础设施与日志 (Infrastructure & Logging)

### 并行策略 (`strategy`)
-   **Actor:** `fsdp`
    *   全切片数据并行，适合 8 卡训练 4B 模型，显存利用率高。
-   **Rollout:** `vLLM` (TP=1)
    *   不使用模型切片（Tensor Parallel），每张卡独立跑一个完整模型副本进行推理，最大化吞吐量。

### 日志监控
-   **WandB**: 关注 `val-core/countdown/acc`（验证集准确率）和 `actor/entropy`（探索程度）。
-   **Completion Logs**: 检查生成的 `<think>` 标签，确认模型是否真的在进行步骤推演。

---

## 4. 序列长度 (Sequence Lengths)

### 最大 Prompt 长度
-   **设置:** `256`
-   **设计:** Countdown 的题目很短（"使用 [1, 2, 3, 4] 得到 24"），256 绰绰有余。

### 最大响应长度
-   **设置:** `2048`
-   **设计:** 这是一个宽裕的设置。虽然最终答案很短，但我们鼓励模型在 `<think>` 块中进行大量的试错和验算（Chain-of-Thought），截断会直接导致训练失败。
