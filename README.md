# Verl Study: GRPO 强化学习实践 (Countdown & GSM8K)

本项目基于 [Verl](https://github.com/volcengine/verl) 框架，在 8x NVIDIA A100 环境下实践了 GRPO (Group Relative Policy Optimization) 算法。通过对 Countdown（数字游戏）和 GSM8K（数学应用题）的深度优化，我们建立了一套完整的 RLHF 训练与评估体系。

## 1. 核心安装流程

> 详细环境配置请参考 [docs/INSTALLATION.md](docs/INSTALLATION.md)。本项目强制锁定核心包版本（**torch==2.8.0**, **vllm==0.11.0**, **flash-attn==2.8.1**）以确保 A100 环境下的稳定性。

1.  **环境同步**:
    ```bash
    # 推荐使用 uv 管理依赖
    uv sync
    ```

2.  **手动修复编译 (关键步)**:
    由于 FlashAttention 与 PyTorch 存在敏感的二进制依赖，必须禁用构建隔离进行原地编译：
    ```bash
    uv pip install --no-build-isolation --force-reinstall "flash-attn==2.8.1" "torch==2.8.0"
    ```

3.  **数据准备**:
    ```bash
    # 生成/下载训练数据 (Parquet 格式)
    uv run scripts/data_prep_countdown.py
    uv run scripts/data_prep_gsm8k.py
    ```

## 2. 基础模型选型策略 (Model Selection)

为了最大化 GRPO 的训练收益，我们针对不同任务采用了差异化的基座模型选择策略。

*   **Countdown 任务 → Qwen3-4B**
    *   **理由**: Countdown 对格式依从性（`<think>` 标签的使用）和基础算术能力要求极高。Qwen3-4B 具备更强的指令遵循能力，能够快速适应复杂的奖励格式，避免因格式错误导致训练早夭。
    
*   **GSM8K 任务 → Qwen2.5-3B**
    *   **理由**: 
        *   **Qwen3-4B 太强了**: 我们的信号探测显示，Qwen3-4B 在 GSM8K 上的 Zero-shot 准确率已高达 82%，且熵极低 (0.1)。这导致 GRPO 缺乏足够的“错误样本”来构建对比优势，训练容易陷入模式坍缩。
        *   **Qwen2.5-3B 刚刚好**: 其准确率较低 (~6.5%) 且方差较高 (Variance > 0.05)。这意味着它处于“懂一点但不多”的状态，正是强化学习（RL）通过试错反馈提升效果的最佳区间。

## 3. 数据集说明

### 3.1 来源与合成原理
*   **Countdown (倒计时/算点数)**
    *   **来源**: Hugging Face `Jiayi-Pan/Countdown-Tasks-3to4`。
    *   **逻辑**: 这是一个合成数据集。给定一组数字（如 `[2, 8, 5]`）和一个目标值（如 `10`），模型需要生成一个数学表达式（如 `8 + 2 * 1` 错误, 应为 `2 * 5` 等），使得结果等于目标值。数据预处理会将其包装为 Chain-of-Thought (CoT) 格式，要求模型先在 `<think>` 标签中推理。
*   **GSM8K (小学数学应用题)**
    *   **来源**: OpenAI 官方 GSM8K 数据集 (`gsm8k/main`)。
    *   **逻辑**: 包含高质量的小学级多步数学推理题。我们提取标准答案中的数值部分（`####` 后面的数字）作为 Ground Truth，用于奖励函数的最终验证。

### 3.2 数据集与信号评估 (Signal Probing)

在正式训练前，我们必须评估模型在数据集上的“学习潜力”。如果模型对所有问题都能完美回答（方差=0）或完全不会（准确率=0），GRPO 将无法工作。

#### 评估命令
```bash
# 探测 GSM8K 数据的训练价值 (采样 16 次)
uv run scripts/eval_gsm8k.py --model_path /path/to/base_model analyse --rollout_n 16
```

#### 关键指标解读
| 指标 | 理想范围 | 解读 |
| :--- | :--- | :--- |
| **Accuracy (准确率)** | 10% - 80% | 如果 <5%，建议先做 SFT；如果 >95%，建议换更难的数据集。 |
| **Variance (方差)** | **> 0.05** | **核心指标**。表示模型在面对同一问题时产生了分歧（有对有错）。这是 RL 梯度的主要来源。 |
| **Entropy (熵)** | 0.2 - 0.8 | 表示模型输出的不确定性。过低说明模型可能在“背书”。 |

## 4. 训练调优与核心参数

> 详细参数原理请查阅优化指南: [Countdown](docs/OPTIMIZATION_COUNTDOWN.md) | [GSM8K](docs/OPTIMIZATION_GSM8K.md)

| 参数 | Countdown 设置 | GSM8K 设置 | 设计意图 |
| :--- | :--- | :--- | :--- |
| `learning_rate` | **2e-6** | **1e-6** | Countdown 逻辑简单，可用大 LR 加速；GSM8K 需防止语言能力退化。 |
| `rollout.n` | 16 | 16 | 保证足够的基线（Baseline）估计稳定性。 |
| `kl_coef` | 0.01 | 0.01 | 防止 Reward Hacking。**注意：日志中 KL Loss 为 0 是正常的（因为 GRPO 将 KL 计入 Reward）。** |
| `batch_size` | 64 | 32 | 配合梯度累积，平衡显存与更新稳定性。 |

## 5. 训练产物与 Checkpoint

训练过程中会自动保存 Checkpoint，路径结构如下：

```text
artifacts/checkpoints/{task_name}/global_step_{step}/actor/
├── huggingface/            # [重要] 可直接用于 vLLM/Transformers 加载的权重
│   ├── config.json
│   ├── model-00001-of-00004.safetensors
│   └── tokenizer.json
└── fsdp_config.json        # 训练过程中的分布式状态（用于 Resume）
```

**加载说明**: 在进行推理或评估时，请直接指定到 `actor/huggingface` 目录。

## 6. 训练监控与评估

### 6.1 训练指标解读 (WandB/Logs)
*   **`val-core/.../acc`**: 验证集准确率。最直接的成功指标。
*   **`actor/entropy`**: 策略熵。
    *   *健康趋势*: 保持稳定或**微涨**（如 0.2 -> 0.3），说明模型在积极探索。
    *   *异常*: 跌至 0.01 以下，说明模型丧失探索能力（Mode Collapse）。
*   **`actor/grad_norm`**: 梯度范数。应稳定在 0.05 - 0.2 之间。

### 6.2 模型效果对比评估 (Post-Training)

训练完成后，使用以下命令对比基座模型与训练后模型的效果。

#### Countdown 评估命令
```bash
# 评估训练后的模型 (替换 path 为实际 checkpoint 路径)
uv run scripts/eval_countdown.py \
    --model_path artifacts/checkpoints/countdown/global_step_100/actor/huggingface \
    --limit 100
```

#### GSM8K 评估命令
```bash
# 评估训练后的模型
uv run scripts/eval_gsm8k.py \
    --model_path artifacts/checkpoints/gsm8k/global_step_233/actor/huggingface \
    --limit 50 reward
```

#### 结果指标深度解读
以下为典型的 Countdown 任务评估报告示例：

| 指标 | 含义 | 示例数据 (Base -> Trained) | 解读 |
| :--- | :--- | :--- | :--- |
| **Mean Accuracy** | **平均得分**。即所有生成样本的平均分。由于我们的奖励函数包含部分分（格式分、数字约束分），此分数通常略高于完全正确率。 | `56.00% -> 78.40%` (**+22.4%**) | 模型的整体依从性变强，不仅算对的多了，格式错误的也少了。 |
| **Pass@1** | **一次通过率**。仅计算获得满分（1.0）的比例。这是最严苛的指标。 | `56.00% -> 78.00%` (**+22.0%**) | 模型解决核心数学问题的能力大幅提升。 |
| **Avg Entropy** | **生成熵**。反映模型在生成过程中的“思考多样性”。 | `0.0000 -> 0.0144` | **从死板到灵活**。0 意味着只会背答案；微涨意味着模型学会了尝试不同的计算路径（多路径思考）。 |

## 📚 文档索引
*   [📂 评估体系设计文档 (EVALUATION_DESIGN.md)](docs/EVALUATION_DESIGN.md)
*   [⚙️ 训练优化指南 (GSM8K)](docs/OPTIMIZATION_GSM8K.md)
*   [⚙️ 训练优化指南 (Countdown)](docs/OPTIMIZATION_COUNTDOWN.md)
*   [💰 奖励函数设计 (GSM8K)](docs/REWARD_GSM8K.md)