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

## 2. 数据集与信号评估 (Signal Probing)

在正式训练前，我们必须评估模型在数据集上的“学习潜力”。如果模型对所有问题都能完美回答（方差=0）或完全不会（准确率=0），GRPO 将无法工作。

### 评估命令
```bash
# 探测 GSM8K 数据的训练价值 (采样 16 次)
uv run scripts/eval_gsm8k.py --model_path /path/to/base_model analyse --rollout_n 16
```

### 关键指标解读
| 指标 | 理想范围 | 解读 |
| :--- | :--- | :--- |
| **Accuracy (准确率)** | 10% - 80% | 如果 <5%，建议先做 SFT；如果 >95%，建议换更难的数据集。 |
| **Variance (方差)** | **> 0.05** | **核心指标**。表示模型在面对同一问题时产生了分歧（有对有错）。这是 RL 梯度的主要来源。 |
| **Entropy (熵)** | 0.2 - 0.8 | 表示模型输出的不确定性。过低说明模型可能在“背书”。 |

## 3. 训练调优与核心参数

> 详细参数原理请查阅优化指南: [Countdown](docs/OPTIMIZATION_COUNTDOWN.md) | [GSM8K](docs/OPTIMIZATION_GSM8K.md)

### 核心参数速查
| 参数 | Countdown 设置 | GSM8K 设置 | 设计意图 |
| :--- | :--- | :--- | :--- |
| `learning_rate` | **2e-6** | **1e-6** | Countdown 逻辑简单，可用大 LR 加速；GSM8K 需防止语言能力退化。 |
| `rollout.n` | 16 | 16 | 保证足够的基线（Baseline）估计稳定性。 |
| `kl_coef` | 0.01 | 0.01 | 防止 Reward Hacking。**注意：日志中 KL Loss 为 0 是正常的（因为 GRPO 将 KL 计入 Reward）。** |
| `batch_size` | 64 | 32 | 配合梯度累积，平衡显存与更新稳定性。 |

### 启动训练
```bash
# 启动 GSM8K 训练 (自动处理日志清理与 WandB 同步)
bash scripts/run_grpo_gsm8k.sh
```

## 4. 训练监控与指标

训练过程中重点关注 **WandB** 或 `logs/train_*.log` 中的以下指标：

1.  **`val-core/.../acc`**: 验证集准确率。最直接的成功指标。
2.  **`actor/entropy`**: 策略熵。
    *   *健康趋势*: 保持稳定或**微涨**（如 0.2 -> 0.3）。
    *   *异常*: 跌至 0.01 以下，说明模型丧失探索能力（Mode Collapse）。
3.  **`actor/grad_norm`**: 梯度范数。应稳定在 0.05 - 0.2 之间。

## 5. 模型效果对比评估

训练完成后，使用评估脚本对比基座模型与 Checkpoint 的效果。

### 评估命令
```bash
# 评估 Countdown 模型 (对比 Pass@1)
uv run scripts/eval_countdown.py \
    --model_path artifacts/checkpoints/countdown/global_step_100/actor/huggingface \
    --limit 100
```

### 预期效果 (示例)
```text
STRICT ALIGNED EVALUATION REPORT (n=1)
----------------------------------------
Metric          Base Model    Trained Model
Mean Accuracy   56.00%   ->   78.40%  (+22.4%)
Pass@1          56.00%   ->   78.00%  (+22.0%)
Avg Entropy     0.0000   ->   0.0144  (学会了多路径思考)
```

## 📚 文档索引
*   [📂 评估体系设计文档 (EVALUATION_DESIGN.md)](docs/EVALUATION_DESIGN.md) - **新增**
*   [⚙️ 训练优化指南 (GSM8K)](docs/OPTIMIZATION_GSM8K.md)
*   [⚙️ 训练优化指南 (Countdown)](docs/OPTIMIZATION_COUNTDOWN.md)
*   [💰 奖励函数设计 (GSM8K)](docs/REWARD_GSM8K.md)
