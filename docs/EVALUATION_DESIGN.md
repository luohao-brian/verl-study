# GRPO 评估体系设计文档

本文档详细阐述了本项目中针对数学推理任务（GSM8K/Countdown）的评估方法论，涵盖数据质量探测、训练过程监控及最终模型效果验证。

## 1. 核心评估理念：信号探测 (Signal Probing)

在强化学习（RL）中，仅仅知道模型“答对了多少”（Accuracy）是不够的。我们需要知道模型**对答案的确定性**以及**不同采样路径之间的冲突程度**，这直接决定了 GRPO 算法能否有效地计算优势（Advantage）并更新梯度。

本评估体系围绕以下三个核心维度构建：

1.  **准确性 (Accuracy / Mean Reward)**: 模型解决问题的基础能力。
2.  **多样性 (Variance / Conflict)**: 模型在多次尝试中是否产生了不同的结果（有对有错）。GRPO 极其依赖这种“对比信号”。
3.  **不确定性 (Entropy)**: 模型在生成过程中的“纠结”程度。

---

## 2. 评估指标详解

### 2.1 基础指标

*   **Pass@1 (Greedy Accuracy)**
    *   **定义**: 使用贪婪解码（Temperature=0）生成一次，答案正确的比例。
    *   **用途**: 衡量模型在生产环境下的最终性能。

*   **Pass@N (Best-of-N)**
    *   **定义**: 使用采样解码（Temperature>0）生成 N 次，其中至少有一次回答正确的比例。
    *   **用途**: 衡量模型的潜在能力上限（上限高说明模型知道解法，只是不稳定）。

### 2.2 进阶信号指标 (GRPO 特有)

这些指标通过对同一问题进行 $G$ 次采样 ($y_1, ..., y_G$) 计算得出。

*   **组内方差 (Group Reward Variance)**
    *   **公式**: $\text{Var}(R) = \frac{1}{G} \sum (r_i - \bar{r})^2$
    *   **解读**:
        *   `Var ≈ 0`: 信号**无效**。模型要么全对（$r_i=1$），要么全错（$r_i=0$）。此时 GRPO 的 Baseline 等于 Reward，Advantage 为 0，梯度消失。
        *   `Var > 0`: 信号**有效**。模型在同一问题上产生了分歧。这种“半懂不懂”的状态是 RL 学习效率最高的区域。
    *   **阈值**: 我们通常筛选 `Var > 0.05` 的数据作为高价值训练样本。

*   **组内熵 (Group Entropy)**
    *   **公式**: 基于采样结果分布计算的信息熵。
    *   **解读**:
        *   `Entropy 低`: 模型非常自信（或者死记硬背）。
        *   `Entropy 高`: 模型在进行尝试和探索。
    *   **用途**: 监控训练初期的探索能力。如果熵过低，需要提升 Temperature 或 KL 惩罚。

---

## 3. 评估工具使用指南

### 3.1 数据集质量评估 (Pre-Training)

在训练前，我们需要知道当前模型在这个数据集上是否有“学习空间”。

**命令:** 
```bash
# 对 GSM8K 测试集采样 16 次进行分析
uv run scripts/eval_gsm8k.py --model_path /path/to/base_model analyse \
    --rollout_n 16 --temperature 1.0
```

**输出解读:**
*   如果 `Avg Variance` 极低 (< 0.01)：说明基础模型对该数据集要么太强（全对），要么太弱（全不会）。建议更换更难/更简单的数据集。
*   如果 `Avg Accuracy` < 5%：模型基本不具备解决该任务的能力，RL 很难“无中生有”，建议先进行 SFT（监督微调）。

### 3.2 最终效果对比 (Post-Training)

对比基座模型与 RL 训练后模型的效果。

**命令:** 
```bash
# 评估训练后的模型
uv run scripts/eval_countdown.py \
    --model_path artifacts/checkpoints/countdown/global_step_100/actor/huggingface \
    --limit 100
```

**对比维度:**
| 指标 | 基座模型 | 训练后模型 | 理想变化 |
| :--- | :---: | :---: | :--- |
| **Pass@1** | 50% | 80% | **大幅提升** (目标) |
| **Mean Score** | 0.5 | 0.82 | **提升** (减少部分错误) |
| **Avg Entropy** | 0.0 | 0.2 | **微涨** (模型学会了思考的多样性) |

---

## 4. 奖励函数设计验证

评估代码还会自动验证 Reward Function 的有效性。

*   **日志审计**:
    *   `logs/completion_*_success.log`: 检查获得高分的样本，确认其 `<think>` 过程是否合理，防止模型利用规则漏洞（Reward Hacking）。
    *   `logs/completion_*_fail.log`: 检查低分样本，确认是否因为格式错误（Format Error）导致，如果是，说明模型需要加强指令遵循能力的训练。
```