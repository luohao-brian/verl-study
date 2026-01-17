# 环境安装与核心配置指南 (8x A100)

针对 **8x NVIDIA A100 (80GB)** 硬件环境，本项目通过 `uv` 实现了严格的版本锁定与高性能环境配置。由于 **FlashAttention 2**、**vLLM** 和 **PyTorch** 之间存在极其敏感的二进制依赖关系，安装流程分为“自动同步”与“手动修复”两部分。

---

## 1. `pyproject.toml` 核心配置解析

本项目的配置文件 `pyproject.toml` 包含以下关键设计，确保了 A100 环境下的训练性能：

### 1.1 核心版本锁定
为了避免 `undefined symbol` 错误，我们将核心包强制锁定在验证过的兼容版本：
*   **`torch==2.8.0`**: 作为底层框架，与 vLLM 0.11.0 深度绑定。
*   **`vllm==0.11.0`**: 针对 A100 优化的生成引擎。
*   **`flash-attn==2.8.1`**: 必须使用的注意力加速库。
*   **`verl` (rev v0.7.0)**: 从官方 Git 仓库的特定版本安装，确保 GRPO 算法逻辑的稳定性。

### 1.2 构建依赖声明 (`extra-build-dependencies`)
由于 `flash-attn` 在安装时需要检测本地的 `torch` 环境，我们在 `[tool.uv.extra-build-dependencies]` 中显式声明了构建时需求：
```toml
[tool.uv.extra-build-dependencies]
flash-attn = ["torch==2.8.0", "setuptools", "wheel"]
```
这确保了即使在隔离构建环境下，`flash-attn` 也能找到正确版本的 `torch` 标头。

---

## 2. 完整安装流程

### 第一步：自动化同步 (基础环境)
首先使用 `uv` 初始化虚拟环境并下载所有定义的依赖包：
```bash
uv sync
```
**注意**：此步骤后，自动下载的 `flash-attn` 预编译包（Wheel）可能与当前的 `torch 2.8.0` 存在 ABI 不兼容问题，会导致 `ImportError`。

### 第二步：核心修复 (手动编译 FlashAttention)
这是**最关键**的一步。我们必须在虚拟环境内部，使用 **禁用构建隔离** 的方式强制重新编译 `flash-attn`，确保它链接到当前环境的 `torch` 动态库。

执行以下命令：
```bash
# 强制针对环境内的 torch 2.8.0 重新原地编译 flash-attn
uv pip install --python .venv/bin/python \
    --no-build-isolation \
    --force-reinstall \
    --no-cache-dir \
    "flash-attn==2.8.1" "torch==2.8.0"
```

### 第三步：数据准备
在运行训练前，需要将 GSM8K 原始数据转换为 Parquet 格式：
```bash
uv run scripts/data_prep_gsm8k.py
```
转换后的数据将存储在 `artifacts/data/` 目录下。

### 第四步：日志监控准备 (WandB)
本项目默认开启 WandB 实时监控。如果尚未登录，请执行：
```bash
uv run wandb login
```

---

## 3. 环境验证流程

安装完成后，务必运行以下命令确保环境“无毒”：

1.  **验证 FlashAttention**:
    ```bash
    ./.venv/bin/python -c "import flash_attn; print('FlashAttention 加载成功！')"
    ```
    *如果此处报错 `undefined symbol`，请检查第二步是否执行成功。*

2.  **验证 GPU 可用性**:
    ```bash
    ./.venv/bin/python -c "import torch; print(f'可用 GPU 数量: {torch.cuda.device_count()}')"
    ```

---

## 4. 为什么不运行官方安装脚本？

原官方脚本（`install_vllm_sglang_mcore.sh`）在 A100 环境下存在以下问题，已被本项目优化：
1.  **剔除了 TE/Megatron**: A100 训练 4B 模型使用 Verl 原生 FSDP 效率更高，且避开了 TransformerEngine 极长的编译时间。
2.  **剔除了 OpenCV**: 减少了与图形库相关的系统层依赖报错。
3.  **修复了 ABI 冲突**: 官方脚本在某些 Python 3.12 环境下会导致 Torch 符号错误，本项目通过 `uv pip --no-build-isolation` 彻底解决。
