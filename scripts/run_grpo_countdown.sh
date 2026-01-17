#!/bin/bash
set -e

MODEL_PATH="/mnt/data/oniond/models/Qwen3-4B"

# Stability & WandB Configuration
export WANDB_MODE=online
export WANDB_START_METHOD=thread
export WANDB_PROJECT="countdown_grpo"
export WANDB_NAME="qwen3_4b_countdown_v1"
export COMPLETION_LOG_PATH="logs/completion_countdown_train.log"
export RAY_IGNORE_UNHANDLED_ERRORS=1

# 清理旧的采样日志
rm -f $COMPLETION_LOG_PATH

# Clean up previous runs
pkill -f "ray" || true

echo "Starting Countdown GRPO Full Run..."

uv run python3 -m verl.trainer.main_ppo \
    data.train_files=artifacts/data/countdown/train.parquet \
    data.val_files=artifacts/data/countdown/test.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=256 \
    data.train_max_samples=4096 \
    data.max_prompt_length=256 \
    data.max_response_length=2048 \
    data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=True \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.checkpoint.save_contents='["model","hf_model"]' \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.strategy=fsdp \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.01 \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$WANDB_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs=1 \
    trainer.log_val_generations=10 \
    trainer.default_local_dir=artifacts/checkpoints/countdown \
    custom_reward_function.path=scripts/reward_fn_countdown.py \
    custom_reward_function.name=compute_score \
    trainer.test_freq=20 \
    trainer.save_freq=100 \
    trainer.max_actor_ckpt_to_keep=3 \
    ray_kwargs.ray_init.num_cpus=100 \
    hydra.run.dir=logs/hydra/countdown/${now:%Y-%m-%d}/${now:%H-%M-%S} \
    2>&1 | tee logs/train_countdown.log

# Final cleanup
echo "Probing run finished."
sleep 10
pkill -f "ray" || true