#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Ensure local binary paths are in PATH
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

# Suppress uv hardlink warning on /scratch
export UV_LINK_MODE=copy

# Create Slurm log folder
mkdir -p benchmark/slurm/logs
mkdir -p slurm

# Set GPU execution determinism and thread bounds
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1

# Install / update project dependencies from root
uv pip install -e .

# Run benchmark generator to create Slurm array script
python -m cleanrl_utils.benchmark \
    --env-ids CartPole-v1 Acrobot-v1 MountainCar-v0 LunarLander-v2 \
    --command "python cleanrl/rrdq.py --torch-deterministic --track --beta 1.0 --total-timesteps 200000 --eval_frequency 1000 --exp_name 'hardrrdq' --adv-lr-multiplier 4.0 --two_time_scale --use_target_network" \
    --num-seeds 3 \
    --workers 0 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 4 \
    --slurm-template-path benchmark/triton_1gpu.slurm_template