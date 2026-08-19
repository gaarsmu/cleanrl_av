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
    --env-ids MinAtar/Asterix-v1 MinAtar/Breakout-v1 MinAtar/Freeway-v1 MinAtar/Seaquest-v1 MinAtar/SpaceInvaders-v1 \
    --command "python cleanrl/rpq_separate_network_minatar.py --torch-deterministic --track --beta 1.0 --total-timesteps 10000000 --eval_frequency 1000 --exp_name 'soft_rpq_atari_separate' --value-lr-multiplier 0.5 --use_target_network --l2_coef 0.005 --eval-results-path '/scratch/work/masoudh1/cleanrl_av' --wandb-path '/scratch/work/masoudh1/cleanrl_av/wandb'" \
    --num-seeds 3 \
    --workers 0 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 8 \
    --slurm-template-path benchmark/triton_1gpu.slurm_template