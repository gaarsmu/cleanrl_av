#!/bin/bash

# Enforce strict PyTorch CUDA determinism
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1

uv pip install .

xvfb-run -a uv run python -m cleanrl_utils.benchmark \
    --env-ids CartPole-v1 Acrobot-v1 MountainCar-v0 LunarLander-v2 \
    --command "uv run python cleanrl/rrdq.py --torch-deterministic --track --beta 1.0 --total-timesteps 200000 --eval_frequency 1000 --exp_name 'hardrrdq'" \
    --num-seeds 3 \
    --workers 9 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 4 \
    --slurm-template-path benchmark/triton_1gpu.slurm_template