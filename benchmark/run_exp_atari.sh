#!/usr/bin/env bash
#SBATCH --job-name=cleanrl_atari
#SBATCH --partition=RTX4090
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=1-15
#SBATCH --output=benchmark/slurm/logs/%x_%A_%a.out

# Run from anywhere: bash benchmark/run_exp_atari.sh
# Preview task 1: bash benchmark/run_exp_atari.sh --dry-run
# Preview another task: SLURM_ARRAY_TASK_ID=15 bash benchmark/run_exp_atari.sh --dry-run
# Scheduler overrides: mkdir -p benchmark/slurm/logs
#   sbatch --partition=A100 benchmark/run_exp_atari.sh  (from the repository root)
# Local proxy exports belong in benchmark/.env.atari.local (ignored by Git).

set -euo pipefail

if [[ -n ${SLURM_JOB_ID:-} ]]; then
    PROJECT_ROOT=${CLEANRL_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}
else
    PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
fi
cd "$PROJECT_ROOT"

DRY_RUN=false
if [[ ${1:-} == --dry-run ]]; then
    DRY_RUN=true
    shift
fi

if [[ -z ${SLURM_JOB_ID:-} && $DRY_RUN == false ]]; then
    mkdir -p benchmark/slurm/logs
    exec sbatch --chdir="$PROJECT_ROOT" "$PROJECT_ROOT/benchmark/run_exp_atari.sh" "$@"
fi

ENVS=(AsterixNoFrameskip-v4 BreakoutNoFrameskip-v4 FreewayNoFrameskip-v4 SeaquestNoFrameskip-v4 SpaceInvadersNoFrameskip-v4)
SEEDS=(1 2 3)
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
if [[ ! $TASK_ID =~ ^[0-9]+$ ]] || (( TASK_ID < 1 || TASK_ID > ${#ENVS[@]} * ${#SEEDS[@]} )); then
    echo "Invalid array task ID: expected 1 through $((${#ENVS[@]} * ${#SEEDS[@]}))" >&2
    exit 1
fi
INDEX=$((TASK_ID - 1))
CURRENT_ENV=${ENVS[INDEX / ${#SEEDS[@]}]}
CURRENT_SEED=${SEEDS[INDEX % ${#SEEDS[@]}]}

PYTHON_BIN=${CLEANRL_PYTHON:-$HOME/miniconda3/envs/cleanrl/bin/python}
OUTPUT_ROOT=${CLEANRL_OUTPUT_ROOT:-$PROJECT_ROOT/runs/atari}
COMMAND=(
    "$PYTHON_BIN" -m cleanrl.rpq_separate_network_atari
    --env-id "$CURRENT_ENV"
    --seed "$CURRENT_SEED"
    --cuda
    --torch-deterministic
    --track
    --beta 1.0
    --total-timesteps 10000000
    --eval-frequency 200000
    --exp-name rpq_separate_network_atari
    --value-lr-multiplier 1
    --adv-lr-multiplier 1
    --use-target-network
    --exploration-fraction 0.1
    --l2-coef 0.005
    --save-model
    --remove_one
    --eval-results-path "$OUTPUT_ROOT"
    --wandb-path "$OUTPUT_ROOT/wandb"
    "$@"
)

echo "Task $TASK_ID | Env: $CURRENT_ENV | Seed: $CURRENT_SEED"
if [[ $DRY_RUN == true ]]; then
    printf '%q ' srun "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

if [[ ! -x $PYTHON_BIN ]]; then
    echo "Python not found: set CLEANRL_PYTHON to the cleanrl environment's Python executable" >&2
    exit 1
fi
mkdir -p "$OUTPUT_ROOT/wandb"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Disable tracing before reading private values; never echo the proxy settings.
set +x
PROXY_FILE=${CLEANRL_PROXY_FILE:-$PROJECT_ROOT/benchmark/.env.atari.local}
if [[ -f $PROXY_FILE ]]; then
    source "$PROXY_FILE"
fi

# Let Slurm select the allocated GPU through CUDA_VISIBLE_DEVICES.
exec srun "${COMMAND[@]}"
