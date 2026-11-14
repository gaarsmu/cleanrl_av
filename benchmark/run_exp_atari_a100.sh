#!/usr/bin/env bash
#SBATCH --job-name=cleanrl
#SBATCH --partition=A100
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --array=1-4
#SBATCH --output=benchmark/slurm/logs/%x_%A_%a.out

# Run from anywhere: bash benchmark/run_exp_atari_a100.sh
# Preview task 1: bash benchmark/run_exp_atari_a100.sh --dry-run
# Preview another task: SLURM_ARRAY_TASK_ID=4 bash benchmark/run_exp_atari_a100.sh --dry-run
# Flatten five games x three seeds into four GPU jobs: 4 + 4 + 4 + 3 runs.
# Runs sharing a GPU may belong to different games; each has its own log.
# Proxy exports belong in benchmark/.env.atari.local (ignored by Git).

set -euo pipefail

if [[ -n ${SLURM_JOB_ID:-} ]]; then
    PROJECT_ROOT=${CLEANRL_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}
else
    PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
fi
cd "$PROJECT_ROOT"

WORKER=false
if [[ ${1:-} == --worker ]]; then
    WORKER=true
    shift
fi
DRY_RUN=false
if [[ ${1:-} == --dry-run ]]; then
    DRY_RUN=true
    shift
fi
if [[ -z ${SLURM_JOB_ID:-} && $DRY_RUN == false ]]; then
    mkdir -p benchmark/slurm/logs
    exec sbatch --chdir="$PROJECT_ROOT" "$PROJECT_ROOT/benchmark/run_exp_atari_a100.sh" "$@"
fi

ENVS=(AsterixNoFrameskip-v4 BreakoutNoFrameskip-v4 FreewayNoFrameskip-v4 SeaquestNoFrameskip-v4 SpaceInvadersNoFrameskip-v4)
SEEDS=(4 5 6)
EXPERIMENTS_PER_GPU=4
TOTAL_EXPERIMENTS=$(( ${#ENVS[@]} * ${#SEEDS[@]} ))
TOTAL_TASKS=$(( (TOTAL_EXPERIMENTS + EXPERIMENTS_PER_GPU - 1) / EXPERIMENTS_PER_GPU ))
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
if [[ ! $TASK_ID =~ ^[0-9]+$ ]] || (( TASK_ID < 1 || TASK_ID > TOTAL_TASKS )); then
    echo "Invalid array task ID: expected 1 through $TOTAL_TASKS" >&2
    exit 1
fi
FIRST_INDEX=$(( (TASK_ID - 1) * EXPERIMENTS_PER_GPU ))
TASK_ENVS=()
TASK_SEEDS=()
for (( index=FIRST_INDEX; index<TOTAL_EXPERIMENTS && index<FIRST_INDEX+EXPERIMENTS_PER_GPU; index++ )); do
    TASK_ENVS+=("${ENVS[index / ${#SEEDS[@]}]}")
    TASK_SEEDS+=("${SEEDS[index % ${#SEEDS[@]}]}")
done

PYTHON_BIN=${CLEANRL_PYTHON:-$HOME/miniconda3/envs/cleanrl/bin/python}
OUTPUT_ROOT=${CLEANRL_OUTPUT_ROOT:-$PROJECT_ROOT/runs/atari_a100_remove_one}
BASE_COMMAND=(
    "$PYTHON_BIN" -m cleanrl.rpq_separate_network_atari
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
    --remove-one
    --save-model
    --eval-results-path "$OUTPUT_ROOT"
    --wandb-path "$OUTPUT_ROOT/wandb"
    "$@"
)

echo "Task $TASK_ID | Concurrent experiments: ${#TASK_SEEDS[@]}"
if [[ $DRY_RUN == true ]]; then
    for index in "${!TASK_SEEDS[@]}"; do
        printf '%q ' "${BASE_COMMAND[@]}" --env-id "${TASK_ENVS[index]}" --seed "${TASK_SEEDS[index]}"
        printf '\n'
    done
    exit 0
fi

# Use one Slurm step so all children inherit the same GPU assignment.
if [[ $WORKER == false ]]; then
    exec srun --ntasks=1 bash "$PROJECT_ROOT/benchmark/run_exp_atari_a100.sh" --worker "$@"
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
# Save W&B metrics locally while its cloud upload route is failing.
export WANDB_MODE=${WANDB_MODE:-offline}

# Disable tracing before reading private values; never echo the proxy settings.
set +x
PROXY_FILE=${CLEANRL_PROXY_FILE:-$PROJECT_ROOT/benchmark/.env.atari.local}
if [[ -f $PROXY_FILE ]]; then
    source "$PROXY_FILE"
fi

# Do not override CUDA_VISIBLE_DEVICES: all workers use Slurm's assigned GPU.
LOG_DIR="$OUTPUT_ROOT/jobs/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}/$TASK_ID"
mkdir -p "$LOG_DIR"
PIDS=()
stop_workers() {
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait || true
}
trap 'stop_workers; exit 130' INT
trap 'stop_workers; exit 143' TERM
TASK_LOGS=()
for index in "${!TASK_SEEDS[@]}"; do
    env_id=${TASK_ENVS[index]}
    seed=${TASK_SEEDS[index]}
    log_file="$LOG_DIR/${env_id}__seed_${seed}.out"
    TASK_LOGS+=("$log_file")
    echo "Starting $env_id seed $seed | log: $log_file"
    "${BASE_COMMAND[@]}" --env-id "$env_id" --seed "$seed" > "$log_file" 2>&1 &
    PIDS+=("$!")
done
STATUS=0
for index in "${!PIDS[@]}"; do
    if wait "${PIDS[index]}"; then
        echo "${TASK_ENVS[index]} seed ${TASK_SEEDS[index]} finished successfully"
    else
        WORKER_STATUS=$?
        echo "${TASK_ENVS[index]} seed ${TASK_SEEDS[index]} failed (exit $WORKER_STATUS); see ${TASK_LOGS[index]}" >&2
        STATUS=$WORKER_STATUS
    fi
done
exit "$STATUS"
