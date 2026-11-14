#!/usr/bin/env bash
#SBATCH --job-name=rpq_atari_4090
#SBATCH --partition=RTX4090
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --array=1-13
#SBATCH --output=benchmark/slurm/logs/%x_%A_%a.out

# Run from anywhere: bash benchmark/run_exp_atari_4090_more.sh
# Preview task 1: bash benchmark/run_exp_atari_4090_more.sh --dry-run
# Preview another task: SLURM_ARRAY_TASK_ID=13 bash benchmark/run_exp_atari_4090_more.sh --dry-run
# Each game uses one job: seeds 1, 2, and 3 share one GPU.
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
    exec sbatch --chdir="$PROJECT_ROOT" "$PROJECT_ROOT/benchmark/run_exp_atari_4090_more.sh" "$@"
fi

# Additional Atari-57 games; excludes the five games already completed.
ENVS=(
    AlienNoFrameskip-v4 AmidarNoFrameskip-v4 AssaultNoFrameskip-v4
    AsteroidsNoFrameskip-v4 AtlantisNoFrameskip-v4 BankHeistNoFrameskip-v4
    BattleZoneNoFrameskip-v4 BeamRiderNoFrameskip-v4 BoxingNoFrameskip-v4
    CentipedeNoFrameskip-v4 ChopperCommandNoFrameskip-v4
    CrazyClimberNoFrameskip-v4 DemonAttackNoFrameskip-v4
)
SEEDS=(1 2 3)
SEEDS_PER_GPU=3
SEED_GROUPS=$(( (${#SEEDS[@]} + SEEDS_PER_GPU - 1) / SEEDS_PER_GPU ))
TOTAL_TASKS=$(( ${#ENVS[@]} * SEED_GROUPS ))
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
if [[ ! $TASK_ID =~ ^[0-9]+$ ]] || (( TASK_ID < 1 || TASK_ID > TOTAL_TASKS )); then
    echo "Invalid array task ID: expected 1 through $TOTAL_TASKS" >&2
    exit 1
fi
INDEX=$((TASK_ID - 1))
CURRENT_ENV=${ENVS[INDEX / SEED_GROUPS]}
FIRST_SEED_INDEX=$(( (INDEX % SEED_GROUPS) * SEEDS_PER_GPU ))
TASK_SEEDS=("${SEEDS[@]:FIRST_SEED_INDEX:SEEDS_PER_GPU}")

PYTHON_BIN=${CLEANRL_PYTHON:-$HOME/miniconda3/envs/cleanrl/bin/python}
OUTPUT_ROOT=${CLEANRL_OUTPUT_ROOT:-$PROJECT_ROOT/runs/atari_4090_more}
BASE_COMMAND=(
    "$PYTHON_BIN" -m cleanrl.rpq_separate_network_atari
    --env-id "$CURRENT_ENV"
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

echo "Task $TASK_ID | Env: $CURRENT_ENV | Seeds: ${TASK_SEEDS[*]}"
if [[ $DRY_RUN == true ]]; then
    for seed in "${TASK_SEEDS[@]}"; do
        printf '%q ' "${BASE_COMMAND[@]}" --seed "$seed"
        printf '\n'
    done
    exit 0
fi

# Use one Slurm step so all children inherit the same GPU assignment.
if [[ $WORKER == false ]]; then
    exec srun --ntasks=1 bash "$PROJECT_ROOT/benchmark/run_exp_atari_4090_more.sh" --worker "$@"
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
# Keep training independent of the currently failing W&B upload route.
# Offline runs retain metrics/configuration for `wandb sync` after completion.
# Override with WANDB_MODE=online when cloud access is reliable again.
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
for seed in "${TASK_SEEDS[@]}"; do
    echo "Starting seed $seed | log: $LOG_DIR/seed_$seed.out"
    "${BASE_COMMAND[@]}" --seed "$seed" > "$LOG_DIR/seed_$seed.out" 2>&1 &
    PIDS+=("$!")
done
STATUS=0
for index in "${!PIDS[@]}"; do
    if wait "${PIDS[index]}"; then
        echo "Seed ${TASK_SEEDS[index]} finished successfully"
    else
        WORKER_STATUS=$?
        echo "Seed ${TASK_SEEDS[index]} failed (exit $WORKER_STATUS); see $LOG_DIR/seed_${TASK_SEEDS[index]}.out" >&2
        STATUS=$WORKER_STATUS
    fi
done
exit "$STATUS"
