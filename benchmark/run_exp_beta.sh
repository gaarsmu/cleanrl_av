#!/bin/bash
#SBATCH --job-name=cleanrl_minatar
#SBATCH --partition=gpu-v100-32g
#SBATCH --time=08:00:00                   
#SBATCH --ntasks=1                        
#SBATCH --cpus-per-task=4                 
#SBATCH --mem=16G                         
#SBATCH --gres=gpu:1                      
#SBATCH --array=1-45                      # 2 envs * 3 betas * 3 seeds = 18 jobs
#SBATCH --output=benchmark/slurm/logs/%x_%A_%a.out

# 1. Load modules and activate environment
module load mamba  
source activate cleanrl

# 2. Define the hyperparameter grid
ENVS=("MinAtar/Asterix-v1" "MinAtar/Breakout-v1" "MinAtar/Freeway-v1" "MinAtar/Seaquest-v1" "MinAtar/SpaceInvaders-v1")
BETAS=(1 10 100)
SEEDS=(4 5 6)

# 3. Calculate mapping indices
# Slurm IDs start at 1, but Bash arrays are 0-indexed (0 to 17)
INDEX=$(($SLURM_ARRAY_TASK_ID - 1))

# Environment index: changes every 9 tasks (3 betas * 3 seeds)
ENV_INDEX=$(($INDEX / 9))

# Remaining index for Beta and Seed within the selected environment
REMAINING=$(($INDEX % 9))

# Beta index: changes every 3 tasks
BETA_INDEX=$(($REMAINING / 3))

# Seed index: cycles 0, 1, 2
SEED_INDEX=$(($REMAINING % 3))

# 4. Assign the variables for this specific task
CURRENT_ENV=${ENVS[$ENV_INDEX]}
CURRENT_BETA=${BETAS[$BETA_INDEX]}
CURRENT_SEED=${SEEDS[$SEED_INDEX]}

echo "Task $SLURM_ARRAY_TASK_ID | Env: $CURRENT_ENV | Beta: $CURRENT_BETA | Seed: $CURRENT_SEED"

# Run RPQ with calculated beta value
srun python cleanrl/rpq_separate_network_minatar.py \
    --env-id "$CURRENT_ENV" \
    --seed $CURRENT_SEED \
    --torch-deterministic \
    --track \
    --beta $CURRENT_BETA \
    --total-timesteps 10000000 \
    --eval_frequency 200000 \
    --exp_name 'rpq beta effect exp01' \
    --value-lr-multiplier 1 \
    --use_target_network \
    --exploration-fraction 0.1 \
    --max_rarity 5 \
    --l2_coef 0.005 \
    --eval-results-path '/scratch/work/masoudh1/cleanrl_av' \
    --wandb-path '/scratch/work/masoudh1/cleanrl_av/wandb'