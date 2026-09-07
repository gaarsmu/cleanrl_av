#!/bin/bash
#SBATCH --job-name=cleanrl_minatar
#SBATCH --partition=gpu-v100-32g
#SBATCH --time=05:00:00                   
#SBATCH --ntasks=1                        
#SBATCH --cpus-per-task=4                 
#SBATCH --mem=16G                         
#SBATCH --gres=gpu:1                      
#SBATCH --array=1-15                      # Spawns 15 parallel jobs
#SBATCH --output=benchmark/slurm/logs/%x_%A_%a.out


# 1. Load modules and activate environment
module load mamba  
source activate cleanrl

# 2. Define the hyperparameter grid
ENVS=("MinAtar/Asterix-v1" "MinAtar/Breakout-v1" "MinAtar/Freeway-v1" "MinAtar/Seaquest-v1" "MinAtar/SpaceInvaders-v1")
SEEDS=(1 2 3)

# 3. Calculate mapping indices
# Slurm IDs start at 1, but Bash arrays are 0-indexed
INDEX=$(($SLURM_ARRAY_TASK_ID - 1))

# Integer division finds the environment index (changes every 3 tasks)
ENV_INDEX=$(($INDEX / 3))

# Modulo operator finds the seed index (repeats 0, 1, 2)
SEED_INDEX=$(($INDEX % 3))

# 4. Assign the variables for this specific task
CURRENT_ENV=${ENVS[$ENV_INDEX]}
CURRENT_SEED=${SEEDS[$SEED_INDEX]}

echo "Starting Task $SLURM_ARRAY_TASK_ID | Env: $CURRENT_ENV | Seed: $CURRENT_SEED"

# # 5. Run RPQ and RDQ
# srun python cleanrl/rpq_separate_network_minatar.py \
#     --env-id "$CURRENT_ENV" \
#     --seed $CURRENT_SEED \
#     --torch-deterministic \
#     --track \
#     --beta 100.0 \
#     --beta_final 1.0 \
#     --beta_fraction 0.8 \
#     --beta_scheduling \
#     --total-timesteps 10000000 \
#     --eval_frequency 200000 \
#     --exp_name 'soft rpq beta scheduling' \
#     --value-lr-multiplier 1 \
#     --adv-lr-multiplier 1 \
#     --use_target_network \
#     --exploration-fraction 0.1 \
#     --l2_coef 0.005 \
#     --eval-results-path '/scratch/work/masoudh1/cleanrl_av' \
#     --wandb-path '/scratch/work/masoudh1/cleanrl_av/wandb'

# # 5. Run RDQ
# srun python cleanrl/rdq_separate_network_minatar.py \
#     --env-id "$CURRENT_ENV" \
#     --seed $CURRENT_SEED \
#     --torch-deterministic \
#     --track \
#     --beta 0.0 \
#     --total-timesteps 10000000 \
#     --eval_frequency 200000 \
#     --exp_name 'soft rdq - exploration effect' \
#     --value-lr-multiplier 1 \
#     --use_target_network \
#     --exploration-fraction 0.9 \
#     --l2_coef 0.005 \
#     --eval-results-path '/scratch/work/masoudh1/cleanrl_av' \
#     --wandb-path '/scratch/work/masoudh1/cleanrl_av/wandb'

# Run DQN
# srun python cleanrl/dqn_minatar.py \
#     --env-id "$CURRENT_ENV" \
#     --seed $CURRENT_SEED \
#     --torch-deterministic \
#     --track \
#     --total-timesteps 10000000 \
#     --eval_frequency 200000 \
#     --exp_name 'dqn' \
#     --use_target_network \
#     --eval-results-path '/scratch/work/masoudh1/cleanrl_av' \
#     --wandb-path '/scratch/work/masoudh1/cleanrl_av/wandb'


# # 5. Run ME Layer Idea!
srun python cleanrl/Mean_Expansion.py \
    --env-id "$CURRENT_ENV" \
    --seed $CURRENT_SEED \
    --torch-deterministic \
    --track \
    --total-timesteps 10000000 \
    --eval_frequency 200000 \
    --exp_name 'ME Layer Machado' \
    --use_target_network \
    --eval-results-path '/scratch/work/masoudh1/cleanrl_av' \
    --wandb-path '/scratch/work/masoudh1/cleanrl_av/wandb'
