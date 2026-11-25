#!/bin/bash
#SBATCH --job-name=chess-ctrain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=runs/logs/ctrain_%a_%j.out
#SBATCH --array=0-8

set -e

# 3 seeds x 3 configs = 9 jobs
SEEDS=(42 123 456)
CONFIGS=("move_variants" "all_variants" "chess_only")
SEED_IDX=$((SLURM_ARRAY_TASK_ID / 3))
CONFIG_IDX=$((SLURM_ARRAY_TASK_ID % 3))
SEED=${SEEDS[$SEED_IDX]}
CONFIG=${CONFIGS[$CONFIG_IDX]}

python scripts/continue_train.py \
    --config "$CONFIG" \
    --pretrain-checkpoint "runs/pretrain/chess_base_seed${SEED}.pt" \
    --seed "$SEED" \
    --max-steps 10000 \
    --output runs/variant_train/
