#!/bin/bash
#SBATCH --job-name=chess-pretrain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=runs/logs/pretrain_seed%a_%j.out
#SBATCH --array=0-2

set -e

SEEDS=(42 123 456)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

python scripts/pretrain.py \
    --seed "$SEED" \
    --config small \
    --max-steps 20000 \
    --max-epochs 100 \
    --output runs/pretrain/
