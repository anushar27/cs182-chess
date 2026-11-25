#!/usr/bin/env bash
#SBATCH --job-name=chess-v2-profile
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
# Inherit scheduler visibility. Never assume GPU 1 or overwrite CUDA_VISIBLE_DEVICES.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME="${HF_HOME:-$PWD/work/hf-cache}"
python scripts/v2_profile_qwen.py --output results/h100_profile.json
