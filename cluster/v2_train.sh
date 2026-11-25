#!/usr/bin/env bash
#SBATCH --job-name=chess-v2-train
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME="${HF_HOME:-$PWD/work/hf-cache}"
: "${V2_DATA:?Set V2_DATA to the audited complete-label dataset}"
: "${V2_CONFIG:?Set V2_CONFIG to the training JSON}"
: "${V2_OUTPUT:?Set a new V2_OUTPUT directory}"
args=(--data "$V2_DATA" --config "$V2_CONFIG" --output "$V2_OUTPUT")
if [[ -n "${V2_WARMUP:-}" ]]; then args+=(--initial-checkpoint "$V2_WARMUP"); fi
if [[ -n "${V2_GATES:-}" ]]; then args+=(--gates "$V2_GATES"); fi
python -m chess_icl_v2.cli train "${args[@]}"
