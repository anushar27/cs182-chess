#!/usr/bin/env bash
#SBATCH --job-name=chess-v2-stage0
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
python -m chess_icl_v2.cli audit --output results/cluster_simulator_audit.json
python -m chess_icl_v2.cli generate --config configs/stage0_all_rules.json --output results/cluster_stage0.json
