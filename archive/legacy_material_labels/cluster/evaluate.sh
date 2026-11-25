#!/bin/bash
#SBATCH --job-name=chess-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=runs/logs/eval_%j.out

set -e

# Evaluate all 9 checkpoints (3 configs x 3 seeds) on test variants
for CONFIG in move_variants all_variants chess_only; do
    for SEED in 42 123 456; do
        CKPT="runs/variant_train/${CONFIG}_seed${SEED}.pt"
        if [ -f "$CKPT" ]; then
            echo "Evaluating $CKPT..."
            python scripts/run_eval.py \
                --checkpoint "$CKPT" \
                --test-variants short_knight mobile_king piece_capture \
                    forward_pawn_capture dark_square_immunity blocked_squares \
                    long_board knight_army antichess \
                --k-values 0 1 2 4 8 16 \
                --seeds 42 123 456 \
                --output "runs/results/${CONFIG}_seed${SEED}.json"
        fi
    done
done

# Run post-hoc analysis and plotting
python scripts/analyze_results.py --results-dir runs/results --output-dir runs/analysis
