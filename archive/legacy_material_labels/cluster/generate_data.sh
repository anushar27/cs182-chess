#!/bin/bash
#SBATCH --job-name=chess-datagen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=runs/logs/datagen_%j.out

set -e

# Standard chess
python scripts/generate_data.py --variant standard --count 10000 --seed 42

# All 13 variants (heuristic + random demos)
for variant in diagonal_rook straight_bishop weakened_queen extended_pawn \
               short_knight mobile_king piece_capture forward_pawn_capture \
               dark_square_immunity blocked_squares long_board knight_army antichess; do
    echo "Generating variant: $variant"
    python scripts/generate_data.py --variant "$variant" --count 2000 --policy heuristic --seed 42
    python scripts/generate_data.py --variant "$variant" --count 500 --policy random --seed 42
done

echo "Data generation complete."
