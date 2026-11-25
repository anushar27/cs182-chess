#!/bin/bash
# Model Scaling Study: Tiny (1.3M) vs Small (7.4M) vs Medium (23M)
# Tests the effect of transformer scale on in-context rule learning

set -e
mkdir -p runs/cache/tmp runs/cache/matplotlib runs/logs runs/pretrain runs/variant_train runs/results runs/analysis
export TMPDIR="$(pwd)/runs/cache/tmp"
export MPLCONFIGDIR="$(pwd)/runs/cache/matplotlib"

echo "=========================================================="
echo " Model Scaling Study: Tiny (1.3M) vs Small (7.4M) vs Medium (23M)"
echo "=========================================================="

# 1. Pretrain Tiny and Medium models on GPUs 1 and 2
echo "Step 1: Pretraining Tiny (GPU 1) and Medium (GPU 2) on standard chess..."
python3 scripts/pretrain.py \
    --seed 42 \
    --config tiny \
    --max-steps 20000 \
    --max-epochs 100 \
    --max-seq-len 4096 \
    --gpu 1 \
    --output runs/pretrain/ > "runs/logs/pretrain_tiny_seed42.log" 2>&1 &
PID_TINY=$!

python3 scripts/pretrain.py \
    --seed 42 \
    --config medium \
    --max-steps 20000 \
    --max-epochs 100 \
    --max-seq-len 4096 \
    --batch-size 64 \
    --gpu 2 \
    --output runs/pretrain/ > "runs/logs/pretrain_medium_seed42.log" 2>&1 &
PID_MED=$!

wait $PID_TINY $PID_MED
echo "Pretraining for Tiny and Medium complete."

# 2. Continue-train Tiny and Medium on all_variants and move_variants
echo "Step 2: Continue-training Tiny and Medium across variants..."
# Tiny all_variants on GPU 1
python3 scripts/continue_train.py \
    --config all_variants \
    --pretrain-checkpoint "runs/pretrain/chess_base_tiny_seed42.pt" \
    --seed 42 \
    --max-steps 10000 \
    --max-seq-len 4096 \
    --gpu 1 \
    --output runs/variant_train/scaling_tiny/ > "runs/logs/ctrain_tiny_all_seed42.log" 2>&1 &

# Medium all_variants on GPU 2
python3 scripts/continue_train.py \
    --config all_variants \
    --pretrain-checkpoint "runs/pretrain/chess_base_medium_seed42.pt" \
    --seed 42 \
    --max-steps 10000 \
    --max-seq-len 4096 \
    --batch-size 16 \
    --gpu 2 \
    --output runs/variant_train/scaling_medium/ > "runs/logs/ctrain_med_all_seed42.log" 2>&1 &

# Tiny move_variants on GPU 3
python3 scripts/continue_train.py \
    --config move_variants \
    --pretrain-checkpoint "runs/pretrain/chess_base_tiny_seed42.pt" \
    --seed 42 \
    --max-steps 10000 \
    --max-seq-len 4096 \
    --gpu 3 \
    --output runs/variant_train/scaling_tiny/ > "runs/logs/ctrain_tiny_move_seed42.log" 2>&1 &

# Medium move_variants on GPU 4
python3 scripts/continue_train.py \
    --config move_variants \
    --pretrain-checkpoint "runs/pretrain/chess_base_medium_seed42.pt" \
    --seed 42 \
    --max-steps 10000 \
    --max-seq-len 4096 \
    --batch-size 16 \
    --gpu 4 \
    --output runs/variant_train/scaling_medium/ > "runs/logs/ctrain_med_move_seed42.log" 2>&1 &

wait
echo "Continue-training for scaling study complete."

# 3. Evaluate Tiny and Medium models
echo "Step 3: Running evaluation matrix for Tiny and Medium..."
TEST_VARIANTS=(short_knight mobile_king piece_capture forward_pawn_capture dark_square_immunity blocked_squares long_board knight_army antichess)

python3 scripts/run_eval.py \
    --checkpoint "runs/variant_train/scaling_tiny/all_variants_seed42.pt" \
    --test-variants "${TEST_VARIANTS[@]}" \
    --k-values 0 1 2 4 8 16 \
    --seeds 42 123 \
    --max-seq-len 4096 \
    --gpu 1 \
    --output "runs/results/scaling_tiny_all_variants_seed42.json" > "runs/logs/eval_tiny_all.log" 2>&1 &

python3 scripts/run_eval.py \
    --checkpoint "runs/variant_train/scaling_medium/all_variants_seed42.pt" \
    --test-variants "${TEST_VARIANTS[@]}" \
    --k-values 0 1 2 4 8 16 \
    --seeds 42 123 \
    --max-seq-len 4096 \
    --gpu 2 \
    --output "runs/results/scaling_medium_all_variants_seed42.json" > "runs/logs/eval_med_all.log" 2>&1 &

python3 scripts/run_eval.py \
    --checkpoint "runs/variant_train/scaling_tiny/move_variants_seed42.pt" \
    --test-variants "${TEST_VARIANTS[@]}" \
    --k-values 0 1 2 4 8 16 \
    --seeds 42 123 \
    --max-seq-len 4096 \
    --gpu 3 \
    --output "runs/results/scaling_tiny_move_variants_seed42.json" > "runs/logs/eval_tiny_move.log" 2>&1 &

python3 scripts/run_eval.py \
    --checkpoint "runs/variant_train/scaling_medium/move_variants_seed42.pt" \
    --test-variants "${TEST_VARIANTS[@]}" \
    --k-values 0 1 2 4 8 16 \
    --seeds 42 123 \
    --max-seq-len 4096 \
    --gpu 4 \
    --output "runs/results/scaling_medium_move_variants_seed42.json" > "runs/logs/eval_med_move.log" 2>&1 &

wait
echo "Evaluation complete."

# 4. Generate updated analysis
echo "Step 4: Generating scaling comparison plots..."
python3 scripts/analyze_results.py --results-dir runs/results --output-dir runs/analysis
echo "Scaling study finished successfully!"
