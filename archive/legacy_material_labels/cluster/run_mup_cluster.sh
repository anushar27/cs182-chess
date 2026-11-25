#!/bin/bash
# Multi-GPU Cluster Pipeline for muP Scaling Study
# Uses 7x NVIDIA H100 GPUs concurrently (GPUs 1 to 7)

set -e
mkdir -p runs/cache/tmp runs/cache/matplotlib runs/logs runs/pretrain runs/variant_train/mup runs/results runs/analysis
export TMPDIR="$(pwd)/runs/cache/tmp"
export MPLCONFIGDIR="$(pwd)/runs/cache/matplotlib"

echo "=========================================================="
echo " muP Hyperparameter Scaling & Multi-Scale Variant Study"
echo "=========================================================="
date

# -------------------------------------------------------------
# STEP 1: Pretraining across 3 scales with muP (Zero-Shot LR = 3e-4)
# -------------------------------------------------------------
echo "[Phase 1] Pretraining Tiny, Small, and Medium with muP on GPUs 1, 2, 3..."

python3 scripts/pretrain.py \
    --seed 42 \
    --config tiny \
    --use-mup \
    --max-steps 6000 \
    --eval-interval 250 \
    --batch-size 128 \
    --lr 3e-4 \
    --gpu 1 \
    --output runs/pretrain/ > "runs/logs/pretrain_tiny_mup_seed42.log" 2>&1 &
PID_PRE_TINY=$!

python3 scripts/pretrain.py \
    --seed 42 \
    --config small \
    --use-mup \
    --max-steps 6000 \
    --eval-interval 250 \
    --batch-size 128 \
    --lr 3e-4 \
    --gpu 2 \
    --output runs/pretrain/ > "runs/logs/pretrain_small_mup_seed42.log" 2>&1 &
PID_PRE_SMALL=$!

python3 scripts/pretrain.py \
    --seed 42 \
    --config medium \
    --use-mup \
    --max-steps 6000 \
    --eval-interval 250 \
    --batch-size 64 \
    --lr 3e-4 \
    --gpu 3 \
    --output runs/pretrain/ > "runs/logs/pretrain_medium_mup_seed42.log" 2>&1 &
PID_PRE_MED=$!

wait $PID_PRE_TINY $PID_PRE_SMALL $PID_PRE_MED
echo "[Phase 1 Complete] Pretraining finished across all 3 model widths."
date

# -------------------------------------------------------------
# STEP 2: Continue-Training with muP across GPUs 1-7
# -------------------------------------------------------------
echo "[Phase 2] Continue-training 7 model variants in parallel on GPUs 1-7..."

# GPU 1: Tiny all_variants
python3 scripts/continue_train.py \
    --config all_variants \
    --pretrain-checkpoint "runs/pretrain/chess_base_tiny_mup_seed42.pt" \
    --use-mup \
    --seed 42 \
    --max-steps 4000 \
    --batch-size 32 \
    --lr 3e-5 \
    --gpu 1 \
    --output runs/variant_train/mup/tiny/ > "runs/logs/ctrain_tiny_mup_all_seed42.log" 2>&1 &
PID_CT_1=$!

# GPU 2: Small all_variants
python3 scripts/continue_train.py \
    --config all_variants \
    --pretrain-checkpoint "runs/pretrain/chess_base_small_mup_seed42.pt" \
    --use-mup \
    --seed 42 \
    --max-steps 4000 \
    --batch-size 32 \
    --lr 3e-5 \
    --gpu 2 \
    --output runs/variant_train/mup/small/ > "runs/logs/ctrain_small_mup_all_seed42.log" 2>&1 &
PID_CT_2=$!

# GPU 3: Medium all_variants
python3 scripts/continue_train.py \
    --config all_variants \
    --pretrain-checkpoint "runs/pretrain/chess_base_medium_mup_seed42.pt" \
    --use-mup \
    --seed 42 \
    --max-steps 4000 \
    --batch-size 16 \
    --lr 3e-5 \
    --gpu 3 \
    --output runs/variant_train/mup/medium/ > "runs/logs/ctrain_medium_mup_all_seed42.log" 2>&1 &
PID_CT_3=$!

# GPU 4: Tiny move_variants
python3 scripts/continue_train.py \
    --config move_variants \
    --pretrain-checkpoint "runs/pretrain/chess_base_tiny_mup_seed42.pt" \
    --use-mup \
    --seed 42 \
    --max-steps 4000 \
    --batch-size 32 \
    --lr 3e-5 \
    --gpu 4 \
    --output runs/variant_train/mup/tiny/ > "runs/logs/ctrain_tiny_mup_move_seed42.log" 2>&1 &
PID_CT_4=$!

# GPU 5: Small move_variants
python3 scripts/continue_train.py \
    --config move_variants \
    --pretrain-checkpoint "runs/pretrain/chess_base_small_mup_seed42.pt" \
    --use-mup \
    --seed 42 \
    --max-steps 4000 \
    --batch-size 32 \
    --lr 3e-5 \
    --gpu 5 \
    --output runs/variant_train/mup/small/ > "runs/logs/ctrain_small_mup_move_seed42.log" 2>&1 &
PID_CT_5=$!

# GPU 6: Medium move_variants
python3 scripts/continue_train.py \
    --config move_variants \
    --pretrain-checkpoint "runs/pretrain/chess_base_medium_mup_seed42.pt" \
    --use-mup \
    --seed 42 \
    --max-steps 4000 \
    --batch-size 16 \
    --lr 3e-5 \
    --gpu 6 \
    --output runs/variant_train/mup/medium/ > "runs/logs/ctrain_medium_mup_move_seed42.log" 2>&1 &
PID_CT_6=$!

# GPU 7: Small chess_only (control)
python3 scripts/continue_train.py \
    --config chess_only \
    --pretrain-checkpoint "runs/pretrain/chess_base_small_mup_seed42.pt" \
    --use-mup \
    --seed 42 \
    --max-steps 4000 \
    --batch-size 32 \
    --lr 3e-5 \
    --gpu 7 \
    --output runs/variant_train/mup/small/ > "runs/logs/ctrain_small_mup_chess_only_seed42.log" 2>&1 &
PID_CT_7=$!

wait $PID_CT_1 $PID_CT_2 $PID_CT_3 $PID_CT_4 $PID_CT_5 $PID_CT_6 $PID_CT_7
echo "[Phase 2 Complete] All continue-training runs finished."
date

# -------------------------------------------------------------
# STEP 3: Parallel Evaluation across GPUs 1-7
# -------------------------------------------------------------
echo "[Phase 3] Running full test evaluation matrix across GPUs 1-7..."

python3 scripts/run_eval.py \
    --checkpoint "runs/variant_train/mup/tiny/all_variants_seed42.pt" \
    --output "runs/results/eval_tiny_mup_all_variants.json" \
    --gpu 1 > "runs/logs/eval_tiny_mup_all.log" 2>&1 &
PID_EV_1=$!

python3 scripts/run_eval.py \
    --checkpoint "runs/variant_train/mup/small/all_variants_seed42.pt" \
    --output "runs/results/eval_small_mup_all_variants.json" \
    --gpu 2 > "runs/logs/eval_small_mup_all.log" 2>&1 &
PID_EV_2=$!

python3 scripts/run_eval.py \
    --checkpoint "runs/variant_train/mup/medium/all_variants_seed42.pt" \
    --output "runs/results/eval_medium_mup_all_variants.json" \
    --gpu 3 > "runs/logs/eval_medium_mup_all.log" 2>&1 &
PID_EV_3=$!

python3 scripts/run_eval.py \
    --checkpoint "runs/variant_train/mup/tiny/move_variants_seed42.pt" \
    --output "runs/results/eval_tiny_mup_move_variants.json" \
    --gpu 4 > "runs/logs/eval_tiny_mup_move.log" 2>&1 &
PID_EV_4=$!

python3 scripts/run_eval.py \
    --checkpoint "runs/variant_train/mup/small/move_variants_seed42.pt" \
    --output "runs/results/eval_small_mup_move_variants.json" \
    --gpu 5 > "runs/logs/eval_small_mup_move.log" 2>&1 &
PID_EV_5=$!

python3 scripts/run_eval.py \
    --checkpoint "runs/variant_train/mup/medium/move_variants_seed42.pt" \
    --output "runs/results/eval_medium_mup_move_variants.json" \
    --gpu 6 > "runs/logs/eval_medium_mup_move.log" 2>&1 &
PID_EV_6=$!

python3 scripts/run_eval.py \
    --checkpoint "runs/variant_train/mup/small/chess_only_seed42.pt" \
    --output "runs/results/eval_small_mup_chess_only.json" \
    --gpu 7 > "runs/logs/eval_small_mup_chess_only.log" 2>&1 &
PID_EV_7=$!

wait $PID_EV_1 $PID_EV_2 $PID_EV_3 $PID_EV_4 $PID_EV_5 $PID_EV_6 $PID_EV_7
echo "[Phase 3 Complete] Evaluation matrix finished."
date

# -------------------------------------------------------------
# STEP 4: Analysis & Figure / Table Generation
# -------------------------------------------------------------
echo "[Phase 4] Running analysis and generating LaTeX tables & figures..."
python3 scripts/analyze_mup_study.py \
    --pretrain-dir runs/pretrain \
    --results-dir runs/results \
    --output-dir runs/analysis

echo "=========================================================="
echo " muP Study Pipeline Complete!"
echo "=========================================================="
date
