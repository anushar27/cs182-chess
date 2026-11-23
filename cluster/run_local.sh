#!/bin/bash
# Local multi-GPU runner for CS182 Chess ICL Research Pipeline
# Automatically distributes jobs across available GPUs (e.g. 8x H100)

set -e
export TMPDIR=/tmp
export MPLCONFIGDIR=/tmp/matplotlib
mkdir -p /tmp/matplotlib runs/logs runs/pretrain runs/variant_train runs/results runs/analysis

echo "=========================================================="
echo " Starting Full Chess ICL Research Pipeline"
echo "=========================================================="

# 1. Generate full dataset (CPU multiprocessing)
echo "Step 1: Generating datasets..."
python3 scripts/generate_data.py --all-variants --count 2000 --seed 42 > runs/logs/datagen.log 2>&1
echo "Data generation complete."

# 2. Pretraining in parallel on GPUs 1, 2, 3
echo "Step 2: Pretraining on standard chess (3 seeds in parallel)..."
SEEDS=(42 123 456)
GPUS=(1 2 3)

for idx in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$idx]}"
    GPU="${GPUS[$idx]}"
    echo "  Launching pretrain seed $SEED on GPU $GPU..."
    python3 scripts/pretrain.py \
        --seed "$SEED" \
        --config small \
        --max-steps 20000 \
        --max-epochs 100 \
        --gpu "$GPU" \
        --output runs/pretrain/ > "runs/logs/pretrain_seed${SEED}.log" 2>&1 &
done
wait
echo "Pretraining complete."

# 3. Continue-training in parallel on GPUs 1-6 (3 configs x 3 seeds = 9 jobs)
echo "Step 3: Continue-training on variants (9 jobs distributed across GPUs)..."
CONFIGS=("move_variants" "all_variants" "chess_only")
JOB_GPUS=(1 2 3 4 5 6)
PIDS=()
JOB_COUNT=0

for CONFIG in "${CONFIGS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        GPU_IDX=$((JOB_COUNT % 6))
        GPU="${JOB_GPUS[$GPU_IDX]}"
        echo "  Launching $CONFIG seed $SEED on GPU $GPU..."
        python3 scripts/continue_train.py \
            --config "$CONFIG" \
            --pretrain-checkpoint "runs/pretrain/chess_base_seed${SEED}.pt" \
            --seed "$SEED" \
            --max-steps 10000 \
            --gpu "$GPU" \
            --output runs/variant_train/ > "runs/logs/ctrain_${CONFIG}_seed${SEED}.log" 2>&1 &
        PIDS+=($!)
        JOB_COUNT=$((JOB_COUNT + 1))
        # Batch by available GPUs
        if [ "$((JOB_COUNT % 6))" -eq 0 ]; then
            echo "  Waiting for active batch of jobs to finish..."
            wait
        fi
    done
done
wait
echo "Continue-training complete."

# 4. Evaluation across all test variants
echo "Step 4: Running full evaluation matrix..."
EVAL_GPUS=(1 2 3)
E_IDX=0
for CONFIG in "${CONFIGS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CKPT="runs/variant_train/${CONFIG}_seed${SEED}.pt"
        if [ -f "$CKPT" ]; then
            GPU="${EVAL_GPUS[$((E_IDX % 3))]}"
            echo "  Evaluating $CKPT on GPU $GPU..."
            python3 scripts/run_eval.py \
                --checkpoint "$CKPT" \
                --test-variants short_knight mobile_king piece_capture \
                    forward_pawn_capture dark_square_immunity blocked_squares \
                    long_board knight_army antichess \
                --k-values 0 1 2 4 8 16 \
                --seeds 42 123 456 \
                --gpu "$GPU" \
                --output "runs/results/${CONFIG}_seed${SEED}.json" > "runs/logs/eval_${CONFIG}_seed${SEED}.log" 2>&1 &
            E_IDX=$((E_IDX + 1))
            if [ "$((E_IDX % 3))" -eq 0 ]; then
                wait
            fi
        fi
    done
done
wait
echo "Evaluation complete."

# 5. Analysis & Visualization
echo "Step 5: Aggregating results and generating figures & LaTeX tables..."
python3 scripts/analyze_results.py --results-dir runs/results --output-dir runs/analysis
echo "=========================================================="
echo " Pipeline Complete! Results saved to runs/analysis/"
echo "=========================================================="
