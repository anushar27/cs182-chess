#!/bin/bash
# Submit the full pipeline with SLURM dependencies
set -e

echo "=== Chess ICL Pipeline ==="

# Step 1: Data generation
DATAGEN=$(sbatch --parsable cluster/generate_data.sh)
echo "Data generation job: $DATAGEN"

# Step 2: Pretraining (3 seeds)
PRETRAIN=$(sbatch --parsable --dependency=afterok:$DATAGEN cluster/pretrain.sh)
echo "Pretrain array job: $PRETRAIN"

# Step 3: Continue-training (9 jobs: 3 seeds x 3 configs)
CTRAIN=$(sbatch --parsable --dependency=afterok:$PRETRAIN cluster/continue_train.sh)
echo "Continue-train array job: $CTRAIN"

# Step 4: Evaluation & Analysis
EVAL=$(sbatch --parsable --dependency=afterok:$CTRAIN cluster/evaluate.sh)
echo "Evaluation job: $EVAL"

echo ""
echo "Pipeline submitted. Monitor with: squeue -u $USER"
echo "Dependency chain: $DATAGEN -> $PRETRAIN -> $CTRAIN -> $EVAL"
