# Fixed-policy win/draw/loss research pipeline

Predict empirical W/D/L probabilities for the **player to move**, under a fixed,
versioned policy in an exact variant simulator. These probabilities do not estimate
optimal or strong play. Full demonstration games plus the complete query state
are the only model input; rule names/descriptions and query labels are excluded.
Evaluation freezes all model and adapter weights.

**Completed H100 study:** three primary Qwen-7B seeds and one all-rule exposure
reference, 614 R32 roots, and 56 independent R256 test roots. The K1 experiment
does **not demonstrate beneficial rule-specific ICL**: paired NLL gain is
−0.034650 for familiar rules and −0.013547 for held-out rules; relevance advantages
over matched foreign-rule context have intervals spanning zero. These are small
exploratory results under the declared policy. See the [full study and intervals](docs/EXPLORATORY_STUDY.md),
[publication snapshot](results/published/final-test-v2/), [CSV/LaTeX tables](tables/),
and [figure captions](figures/README.md). All 51 tests and the complete prediction
audit pass. Full raw data, predictions, traces and checkpoints are in the local
research-artifact ZIP, with exact hashes in the snapshot.

Implementation lives in `chess_icl_v2/`. The original material-label package,
scripts, tests, configurations and results are preserved in
[`archive/legacy_material_labels/`](archive/README.md). Their material/centipawn targets
and old accuracy numbers are not evidence for this experiment.

**Cluster update:** eight H100 GPUs are now accessible through approved host execution.
The earlier sandbox could not see devices. Actual Qwen training and profiling
results are recorded in [CLUSTER_RESULTS.md](docs/CLUSTER_RESULTS.md).

## Read before using results

- [Completed exploratory study: methods, results and interpretation](docs/EXPLORATORY_STUDY.md)
- [Actual cluster results](docs/CLUSTER_RESULTS.md)
- [Artifact validity and superseded runs](docs/ARTIFACT_STATUS.md)
- [Historical Stage0 results](archive/development_stage0/RESULTS.md)
- [Exact rules and terminal conventions](docs/RULES.md)
- [Every design decision and development incident](docs/DECISIONS.md)
- [Data/model contracts and implementation map](docs/IMPLEMENTATION.md)
- [Analysis plan and confirmatory protocol](docs/ANALYSIS_PLAN.md)
- [Reproduction and cluster operations](docs/REPRODUCING.md)

The study is exploratory and small. Its explicit exploratory seal preserves the
negative scratch-pilot gate; simulator/data/R/hardware gates pass. The initial
CPU-stage findings are retained separately. Final test inference uses individually
scored unpadded episodes after the documented numerical audit; original padded
reports are retained as diagnostics. Checkpoint selection was not repeated.

## Install and verify

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[train,test]'
pip install pyffish==0.0.91
pytest tests/test_v2*.py -q
python -m chess_icl_v2.cli audit --output results/my_audit.json
```

The pinned PyTorch wheel should match the cluster's CUDA runtime. Completed H100
runs used torch 2.6.0+cu124; the earlier CPU stage used torch 2.6.0+cpu. Both package
records are retained in the full artifact bundle and published environment record.

## Generate, extend and inspect

```bash
python -m chess_icl_v2.cli generate \
  --config configs/stage0_all_rules.json --output results/my_stage0.json
python -m chess_icl_v2.cli extend --data results/my_stage0.json \
  --max-plies 1280 --workers 2 --output results/my_extended.json
python -m chess_icl_v2.cli tokens --data results/my_extended.json \
  --tokenizer Qwen/Qwen2.5-7B-Instruct \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --output results/my_tokens.json
```

Never turn unresolved outcomes into draws. A label is usable only when all R
continuations terminate. `extend` replays the same streams and confirms their old
prefixes, keeps the roots fixed, and preserves lineage. It does not sample until
a convenient result appears.

## CPU pilot and frozen controls

```bash
python -m chess_icl_v2.cli train --data results/my_extended.json \
  --config configs/v2_tiny_pilot.json --output results/my_tiny
python -m chess_icl_v2.cli evaluate --data results/my_extended.json \
  --checkpoint results/my_tiny/best.pt --k 0 1 --draws 5 \
  --output results/my_tiny_validation.json
python scripts/v2_baselines.py --data results/my_extended.json \
  --evaluation results/my_tiny_validation.json \
  --manifest results/my_tiny/manifest.json --output results/my_baselines.json
```

The pilot config explicitly permits logged coverage exclusions. Main configs do
not. No headline result should be computed by silently averaging different roots
at different K values. Mismatch controls must fit and match demo token length to
within 10%, or the root is excluded from the entire paired evaluation matrix.

## H100 work

Run `cluster/v2_qwen_profile.sh` first. Then use `cluster/v2_train.sh` with the
`V2_DATA`, `V2_CONFIG`, `V2_OUTPUT`, `V2_WARMUP` and `V2_GATES` variables documented
in [REPRODUCING.md](docs/REPRODUCING.md). Never overwrite scheduler-provided GPU
visibility. Do not run the archived `archive/legacy_material_labels/cluster/run_all.sh` for v2.

Research gates require simulator validation, complete labels, measured outcome
signal, successful pilot comparisons, and GPU measurements. The gate command
returns nonzero when requirements are missing. A saved test seal binds the exact
dataset, checkpoint and analysis protocol; the evaluator rejects unsealed final
test access. This is a procedural seal, not an encrypted data-access system.
