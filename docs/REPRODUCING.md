# Reproduction and cluster runbook

## Local environment and run record

Start in the repository root. For cluster scripts, set `export HF_HOME="$PWD/work/hf-cache"`
before tokenizer/model downloads; scripts respect an existing HF_HOME. Install the package as shown in README_V2. Python
3.12.12, chess1.11.2, numpy2.2.6, pytest8.3.5, torch2.6.0+cpu,
transformers4.49.0, peft0.14.0, accelerate1.4.0 and pyffish0.0.91 were used locally.
The entire resolved package list is stored with results. Do not treat that CPU
wheel as a tested H100 environment. H100 CUDA/driver and kernel versions must be
captured in the GPU report. Model/config/tokenizer use immutable revision
`a09a35458c702b33eeacc393d103063234e8bc28`.

Primary executable commands used for local evidence:

```bash
pytest tests/test_v2*.py -q --junitxml=results/tests.xml
python -m chess_icl_v2.cli generate --config configs/stage0_all_rules.json \
  --output results/stage0_all_rules.dataset.json
python -m chess_icl_v2.cli extend --data results/stage0_all_rules.dataset.json \
  --max-plies 1280 --workers 2 --output results/stage0_extended.dataset.json
python -m chess_icl_v2.cli audit --games 3 --plies 120 \
  --output results/simulator_audit_final.json
python scripts/v2_policy_probe.py --data results/stage0_all_rules.dataset.json \
  --output results/terminal_aware_probe.json
python -m chess_icl_v2.cli train --data results/stage0_extended.dataset.json \
  --config configs/v2_tiny_pilot.json --output results/tiny_pilot
python -m chess_icl_v2.cli evaluate --data results/stage0_extended.dataset.json \
  --checkpoint results/tiny_pilot/best.pt --k 0 1 --draws 5 \
  --output results/tiny_validation.json
python scripts/v2_baselines.py --data results/stage0_extended.dataset.json \
  --evaluation results/tiny_validation.json --manifest results/tiny_pilot/manifest.json \
  --output results/tiny_baselines.json
python scripts/v2_pilot_gate.py --evaluation results/tiny_validation.json \
  --baselines results/tiny_baselines.json --output results/pilot_gate.json
```

The first two-rule uniform sample predates the repository link; it uses the same
pinned standard/antichess rules and is kept separately, not merged with the
all-rule sample. All subsequent exploratory data and models use the fixed-policy
objective. Paper numbers must cite the precise run/source/config hashes, not just
file names. Runs started while implementation was still evolving preserve their
start-source hashes; rerun the final package before a confirmatory experiment.

## Extending labels and increasing R

An administrative extension must retain source families, root states, continuation
seeds, and every old prefix. The `extend` command verifies this and appends parent
lineage. It keeps already-complete labels. To obtain higher R, configure a new
generation run with the same seed and all same source settings except R; label
seeds are indexed by root and replicate, so existing replicate indices are
reproduced. This currently repeats deterministic source generation rather than
incrementally appending a persistent database. Record parent hashes and compare
root IDs before combining such runs. Prefer a newly frozen confirmatory seed for
paper data after exploratory Stage0.

## H100 access and profiling

The default restricted shell did not expose NVIDIA devices. Approved host execution
confirmed eight H10080GB GPUs on wnckcaina01, driver580.95.05. CUDA PyTorch2.6.0+cu124
and the pinned Qwen weights are installed in the workspace environment/cache.
Run GPU commands with host device access; do not infer missing hardware from the
restricted shell. The concrete cluster scripts and logs preserve GPU mappings.

On the cluster, install the pinned train dependencies using an appropriate CUDA
PyTorch wheel and run:

```bash
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())'
sbatch cluster/v2_qwen_profile.sh
```

The profile uses one real optimizer update at 4k/8k/16k/24k/32k, records loss, wall
time, peak allocated/reserved memory, then checks save/reload and repeat-prediction
identity. It skips lengths beyond the pinned config and records OOM explicitly.
Tokens used in the memory probe are synthetic repeated IDs; this measures memory
and mechanics, not task quality. The script defaults to SDPA (no uninstalled Flash
Attention dependency). Profile and choose the native token limit before main runs.
Gradient checkpointing is enabled. Never assume a specific physical GPU number.

## Data and training stages

1. Run the simulator and regression suites. Review the four in-house variants'
golden coverage and independently check full examples per rule. Do not call
property testing formal correctness.
2. Choose a policy on development data, considering decisive outcomes, unresolved
rates and measured cost. Freeze rule versions and horizon/claim conventions.
3. Generate a larger family-separated dataset, enough complete demos per fold for
all K, R≥32 training labels and independently high-R final evaluation labels.
Extend the same trajectories where needed. Do not shorten horizon to create draws.
4. Run a scratch pilot on at least two familiar rules and all controls/baselines.
Use three seeds for scientific inference. Preserve negative results. A no-gain
pilot does not authorize expensive main training.
5. Produce `pilot_gate.json` from the actual paired evaluation and baseline files.
Run the necessary-gate command with actual evidence:

```bash
python -m chess_icl_v2.cli gates --data data/v2_complete.json \
  --audits results/cluster_simulator_audit.json --pilot results/pilot_gate.json \
  --profile results/h100_profile.json --output results/main_gates.json
```

6. Generate an adequate **standard chess** warmup dataset with the same policy and
K mixture. The sample with only two demos/fold is insufficient for K=4 warmup.
Train one common warmup parent for each model seed:

```bash
V2_DATA=data/v2_standard.json V2_CONFIG=configs/v2_qwen_cluster_warmup_seed42.json \
V2_OUTPUT=results/qwen_warmup_seed42 sbatch --export=ALL cluster/v2_train.sh
```

7. Clone that parent for each matched main condition:

```bash
V2_DATA=data/v2_complete.json V2_CONFIG=configs/v2_qwen_exploratory_seed42.json \
V2_OUTPUT=results/qwen_familiar_seed42 V2_WARMUP=results/qwen_warmup_seed42/best.pt \
V2_GATES=results/main_gates.json sbatch --export=ALL cluster/v2_train.sh
```

Make new JSON configs for seeds123/456, the zero-context-only training condition
and the separately labeled all-rule exposure reference. Do not edit a config
already associated with a completed run. Rank16/alpha32 requires its own shared
warmup lineage if architecture differs. The loader deliberately rejects a
mismatched architecture. No mid-run optimizer/RNG resume is implemented: a warmup
checkpoint is a weight initialization, not an exact continuation of optimization.

## Eight-GPU allocation

Start one run per allocated GPU, with the scheduler controlling
CUDA_VISIBLE_DEVICES. A candidate packing is six simultaneous Qwen runs, one GPU
for tiny/small pilots and one for evaluation. Three seeds × three conditions need
two waves. This is a proposed schedule, **not measured throughput**. CPU jobs may
request up to the allocation's192 cores only after rollout measurements show it
is useful. The current generator pools label continuations per root; source/demo
generation and deduplication are serial. The simple extrapolation of rollout
core-hours omits this overhead and process startup. Optimize/shard only after
profiling; do not promise linear192-core scaling from the local sample.

## Final-test seal and analysis

Freeze the policy/rules, source seed, splits, checkpoint lineage, K, tokenizer,
metrics and exclusions. Save a protocol JSON listing `heldout_variants` and the
remaining declared choices. The seal binds a dataset and selected checkpoint:

```bash
python -m chess_icl_v2.cli seal --data data/v2_final.json \
  --checkpoint results/qwen_familiar_seed42/best.pt --protocol configs/final_protocol.json \
  --gates results/final_data_gates.json --output results/final_seal_seed42.json
python -m chess_icl_v2.cli evaluate --data data/v2_final.json \
  --checkpoint results/qwen_familiar_seed42/best.pt --split test --k 0 1 2 4 \
  --draws 5 --seal results/final_seal_seed42.json --output results/final_eval_seed42.json
```

The seal is a procedural check, not encrypted storage; Stage0's structural test
fold has already been summarized. Do not reuse it as an unseen confirmatory test.
The current transfer seal rejects a checkpoint trained on a declared held-out
rule. Run a directly exposed reference with a separate protocol that declares
its exposure honestly rather than labeling those rules held out.

Aggregate compatible seed reports with `scripts/v2_aggregate.py`; it rejects
unequal root cohorts. Fit baseline files for each compatible report. Archive raw
prediction rows, baseline rows, noise estimates, hashes, environment, and failed
runs alongside manuscript tables. Calibration and high-R studies require their
own sample sizes/uncertainty review; an exploratory low-R table is not sufficient.
