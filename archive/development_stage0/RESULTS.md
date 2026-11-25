> Capture-conversion rule v1 labels/controls in this historical snapshot were later
> superseded. Use [ARTIFACT_STATUS.md](ARTIFACT_STATUS.md) and the cluster report.

> Historical CPU/Stage0 snapshot. The host does have eight H100 GPUs; the initial
> sandbox hid device access. See [CLUSTER_RESULTS.md](CLUSTER_RESULTS.md) for the
> subsequent CUDA profile, larger dataset and actual Qwen experiments.

# Measured results and remaining work

These are **exploratory local results**, not completed 7B/H100 or confirmatory
paper results. Client calendar date: November 24, 2026. Raw machine timestamps
are UTC. All tables below are generated from retained artifacts by
`scripts/v2_write_results.py`.

## Verification

- V2 regression/integration suite: **37 tests, 0 failures**. Includes rule attacks,
special moves, state/repetition round trips, deterministic independent workers,
query retention/leakage, paired controls, proper scores, checkpoint reload/frozen
inference, tiny local Qwen LoRA injection/backprop and warmup lineage.
- Simulator audit: **4808 reachable positions**, **3452 independent
Fairy-Stockfish legal-move comparisons**, and **4 encountered independent
no-move terminal result comparisons**; all passed. This is limited sampled coverage,
not proof or comprehensive terminal equivalence. Four rules use golden/property
coverage only. [Audit artifact](../results/simulator_audit_final.json).
- Qwen tokenizer/config downloaded at immutable revision
`a09a35458c702b33eeacc393d103063234e8bc28`; native configured context: **32,768**.
No 7B weights were downloaded or trained locally. The local tiny Qwen architecture
test is a mechanics test, not a 7B experiment.
- Python wheel build and cluster shell syntax checked. Full environment and test
logs: [environment](../results/environment.txt), [tests](../results/pytest.log).

## Stage0, all 13 project rules plus standard

Seed20261124; uniform_legal_v1; one query-source family per fold per rule, two
adjacent opposite-side roots/family; R=4. Original pools:84 demonstration games,
42 query-source games,84 roots and336 independent label continuations.
Initial administrative budget:320 plies. Generation wall time:
**225.82 seconds** on this host; process/root replay,
source generation and deduplication are included in that wall time.

**189/336 continuations were unresolved at320 plies.** Extending the same
streams to1280 resolved all336; none were reclassified by material or silently
counted as draws. All84 query states and replicate seeds were retained. One whole
demonstration was excluded because its newly visible suffix was near a query;
its raw trace is retained.83 terminal demonstrations remain. Main-pool integrity
and both-side coverage passed. First extension failure/recomputation cost was not
separately instrumented; do not call225.82 seconds the total cost of all development.

Each row contains24 continuations (6 roots ×4); W/D/L are root-player outcomes.
The original320-ply rollout core-hour projections exclude startup, root replay,
deduplication and extension costs; they are not measured192-core throughput.

| Rule | Unresolved at320 | W at1280 | D at1280 | L at1280 | Unresolved at1280 |
|---|---:|---:|---:|---:|---:|
| standard | 3 | 1 | 23 | 0 | 0 |
| antichess | 0 | 13 | 1 | 10 | 0 |
| diagonal_rook | 14 | 0 | 24 | 0 | 0 |
| straight_bishop | 15 | 0 | 21 | 3 | 0 |
| weakened_queen | 13 | 1 | 21 | 2 | 0 |
| extended_pawn | 18 | 1 | 23 | 0 | 0 |
| short_knight | 17 | 1 | 23 | 0 | 0 |
| mobile_king | 22 | 0 | 23 | 1 | 0 |
| piece_capture | 9 | 7 | 11 | 6 | 0 |
| forward_pawn_capture | 14 | 2 | 21 | 1 | 0 |
| dark_square_immunity | 24 | 0 | 24 | 0 | 0 |
| blocked_squares | 22 | 0 | 24 | 0 | 0 |
| extended_initial_pawn | 11 | 2 | 21 | 1 | 0 |
| knight_army | 7 | 0 | 23 | 1 | 0 |

Raw data and tables: [320-ply dataset](../results/stage0_all_rules.dataset.json),
[extended dataset](../results/stage0_extended.dataset.json),
[CSV](../results/stage0_table.csv), [LaTeX](../results/stage0_table.tex).
R=4 is far too small to establish fine probability differences; label uncertainty
is saved [separately](../results/label_uncertainty.json). Stage0 structural test
folds were inspected as development diagnostics and are not sealed paper tests.

## Terminal-aware development probe

Same two development/train roots per rule, two fresh continuations per root,
320-ply budget (56 continuations total). This is a separate policy experiment;
its labels are never merged into the uniform-policy dataset. Some movement/capture
rules show more decisive outcomes, while other rules remain draw-dominated or
censored. R=2 does not justify selecting a final policy or claiming improvement.
Every trace and cost is in [the probe](../results/terminal_aware_probe.json).

| Rule | W | D | L | Unresolved | Measured rollout seconds |
|---|---:|---:|---:|---:|---:|
| antichess | 2 | 0 | 2 | 0 | 0.0325 |
| blocked_squares | 0 | 1 | 0 | 3 | 0.9037 |
| dark_square_immunity | 0 | 1 | 0 | 3 | 3.6402 |
| diagonal_rook | 0 | 2 | 1 | 1 | 0.8589 |
| extended_initial_pawn | 0 | 3 | 0 | 1 | 0.8815 |
| extended_pawn | 2 | 2 | 0 | 0 | 0.6007 |
| forward_pawn_capture | 1 | 0 | 3 | 0 | 1.0840 |
| knight_army | 0 | 4 | 0 | 0 | 0.7676 |
| mobile_king | 0 | 0 | 2 | 2 | 1.3776 |
| piece_capture | 0 | 0 | 4 | 0 | 0.3925 |
| short_knight | 1 | 1 | 0 | 2 | 0.9778 |
| standard | 0 | 4 | 0 | 0 | 0.3508 |
| straight_bishop | 0 | 4 | 0 | 0 | 0.6005 |
| weakened_queen | 0 | 0 | 0 | 4 | 2.4351 |

## Actual Qwen token measurements

Move and complete-episode counts use the downloaded Qwen tokenizer. Full games
are retained. SAN comparison is restricted to standard/antichess. The choice of
UCI for custom rules is a reproducibility choice, not evidence that it is globally
shorter than SAN. The tiny scratch vocabulary is measured separately from Qwen.

| K | Available full episodes | Median Qwen tokens | Maximum Qwen tokens |
|---|---:|---:|---:|
| 0 | 84 | 696.5 | 1699 |
| 1 | 84 | 3294.5 | 5805 |
| 2 | 82 | 5731.0 | 9887 |
| 4 | 0 | unavailable | unavailable |
| 8 | 0 | unavailable | unavailable |
| 16 | 0 | unavailable | unavailable |

K≥4 is unavailable because Stage0 has only two demo games/fold/rule, not because
this establishes a tokenizer or GPU context limit. Counts at larger K require a
larger demo pool. [Complete measurements](../results/qwen_extended_token_lengths.json).
No H100 memory/throughput measurement exists.

## Scratch pipeline pilot and frozen evaluation

Single seed42, 23,523 parameters, one32-wide causal RoPE
block,337-token square/move vocabulary,30 optimizer updates,16 training episodes
and16 validation episodes over the four familiar rules, K=0/1. Training measured
0.958 seconds after episode construction on CPU; this excludes model/data
setup, evaluation and baseline time. No warmup was used for this debugging run;
main Qwen runs require a common standard warmup. Checkpoint selection used macro
validation NLL on all16 validation episodes: **0.288970**.
[Manifest](../results/tiny_pilot/manifest.json), [training log](../results/tiny_pilot/training_log.json).

Frozen evaluation: **420 predictions**,5 demo draws/K>0, K=0 once/root,
and **8 excluded roots** because a full relevant/mismatched context
pair could not meet10% demo-length tolerance. Exclusions were applied to every
condition for that root. Excluded rule counts: `{'antichess': 2, 'diagonal_rook': 1, 'straight_bishop': 1, 'short_knight': 2, 'dark_square_immunity': 2}`.
These exclusions affect representativeness; do not extrapolate to all held-outs.

| Exposure | Condition | K | Macro NLL | Vector Brier |
|---|---|---:|---:|---:|
| familiar_rules | corrupt_outcomes | 1 | 0.195447 | 0.022187 |
| familiar_rules | corrupt_trajectories | 1 | 0.195371 | 0.022171 |
| familiar_rules | mismatched | 1 | 0.195240 | 0.022159 |
| familiar_rules | relevant | 1 | 0.195448 | 0.022187 |
| familiar_rules | zero | 0 | 0.193117 | 0.021417 |
| heldout_rules | corrupt_outcomes | 1 | 0.521867 | 0.063809 |
| heldout_rules | corrupt_trajectories | 1 | 0.521846 | 0.063810 |
| heldout_rules | mismatched | 1 | 0.522149 | 0.063888 |
| heldout_rules | relevant | 1 | 0.521857 | 0.063806 |
| heldout_rules | zero | 0 | 0.523898 | 0.064390 |
| standard_control | corrupt_outcomes | 1 | 0.113023 | 0.017310 |
| standard_control | corrupt_trajectories | 1 | 0.113059 | 0.017322 |
| standard_control | mismatched | 1 | 0.112853 | 0.017254 |
| standard_control | relevant | 1 | 0.113026 | 0.017311 |
| standard_control | zero | 0 | 0.109368 | 0.016247 |

The main familiar-rule comparison is negative: relevant context does not improve
NLL over K=0, and mismatched context is about as effective. The small positive
held-out gain also occurs with mismatched context; it does not establish transfer.
There is no demonstrated useful rule ICL in this pilot. Single-seed bootstrap
intervals below omit seed uncertainty and all Monte Carlo target noise; each rule
has only one validation source family, so they are especially inadequate for
scientific significance claims.

| Exposure | Context | Paired NLL gain | Conditional bootstrap95% interval |
|---|---|---:|---|
| familiar_rules | mismatched | -0.002123 | [-0.002635, -0.001600] |
| familiar_rules | relevant | -0.002331 | [-0.002841, -0.001830] |
| heldout_rules | mismatched | 0.001749 | [0.001167, 0.002115] |
| heldout_rules | relevant | 0.002042 | [0.001381, 0.002536] |
| standard_control | mismatched | -0.003485 | [-0.003517, -0.003444] |
| standard_control | relevant | -0.003658 | [-0.003666, -0.003650] |

Calibration bins/by-side scores, equal-material slices, class counts, corruptions,
all per-root probabilities and exact input/demo hashes are in
[the evaluation artifact](../results/tiny_validation.json).

## Baselines on the same familiar-rule relevant-context cohort

| Baseline | Macro NLL | Vector Brier |
|---|---:|---:|
| demo_frequency | 0.818780 | 0.418750 |
| demo_majority | 0.818780 | 0.418750 |
| knn_demo_states | 0.612438 | 0.273438 |
| length_histogram | 0.134927 | 0.011751 |
| query_material | 0.231863 | 0.039271 |
| training_frequency | 0.170826 | 0.015625 |
| training_majority | 0.714808 | 0.343750 |

Every baseline prediction and all other exposure/K groups are retained in
[the baseline artifact](../results/tiny_baselines.json). Learned baselines use
training folds only. These are small linear/retrieval controls, not optimized
state-of-the-art chess predictors. Beating kNN alone is insufficient: relevant
context must also beat K=0 and mismatched context.

## Main gates and unexecuted work

- `all_requested_rules_present`: **pass**.
- `both_sides_all_folds`: **pass**.
- `dataset_integrity`: **pass**.
- `measured_h100_profile`: **FAIL / unavailable**.
- `minimum_replicates`: **FAIL / unavailable**.
- `no_censored_targets`: **pass**.
- `outcome_variation_two_familiar_rules`: **pass**.
- `pilot_context_gain`: **FAIL / unavailable**.
- `simulator_audits`: **pass**.

The main run is blocked. Current R=4 is below the required32; the pilot lacks useful
relevant-context gain; no H100 profile or usable remote desktop/terminal exists
in this chat. The CPU host's nvidia-smi cannot communicate with a driver. GPU
scripts and pinned Qwen training code are provided, but **7B training, H100 timing,
8-GPU packing, three-model-seed results, high-R confirmatory labels and sealed-test
results have not been produced**. No old repository accuracy plot is reused.

Next work is development of a useful policy/root distribution, a larger independent
dataset and multi-seed pilot, followed by actual H100 profiling and gated training.
Do not launch the expensive matrix solely because the implementation runs.

Known limitations: sampled simulator coverage; four in-house rule implementations;
rule conventions that differ from a team's unstated interpretation; conservative
near-duplicate filtering; query histories can reveal rules even at K=0; missing
long-K demo pools; low sample sizes; untested7B/CUDA path; no exact mid-run optimizer
resume or validated μP scaling track. [Decisions and incidents](DECISIONS.md).
