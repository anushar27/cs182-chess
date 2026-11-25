# H100 fixed-policy experiment results

This report supersedes the earlier hardware-access conclusion in the CPU/Stage0
snapshot. Restricted execution hid the GPU devices. Approved host execution on
`wnckcaina01` confirmed **8 H10080GB GPUs and192 logical CPUs**,
driver580.95.05, PyTorch2.6.0+cu124 and CUDA12.4. The earlier negative
scratch result remains a result; it was not evidence that the host lacked GPUs.
[Environment](../results/published/validated-core-v2/cluster_environment.json), [decisions](DECISIONS.md).

## Actual Qwen-7B CUDA profile

Pinned Qwen2.5-7B-Instruct backbone, last-four-layer LoRA r8/alpha16 and FP32 W/D/L
head. All lengths below completed a real optimizer update. Exact repeated
predictions and checkpoint save/reload passed. These repeated-token probes measure
memory/mechanics, not chess performance. Read cold-start and sharing effects into
timing rather than inferring a smooth throughput curve from five updates.

| Tokens | Update seconds | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|---:|
| 4096 | 1.569 | 15.377 | 15.734 |
| 8192 | 0.865 | 17.485 | 18.625 |
| 16384 | 2.075 | 21.686 | 24.047 |
| 24576 | 3.707 | 25.886 | 31.854 |
| 32768 | 5.694 | 30.088 | 38.791 |

[Raw profile](../results/published/validated-core-v2/h100_profile.json); full warnings/log in the local artifact ZIP.
The SDPA sliding-window warning was retained and inspected: use_sliding_window
is false; configured sliding_window131072 exceeds native context32768. No context
extension or custom attention mask was introduced.

## Larger development dataset under terminal_aware_v1

Seed2026112402; R32; administrative budget2560;8 demonstration and8 query-source
families per fold per rule; three disjoint folds. **614 roots and
19,648 label continuations** retained.
All continuations and demonstration games terminated; no censoring was converted
to draws. The original generation phase measured **217.97s** with48
persistent workers, before the final audit. The capture-rule repair and evaluation-demo
extension are additional retained phases, not included in that number. Near-duplicate and replay audits passed. There are
47 outcome-independent source/root exclusions, retained with reasons.
The exact compact Hamming index is regression-tested against exhaustive distance;
the parallel generator is tested against the reference seed/trajectory semantics.

Unlike the first uniform-policy sample, development roots show substantial decisive
outcomes. Root distributions also changed because source games use the declared
policy; do not interpret the table as a randomized paired policy effect. The earlier
same-root small policy probe is the paired diagnostic.

| Rule | Train+validation roots | W continuations | D | L | Unresolved |
|---|---:|---:|---:|---:|---:|
| standard | 32 | 416 | 242 | 366 | 0 |
| antichess | 32 | 500 | 8 | 516 | 0 |
| diagonal_rook | 26 | 299 | 307 | 226 | 0 |
| straight_bishop | 31 | 389 | 216 | 387 | 0 |
| weakened_queen | 28 | 317 | 245 | 334 | 0 |
| extended_pawn | 29 | 360 | 210 | 358 | 0 |
| short_knight | 28 | 282 | 364 | 250 | 0 |
| mobile_king | 25 | 109 | 595 | 96 | 0 |
| piece_capture | 30 | 476 | 14 | 470 | 0 |
| forward_pawn_capture | 30 | 369 | 246 | 345 | 0 |
| dark_square_immunity | 30 | 111 | 715 | 134 | 0 |
| blocked_squares | 26 | 174 | 465 | 193 | 0 |
| extended_initial_pawn | 26 | 309 | 237 | 286 | 0 |
| knight_army | 30 | 265 | 426 | 269 | 0 |

[Corrected-data/evaluation manifest](../results/published/final-test-v2/raw_artifact_manifest.json),
[artifact status](ARTIFACT_STATUS.md),
[config](../configs/v2_terminal_pilot.json); full generation log in the artifact ZIP.
Labels are root-player W/D/L under this policy. They are not engine strength,
material adjudications or optimal-play outcomes. Test labels were not used to
train or select checkpoints; the high-R subset selection uses family hashes only.

## Training and checkpoint lineage

Exploratory matrix fixed before opening its test-model scores: seeds42/123/456,
30-update same-seed standard warmup,200-update familiar-rule training, K mixture0/1,
accumulation4, LR1e-4,20-update LR warmup and validation every40 updates. All base
weights stay frozen; **731,651** LoRA/head parameters train. All selected checkpoints
use macro validation NLL. This is a measured small-scale experiment, not the
previously illustrative2,000-update full matrix. No positive-result stopping rule.

| Seed | Condition | Updates | Best validation NLL | Training seconds | Peak allocated GiB |
|---|---|---:|---:|---:|---:|
| 42 | standard warmup | 30 | 1.188974 | 16.04 | 16.141 |
| 42 | four familiar rules | 200 | 1.081595 | 200.26 | 17.896 |
| 123 | standard warmup | 30 | 1.234268 | 16.48 | 16.132 |
| 123 | four familiar rules | 200 | 1.100553 | 206.98 | 17.897 |
| 456 | standard warmup | 30 | 1.165793 | 21.15 | 16.146 |
| 456 | four familiar rules | 200 | 1.090572 | 337.50 | 17.900 |
| 42 | all-rule exposure reference | 200 | 1.020528 | 271.72 | 20.638 |

Training time is the instrumented optimization/validation loop, excluding model
loading and episode construction. GPU sharing and other users' workloads can
change timing. No unrelated processes were stopped. The actual GPU selection and
checkpoint hashes are in [the matrix status](../results/published/final-test-v2/cluster_matrix_status.json).
Parent manifests record the union of all gradient-exposed rules, so an exposed
parent cannot accidentally count as held-out transfer.

## Frozen evaluation

The four training rules are diagonal_rook, straight_bishop, weakened_queen and
extended_pawn; the remaining nine project rules are held out from gradients.
Standard is reported separately. The all-rule condition is an exposure reference,
not a theoretical upper bound or a held-out transfer result. Qwen pretraining may
already contain named variants; no contamination-free claim is made.

Test access binds dataset/checkpoint/protocol hashes and explicitly marks these
experiments exploratory. Data/simulator/R/hardware gates all pass; a positive-pilot
gain is not fabricated. K0, relevantK1, mismatchedK1 and both corruption controls
use common roots, five demo draws and exact length matching; all exclusions are
reported. All weights remain frozen. Larger-K experiments are separate from the
registered primaryK1 analysis.

### Validation results across three seeds

| Exposure | Condition | K | Macro NLL | Vector Brier |
|---|---|---:|---:|---:|
| familiar_rules | corrupt_outcomes | 1 | 1.099065 | 0.121230 |
| familiar_rules | corrupt_trajectories | 1 | 1.106808 | 0.127282 |
| familiar_rules | mismatched | 1 | 1.107444 | 0.127515 |
| familiar_rules | relevant | 1 | 1.105829 | 0.126725 |
| familiar_rules | zero | 0 | 1.075342 | 0.108111 |
| heldout_rules | corrupt_outcomes | 1 | 1.077171 | 0.220584 |
| heldout_rules | corrupt_trajectories | 1 | 1.073384 | 0.218468 |
| heldout_rules | mismatched | 1 | 1.079870 | 0.222583 |
| heldout_rules | relevant | 1 | 1.074777 | 0.219449 |
| heldout_rules | zero | 0 | 1.081327 | 0.223864 |
| standard_control | corrupt_outcomes | 1 | 1.109822 | 0.123722 |
| standard_control | corrupt_trajectories | 1 | 1.097996 | 0.115377 |
| standard_control | mismatched | 1 | 1.098288 | 0.115595 |
| standard_control | relevant | 1 | 1.095247 | 0.113587 |
| standard_control | zero | 0 | 1.047610 | 0.085430 |

| Exposure | Context | Paired NLL gain | Hierarchical95% interval |
|---|---|---:|---|
| familiar_rules | mismatchedK1 | -0.032102 | [-0.044709, -0.020495] |
| familiar_rules | relevantK1 | -0.030486 | [-0.041733, -0.019843] |
| heldout_rules | mismatchedK1 | 0.001457 | [-0.063253, 0.043488] |
| heldout_rules | relevantK1 | 0.006550 | [-0.060615, 0.052974] |
| standard_control | mismatchedK1 | -0.050679 | [-0.076015, -0.025754] |
| standard_control | relevantK1 | -0.047637 | [-0.074991, -0.022493] |

[Full aggregate](../results/published/final-test-v2/qwen_validation_aggregate.json). The hierarchy covers model seeds, demo draws and source families; Monte Carlo target noise is separate.

### Test results across three seeds

| Exposure | Condition | K | Macro NLL | Vector Brier |
|---|---|---:|---:|---:|
| familiar_rules | corrupt_outcomes | 1 | 1.098931 | 0.127767 |
| familiar_rules | corrupt_trajectories | 1 | 1.104345 | 0.131958 |
| familiar_rules | mismatched | 1 | 1.106581 | 0.133573 |
| familiar_rules | relevant | 1 | 1.103794 | 0.131821 |
| familiar_rules | zero | 0 | 1.069144 | 0.109869 |
| heldout_rules | corrupt_outcomes | 1 | 1.092516 | 0.224804 |
| heldout_rules | corrupt_trajectories | 1 | 1.089970 | 0.223716 |
| heldout_rules | mismatched | 1 | 1.095761 | 0.227698 |
| heldout_rules | relevant | 1 | 1.088554 | 0.222578 |
| heldout_rules | zero | 0 | 1.075007 | 0.215263 |
| standard_control | corrupt_outcomes | 1 | 1.106303 | 0.098613 |
| standard_control | corrupt_trajectories | 1 | 1.100416 | 0.095777 |
| standard_control | mismatched | 1 | 1.102852 | 0.097209 |
| standard_control | relevant | 1 | 1.101167 | 0.096765 |
| standard_control | zero | 0 | 1.077991 | 0.081892 |

| Exposure | Context | Paired NLL gain | Hierarchical95% interval |
|---|---|---:|---|
| familiar_rules | mismatchedK1 | -0.037437 | [-0.063466, -0.018842] |
| familiar_rules | relevantK1 | -0.034650 | [-0.058684, -0.015994] |
| heldout_rules | mismatchedK1 | -0.020754 | [-0.079711, 0.019049] |
| heldout_rules | relevantK1 | -0.013547 | [-0.071814, 0.023449] |
| standard_control | mismatchedK1 | -0.024861 | [-0.050613, -0.003227] |
| standard_control | relevantK1 | -0.023176 | [-0.058242, 0.005585] |

[Full aggregate](../results/published/final-test-v2/qwen_test_aggregate.json). The hierarchy covers model seeds, demo draws and source families; Monte Carlo target noise is separate.

### Test_coverage results across three seeds

| Exposure | Condition | K | Macro NLL | Vector Brier |
|---|---|---:|---:|---:|
| familiar_rules | relevant | 1 | 1.103794 | 0.131821 |
| familiar_rules | zero | 0 | 1.069144 | 0.109869 |
| heldout_rules | relevant | 1 | 1.086424 | 0.219400 |
| heldout_rules | zero | 0 | 1.080936 | 0.217107 |
| standard_control | relevant | 1 | 1.101167 | 0.096765 |
| standard_control | zero | 0 | 1.077991 | 0.081892 |

| Exposure | Context | Paired NLL gain | Hierarchical95% interval |
|---|---|---:|---|
| familiar_rules | relevantK1 | -0.034650 | [-0.058684, -0.015994] |
| heldout_rules | relevantK1 | -0.005488 | [-0.066524, 0.028899] |
| standard_control | relevantK1 | -0.023176 | [-0.058242, 0.005585] |

[Full aggregate](../results/published/final-test-v2/qwen_test_coverage_aggregate.json). The hierarchy covers model seeds, demo draws and source families; Monte Carlo target noise is separate.

## Numerical inference audit

Original batch8 test outputs are retained as diagnostics. Final test inputs were reconstructed and hash-checked, then evaluated individually without padding; validation checkpoint selection is unchanged. No weights/targets/cohorts were changed.

| Run | Unique inputs | Rescore wall seconds | Max probability change from padded | Mean absolute change |
|---|---:|---:|---:|---:|
| direct_exposure_seed42 | 4208 | 282.18 | 0.072261 | 0.010756 |
| qwen_seed123 | 4208 | 299.01 | 0.057541 | 0.011234 |
| qwen_seed42 | 4208 | 743.58 | 0.077508 | 0.010123 |
| qwen_seed456 | 4208 | 690.46 | 0.050487 | 0.009428 |

Every run verified unchanged parameter versions and an uncached repeated output. The final eleven-report audit passes; overlapping primary/coverage inputs now have exactly identical probabilities. See the [complete study](EXPLORATORY_STUDY.md) for negative results, baselines, exclusions and independent R256 sensitivity.

## Scope and limitations

This is the full retained record of the implemented small-scale pipeline and
measured cluster experiment. It does not establish optimal play, a scaling law,
a full-powered confirmatory result or a validated μP study. Simulator checks are
sampled; four in-house rules lack an independent-engine equivalence proof. Query
histories may reveal rules already atK0, and setup/barrier geometry is observable.
The independent high-R analysis covers56 roots/28 families (53 roots/27 families in the matched-control cohort), with its actual counts and uncertainty reported separately.
The manuscript should distinguish this exploratory study from a newly preregistered,
larger confirmatory experiment. Every positive/negative condition belongs in the
results, including failed feasibility stages and discarded controls.
