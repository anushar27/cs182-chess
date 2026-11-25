# Long-training study: completed plan, version 1

This document specifies the next experiment. It is a plan, not a claim that the
training runs or their results exist. No new long-training run has launched.
The previous pilot and its negative results remain unchanged.

## 1. Objective and fixed boundaries

Predict player-to-move win/draw/loss probabilities when both players follow
`terminal_aware_v1`. Targets are independent simulator-continuation proportions,
not material labels or optimal-play outcomes. Keep the documented simulator rules,
including `research_rules_v2_royal_conversion` for capture conversion.

Use full terminal demonstration games and the complete query state/history.
Inputs contain no variant names or rule descriptions. Preserve exact K, policy,
state, source-family separation and query-last serialization. Never trim a game
or query to make a context fit. An unfinished continuation remains unresolved;
it cannot silently become a draw or be omitted from a renormalized target.

The old-checkpoint K2/K4 evaluation extension is stopped. It launched no inference
workers and used zero GPU-hours. Its model-blind feasibility work is retained for
the later evaluator. No new test-model scores are opened during training design.

## 2. Data and launch requirements

The first larger dataset has a frozen design, independently seeded with
`2026112501`, for 13 project variants plus a separate standard-chess control:

| Per rule/control | Train | Validation | Test |
|---|---:|---:|---:|
| Query-source families requested | 256 | 64 | 64 |
| Maximum query roots, two adjacent states per family | 512 | 128 | 128 |
| Independent demonstration families requested | 128 | 64 | 64 |

Across all 14 rule/control settings, maxima before admission losses are 7,168
training, 1,792 validation and 1,792 test roots: 10,752 roots and 344,064 R32
continuations. Demonstrations are additional games. These are quotas/upper bounds,
not achieved counts. The separate 77-root optimized benchmark is not the main
training corpus. At plan preparation the main corpus was starting generation.

Keep R=32, initial administrative budget 2,560 plies, and same-stream extensions
when necessary. Preserve null targets/bounds for any remaining censoring. Continue
the conservative replay, seed, family, exact-duplicate and radius-two geometry
audits, including the declared separation from prior pilot query/demo inputs.

Important: the primary four-rule training pool has at most 2,048 new variant
roots, plus at most 512 standard warmup roots. That is still data-limited for a
494M network. Fresh demo combinations do not create new independent query labels.

Therefore continue generating additional **training-only** waves. The next scale
target is 10,000 accepted training roots per rule/control, with many more source
families, subject to measured generation cost. Keep the new validation/test pools
fixed. This expansion is a target, not a prerequisite to profiling or a promise
of available data. Freeze a new snapshot before each added-data training phase;
never consume a mutable live index inside a run.

Training launch requires an immutable checksum-bound snapshot, complete labels,
all intended training rules represented with a balanced sampling pool, full
demonstrations adequate for the declared K mixture, and an audited fixed fresh
validation cohort. Early balanced train prefixes may support profiling or a
separately logged phase; they must not be presented as the entire final corpus.
Prioritize fresh validation labels alongside training labels so validation-only
selection is available. Keep all new test-model scoring unopened.

## 3. Models

### Qwen track

- Pinned Qwen2.5-7B-Instruct revision
  `a09a35458c702b33eeacc393d103063234e8bc28`.
- Preserve the pilot's first-stage adapter architecture: rank 8, alpha 16, q/k/v/o
  projections in the last four layers, plus the FP32 three-way head.
- The base weights remain frozen; 731,651 adapter/head parameters train.
- Use fresh adapters/head for the new study, followed by a fresh standard-only
  warmup. Preserve the old four-rule pilot checkpoints as historical results.
- Expanding adapter capacity is a separate later condition, not an undocumented
  change while claiming a training-duration comparison.

### Scratch track

Use a randomly initialized HF Qwen2-style causal Transformer and three-way head:

| Setting | Value |
|---|---:|
| Transformer layers | 24 |
| Hidden dimension | 896 |
| Feed-forward intermediate dimension | 4,864 |
| Attention heads / KV heads | 14 / 2 |
| Embedding capacity | 152,064 |
| Native maximum input length | 32,768 tokens |
| Total trainable parameters, including head | **494,150,147** |
| Embedding parameters | 136,249,344 |
| Other parameters | 357,900,803 |

All network parameters train from random initialization. Use the same pinned
Qwen tokenizer and full serialized input as the Qwen track. The tokenizer is
shared; the scratch network does not receive pretrained model weights. Do not
silently switch to the short chess-specific vocabulary in the primary comparison.

Use RMSNorm, RoPE, SwiGLU, grouped-query attention, native causal SDPA and
activation checkpointing. Do not scale the old tiny implementation, which builds
dense quadratic masks. Use FP32 trainable/master weights and Adam moments with
BF16 autocast for scratch; retain the frozen BF16 Qwen backbone and FP32 trainable
adapter/head handling. Verify actual attention/memory behavior in the profile.

The small scratch pilot is retained as history; no further tiny-model training is
part of this study.

## 4. Training design and budgets

Use seeds 42, 123 and 456 for each primary architecture. For each architecture/seed,
train one standard-only warmup parent for **1,000 optimizer updates** and select
its weights using standard-only validation every 100 updates. Clone that exact parent into the
familiar-rule and applicable all-rule branches. Main training starts a documented
new optimizer/scheduler phase; this is distinct from resuming an interrupted run.

Each main run has an initial **10,000 optimizer-update** budget:

- Microbatch one episode, accumulate eight episodes per update: effective batch 8.
- 80,000 main episode presentations per run, including repetitions: 50 times the
  old main-update budget and 100 times its episode presentations.
- Uniform K mixture `{0,1,2,4}` and equal rule weighting within the run's exposed
  rule set; sample source families/roots independently of target values.
- Deterministically resample full training-only demonstrations throughout training.
  Use a dedicated sampler RNG/counter independent of model-initialization RNG.
- Match primary Qwen/scratch sampling and inputs by seed where applicable, and
  log input hashes, tokens processed and unique roots/families actually encountered.
- Query-only soft W/D/L cross entropy; no trajectory language-model auxiliary loss
  in the primary condition. Any such pretraining is a separate named experiment.
- AdamW betas `(0.9,0.95)`, norm clipping 1; exclude norms/biases from weight decay.
- Starting learning rates: Qwen adapters/head `1e-4`; scratch `3e-4`.
- Weight decay: Qwen `0.01`; scratch `0.1`.
- Linear warmup for 5% of each declared phase, then cosine decay.

For the initial phases, cosine decay ends at 10% of the peak learning rate;
the main optimizer is new after the selected warmup initialization. These settings
are fixed before new-model validation results. A later extension's exact schedule
is recorded in its new phase configuration before launch.

Save main-step 0, 200, 1,000, 2,500, 5,000 and 10,000 weights for within-run duration
curves. All points in that curve share the declared 10,000-step LR schedule. They
do not recreate a separately trained 200-step schedule. Comparing this new study
against the pilot also changes data and the K mixture; do not attribute all
differences solely to additional updates.

At the 10,000-step milestone, use only training/validation curves to decide whether
to extend toward **50,000 updates**. Record an extension as a new phase with its
initial checkpoint, data snapshot, optimizer continuation/reset convention, LR
schedule and maximum budget frozen before it begins. Never use test scores to
choose duration, and do not promise improvement from more updates alone.

## 5. Eight-GPU allocation

Target eight independent jobs, one per available H100 80GB:

| Logical slots | Jobs |
|---|---|
| 0–2 | Qwen familiar-rule models, seeds 42/123/456 |
| 3–5 | Scratch familiar-rule models, seeds 42/123/456 |
| 6 | Qwen all-rule reference, seed 42 |
| 7 | Scratch all-rule reference, seed 42 |

These are logical slots, not permission to replace scheduler visibility or stop
unrelated processes. Fill available devices and queue other jobs until capacity
is available. Shared warmup parents are prepared first and reused; a reference
can start once its matching seed-42 parent is ready. No separate full warmup is
charged again for a shared parent.

The primary models train on diagonal_rook, straight_bishop, weakened_queen and
extended_pawn. The other nine variant rules receive no gradients in those models.
References receive all 13 variant rules; standard remains the warmup/control.
The one-seed references are descriptive exposure comparisons, not three-seed
causal estimates or held-out-transfer results. Further reference seeds can form
a later wave using freed slots, with their budget recorded before launch.

Independent runs are the initial allocation because each model should fit one
80GB GPU. Use distributed training only if profiling establishes a need; avoid
adding communication overhead without evidence of a benefit.

## 6. Preflight and automation

Before launching long runs:

1. Verify scratch random initialization, exact parameter count and all-parameter
   gradients; verify Qwen's frozen/trainable parameter split and exposure lineage.
2. Test loss, padding/attention behavior, complete-state input reconstruction,
   overflow handling and training-only dynamic context sampling.
3. Run save/resume equivalence tests across an optimizer boundary, including the
   next sampled episode and update, not only a weights reload.
4. Profile 50–100 actual mixed-K updates per architecture, including a longest
   supported K4 training input. Discard profile weights from scientific runs.
5. Record tokens/second, peak allocated/reserved memory, optimizer-update time,
   model-loading time and validation overhead. Set GPU-hour estimates from those
   measurements, not from the old short K1 pilot.

An automated launcher assigns devices, creates immutable run configs/manifests,
starts workers, writes heartbeats and queues the remaining jobs. Monitor summaries
and failure alerts; do not manually issue each update or evaluation. Retry only
recoverable operational failures from valid checkpoints. Record OOM/configuration
changes explicitly; never solve OOM by silently truncating input or changing K.

Agents own data production, Qwen/training orchestration, and the scratch path.
The parent reviews the shared interfaces, resource allocation, QA and scientific
comparisons, and lightly monitors milestones rather than driving every run.

## 7. Checkpoints, validation and failure recovery

Validate every **500 main optimizer updates**, using a fixed validation-context
panel and canonical unpadded inference. A fixed early panel may use the first 16
retained validation family hashes per rule, all retained siblings, selected before
labels/model scores. Use it for online learning curves. Before final test access,
score the saved 500-update candidate checkpoints on the completed fresh validation
fold and select by its macro NLL over the four familiar rules and balanced
K0/1/2/4. Standard warmup uses its fixed standard validation panel; all-rule
references use their declared exposed-rule validation set. Record both panel and
full-fold results without silently combining them into one learning curve.
Do not use held-out-rule validation performance to select primary model weights.

Save an atomic resumable checkpoint every **250 updates** and before a controlled
shutdown. Save at completed optimizer boundaries; an interrupted partial
accumulation restarts from the prior complete checkpoint. Full state includes:

- Trainable/model weights, optimizer moments and scheduler state.
- Python, NumPy and Torch CPU/CUDA RNG state.
- Dedicated rule/family/root/K/demo sampler state and draw counters.
- Optimizer step, accumulated token/episode counters, phase and exposure lineage.
- Exact model/tokenizer/config/data-snapshot/shard hashes.

Keep two rolling full-state checkpoints, best validation weights, and named
milestone weights. Retain weights at every 500-update validation candidate for
the declared full-validation selection; do not retain every Adam state indefinitely.
Check disk estimates before launch: a scratch full-state checkpoint is several
GB, unlike the small LoRA checkpoint. Reserve approximately 300GB for the initial
matrix's checkpoints, separately from raw data/base weights, and re-estimate from
actual serialized files before launch or an extension.

The main corpus generation continues uninterrupted. If producing its early
validation panel requires a separate job, that job uses the exact frozen root/
replicate seeds. Compare duplicate panel/full-corpus labels exactly and count
them once; they are not extra independent continuations. This scheduling
alternative preserves the running generator's design and completed work.

## 8. Frozen evaluation after training

Freeze selected checkpoint hashes, data/protocol hashes, all K, matching rules,
metric definitions and exclusions before opening new test-model comparisons.
Evaluate both architectures at **K0/1/2/4**, using five draws for each nonzero K:

- Relevant full games.
- Foreign-rule full games matched to demonstration token length within 10%.
- Corrupted outcomes and shuffled-trajectory sensitivity controls.
- Seven declared non-neural baselines fitted only on allowed training data.

Use canonical unpadded single-episode predictions and token-ID/model-bound caches;
verify frozen parameters, input hashes and uncached repeat predictions. Report
both each K's feasible cohort and the common cohort across all K/controls, plus
all-root relevant-context coverage where feasible. A bounded matching search
failure is a search failure, not proof that no matched context exists.

Report familiar-rule adaptation, nine-rule gradient-held-out transfer, and
standard separately. Never describe exposed-reference results as transfer.
Report macro NLL, vector/categorical Brier, paired gains and relevance comparisons,
per-rule/side/equal-material slices and calibration with sample counts.

Use the declared seed/draw/family bootstrap and report its assumptions and small
seed-count limits. Keep Monte Carlo label uncertainty separate. Independently
re-estimate a preselected new-test subset at R256 using an outcome-independent
family-hash rule fixed before model scores. Do not reuse the pilot's opened test
as an unseen confirmatory population.

## 9. Research record and completion criteria

Commit and push completed stages to `main`, coordinating the shared Git index
between agents. Keep large datasets/checkpoints outside Git; publish compact
immutable manifests, checksums, configs, decisions, learning curves and results.
Retain failed runs and old pilot findings with explicit validity labels.

Report completed versus selected update counts, unique labels versus repeated
episodes, actual tokens, wall times and summed per-GPU process hours. Do not call
wall time utilization-integrated GPU compute or worker elapsed time CPU core time.

Completion means the planned runs actually train, validation selects weights,
the frozen K0/1/2/4 matrix and baselines finish, numerical/data audits pass, and
all positive, negative and inconclusive results are documented and pushed.
Success is measured experimental evidence; improved scores are a hypothesis,
not a promised outcome. This study compares practical pretrained-adapter and
smaller scratch systems; it does not isolate pretraining alone from capacity and
trainable-parameter differences, prove optimal play, or establish a scaling law.
