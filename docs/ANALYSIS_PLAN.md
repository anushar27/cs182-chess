# Analysis plan before confirmatory experiments

This is a proposed confirmatory protocol, not a record of completed experiments.
Freeze a version/hash after development feasibility and H100 profiling, before
opening an independently generated final test. Save changes to decisions with
reasons; never silently rewrite the protocol after inspecting final-test metrics.

## Questions and comparisons

1. Familiar-rule adaptation: on rules exposed during training, does relevant
context improve macro NLL over K=0 on identical roots?
2. Held-out-rule transfer: repeat with all gradients, checkpoint selection and
policy selection excluding the nine held-out rules. Qwen pretraining exposure is
unknown; describe it as transfer with that limitation.
3. Rule relevance: compare relevant vs mismatched context with identical roots,
K, model seed and within-10% demo-token length. Report distributions and exclusions.
4. Shortcuts: compare kNN, training frequencies/majority, demo frequencies/majority,
material regression and length/outcome histograms. Report the abs(material)≤2
slice without using material in labels.
5. Interventions: flip demo outcome tokens and shuffle demo trajectories. Interpret
invalid-trajectory interventions cautiously. Report setup-only Knight Army and
explicitly visible blocked-square geometry separately.
6. Exposure reference: train a separate all-rule condition after common warmup;
never interpret it as held-out transfer.

Candidate familiar rules: diagonal_rook, straight_bishop, weakened_queen,
extended_pawn. Candidate held-outs: short_knight, mobile_king, piece_capture,
forward_pawn_capture, dark_square_immunity, blocked_squares,
extended_initial_pawn, knight_army, antichess. Standard chess is warmup/reference,
not one of the thirteen. The input includes no rule labels, but observable states
and histories can identify rules; the paper must acknowledge this.

## Selection and gates

Use development data to choose a single fixed policy, budget-extension convention,
root distribution, K set and token budget. Do not mix policies in a target dataset.
Current starting candidates are uniform_legal_v1 and terminal_aware_v1. Main data
must have no unresolved targets at the chosen administrative budget and enough
complete demonstrations to meet every requested K. Retain counts for failed
budgets rather than recoding them as draws.

The included gate checks all adapters, audit results, split integrity, both sides
per fold, R≥32, positive pilot performance relative to kNN/mismatched context and
measured H100 operation. A positive pilot on at least two familiar rules, enough
independent source families, useful outcome variability, stable baselines and an
affordable measured CPU/GPU budget are additional review requirements. A scalar
passed flag is not scientific evidence: preserve the supporting result artifacts.

Do not choose a policy by maximizing pilot ICL gain across a large undocumented
search. Record all policies/roots tried. The current same-root terminal-aware
probe uses only the development train fold, R=2 and 320 plies; it informs
feasibility and is too small to establish model performance.

## Training and checkpoint selection

Three model seeds (42,123,456); same standard-chess warmup parent within each seed
across conditions. Main Qwen config: immutable base revision, BF16 frozen weights,
LoRA r8/alpha16 last-four q/k/v/o projections, FP32 probability head. AdamW
betas(.9,.95), weight decay .01, LR1e-4, linear warmup100 updates/cosine decay,
global norm clip1, microbatch1 and accumulation16. These are proposed starting
settings, not tuned results. Reprofile full context before committing run count.

Select checkpoint solely by macro validation NLL, averaged over exposed variants
and the declared K mixture. Validation families are independent of training.
No final-test-based early stopping or new hyperparameters. Log exclusions; main
strict-coverage mode rejects any missing K or unresolved target. Standard warmup
needs its own adequate standard dataset; the tiny feasibility sample is not one.

## Evaluation and uncertainty

Save per-root probabilities, targets/counts/R, side, family, K, condition, demo
family IDs/token lengths, draw index, exposure category, seed and checkpoint hash.
Five independently sampled demo draws for each K>0, with K=0 computed once. Use a
common root cohort for all requested K/controls/draws; report roots excluded and
why, and report feasibility by variant and K. Do not present favorable complete
cases as the original population.

Primary metric: macro paired NLL(K=0)-NLL(relevant K). Also report relevant minus
mismatched comparisons and best baseline on the same roots. Provide soft-target
NLL, vector/categorical Brier, high-R probability errors, calibration by side,
class counts, secondary argmax accuracy and ties. No 33.3%-chance headline.

Resample model seeds, demo draws, then query-source families with replacement,
stratifying by rule; macro-average rules. Single-seed pilot intervals omit
between-seed uncertainty and must say so. State the bootstrap seed/count. The
bootstrap does not include Monte Carlo label noise; report it separately using
counts, multinomial intervals, and independent high-R re-estimation (proposed
R=256 or more, decided from noise/cost measurements).

No confirmatory p-value or multiple-testing correction is chosen here. Report
intervals and treat the full K/control matrix transparently. If hypothesis tests
are added, preregister their family and correction before final test access.

## Claim boundaries

These probabilities describe both players following the declared policy in the
frozen simulator and draw-claim convention. They do not measure optimal chess,
engine strength or human win probability. A draw-dominant dataset may make a
competent model look accurate while providing no ICL signal. No context gain is
a valid negative pilot result, not permission to relabel outcomes or inspect the
sealed final test. The local scratch pilot has no pretraining contamination, but
it is not evidence for successful held-out rule transfer without the actual
controlled result.
