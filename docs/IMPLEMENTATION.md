# Implementation and artifact contracts

## Module map

| Module | Responsibility and invariant |
|---|---|
| rules.py | Registry of the 13 frozen variants plus standard; no unknown-name fallback; four familiar vs nine held-out candidate rules |
| simulator.py | `VariantBoard(chess.Board)` and twelve custom subclasses; shared attack/movement definitions, legal filtering after transitions, correct special-move bookkeeping; antichess uses the pinned reference class |
| state.py | Versioned replay state, JSON round trip, full-state SHA256, public rule-blind state, geometric duplicate distance, valid perspective transforms |
| policy.py | Sorted legal-action sampling and immediate-win policy, no material evaluation |
| rollout.py | Independent reproducible continuation jobs, multiprocessing spawn, terminal status/reason, traces/timing, empirical counts/targets/bounds |
| data.py | Three disjoint provenance pools, hashed source-family folds, adjacent sides, near-duplicate exclusions, replay and label-integrity checks, cost summaries |
| episodes.py | Deterministic full-game context selection, K/control construction, strict overflow errors, query-last guarantee, length-matched mismatched controls |
| tokenizer.py | Scratch-only vocabulary: 64 explicit squares per state, side/castling/EP/clocks/barriers, from/to/promotion moves, White result; no rule token |
| models.py | Tiny causal RoPE model; Qwen BF16 backbone plus last-four-layer LoRA and FP32 three-way head; query-final pooling, soft CE and frozen prediction interface |
| train.py | Seeded training/validation only; AdamW, clipping, warmup/cosine LR, gradient accumulation, best macro validation NLL checkpoint, trainable-only serialization and warmup lineage |
| evaluate.py | All requested K/conditions/draws on a common root cohort; frozen deterministic inference, familiar/held-out labeling, side/equal-material scores, paired gains and intervals |
| metrics.py | Probability validation; soft NLL, vector and categorical Brier, L1, argmax/ties/class counts, macro aggregation, paired and hierarchical uncertainty |
| baselines.py | Training frequency/majority, material multinomial regression, length/outcome histogram regression, demo prior/majority and demo-state kNN |
| diagnostics.py | Same-stream horizon extension, actual tokenizer counts, reference-only SAN comparison and Jeffreys multinomial uncertainty |
| protocol.py | Necessary main-run gates; deliberately false until actual evidence is supplied |
| cli.py | Audit/generate/extend/train/evaluate/tokens/gates/seal entry points; rejects main runs without gates and warmup lineage |

Scripts provide baseline evaluation on the exact model cohort, multi-seed
aggregation, a same-root development policy probe and actual GPU profiling.
Cluster wrappers inherit the scheduler environment and request one GPU per run.

## Dataset JSON (`chess_icl_v2.1`)

A dataset includes `provenance`, `demonstrations`, `source_games`, `queries`,
`exclusions`, timing, and `integrity`. Provenance stores config, config hash,
Python/platform/package versions, UTC timestamp and hashes of Python source files.
All generated artifacts retain exact seeds. JSON uses finite numbers only and is
written by atomic replace.

Each game carries a family ID, fold, pool, rule ID, initial state, policy/seed,
every UCI move, final FEN, status, true White result or null, termination reason,
ply budget and wall time. A source game may be administratively unfinished: its
past states remain legitimate roots. A demonstration must be terminal and its
entire trajectory must be included.

Each query carries a private ID/family/fold/rule, replay state and label record.
Labels store R, counts `[W,D,L]`, unresolved count, target or null, identification
bounds, plug-in Monte Carlo SE and every independently seeded continuation trace.
Counter/clock/repetition metadata is never discarded to save tokens.

Full-state hash equality is stricter than the conservative position/near-duplicate
filter. Deduplication does not mean that every shared opening prefix is removed;
that would make full chess-game demonstrations impossible. The audit reports its
precise comparison scope. Different seeds alone do not prove independent game
content; conservative query-vs-demo and cross-fold-root checks are also applied.

## Rule-blind episode

```text
<DEMO>
{"initial": {"initial_fen": "...", "history_uci": [], "current_fen": "...", "extras": {}},
 "moves_uci": ["e2e4", "..."], "white_outcome": "W"}
</DEMO>
<QUERY>
{"initial_fen": "...", "history_uci": ["..."], "current_fen": "...", "extras": {}}
</QUERY>
<PREDICT_WDL>
```

The actual serializer uses compact canonical JSON. Private metadata stays outside
the token stream. The model head reads the last real token, with right-padding
handled explicitly. Targets never appear after the query, and no future rollout
appears anywhere in the input. Demonstrations are from the query fold and policy,
with independent source families. Held-out-rule demonstrations are allowed at
frozen evaluation; held-out-rule gradient training is not allowed in a transfer
condition.

Overflow rejects the requested episode; it cannot trim a demo, trim the query,
silently change K, or drop query metadata. Exclusions are persisted with reasons.

## Targets, loss and metrics

For complete labels, `y=(nW,nD,nL)/R`; training loss is
`-sum_c y_c log softmax(logits)_c` at the query only. If any continuation is
unfinished, `target_wdl=null`. For each class, `[n_c/R,(n_c+U)/R]` bounds the eventual
empirical proportion. These are identification intervals for unresolved samples,
not confidence intervals for the underlying policy probability.

NLL uses the same soft cross entropy, with clipping at 1e-12 only for reported
baseline probabilities. Vector Brier is `sum_c(p_c-y_c)^2`; expected categorical
Brier is `1+sum_c p_c^2-2 sum_c p_c y_c`. Their difference is target uncertainty.
Metrics average within rule, then equally across rules. Argmax ties choose the
first class for accuracy and are counted explicitly; accuracy is secondary.

MC plug-in SE is `sqrt(y_c(1-y_c)/R)` and may misleadingly vanish at 0 or 1. The
Dirichlet(1/2,1/2,1/2) posterior utility provides marginal 95% intervals. Main test
work must also create independently seeded high-R labels; low-R pilot metrics do
not substitute for them.

## Baseline details

All learned baselines fit training-fold roots and exposed rules only. Material
regression uses normalized counts for six piece types per side plus an intercept;
softmax linear weights fit soft labels for 500 steps, LR .1 and L2 .001. Histogram
regression uses side, K, mean/std demo length and White outcome proportions plus
side interactions. Training examples include K=0 and up to four K=1 demo draws;
using it at larger K is an extrapolation to report, not a tuned optimal baseline.
Demo majority uses tied-mode votes, then Laplace smoothing; frequencies add one
count per class. kNN uses square Hamming distance over same-side states in the
selected full demo games, five nearest states, White-to-query perspective
conversion and one pseudocount per class. Multiple neighbors may come from one
demo family; a one-neighbor-per-game variant is a separately named future ablation.

A directly all-rule-trained reference uses the same architecture, labels and
warmup, but intentionally exposes every rule to gradients. Its held-out results
are an exposure reference and cannot be called transfer. Larger μP models,
compositional variants, shallow search policies and segment-bidirectional attention
are outside the user's immediate fixed-policy core and are not reported as done.

When full demonstrations are extended beyond the original budget, their newly
visible suffixes are checked against every fixed query root. A conflicting demo
is excluded as a whole and preserved in `excluded_demonstrations`, with its
family/query conflict logged. Query selection and label streams remain fixed.
Insufficient remaining demos cause explicit infeasibility, not partial games.

The wheel includes its versioned Fairy-Stockfish `reference.ini`; an installed
(non-editable) package can run the same audit without relying on the current
working directory. Main-run outcome-variation checks use development folds only;
the nine-rule held-out transfer set never includes the standard warmup control.
Lineage records the union of all rules exposed to gradients in parent checkpoints.

The final hierarchical bootstrap preaggregates gain sums and root counts by
seed/rule/demo-draw/source-family, then resamples those families with replacement.
A sampled family's numerator and denominator both travel together, preserving the
root-weighted mean when a source family retains only one sibling. Seeds and demo
draws are resampled at their declared levels; rules remain equally weighted.
This avoids repeated full-table scans without changing the estimand.
