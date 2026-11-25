# Fresh larger fixed-policy corpus

This is a new independently seeded corpus, not an enlargement of the previously
opened pilot test. The original614-root data, checkpoints, evaluations, protocols,
snapshots and research bundle remain unchanged. Main configuration:
[`v2_large_dataset_v1.json`](../configs/v2_large_dataset_v1.json).

## Frozen size and semantics

| Per rule, including separate standard control | Train | Validation | Future test | Total |
|---|---:|---:|---:|---:|
| Independent query-source families | 256 | 64 | 64 | 384 |
| Maximum query roots, adjacent opposite-side pairs | 512 | 128 | 128 | 768 |
| Independent demonstration candidates | 128 | 64 | 64 | 256 |

Across13 variants plus standard this requests5,376 independent source families,
3,584 demonstration candidates and at most10,752 query roots. AtR32 that is at
most344,064 independent label continuations. Train/validation/test root upper
bounds are7,168/1,792/1,792. These are requested quotas and upper bounds, not
accepted counts. Source-family count is16× the pilot's24 families per rule;
near-duplicate and short-source exclusions can reduce labeled-root counts.

The source split quotas are2/3,1/6,1/6. Stable hash eligibility still uses the
existing70/15/15 buckets; eligible candidates are enumerated until the explicit
fixed quotas are filled. Demonstration quotas are1/2,1/4,1/4. This follows the
pilot's stable family-seed assignment while making the new corpus train-heavy.

Policy remains `terminal_aware_v1`: uniform among legal immediate wins, otherwise
uniform among all sorted legal actions. Rules remain `research_rules_v1` except
corrected capture conversion, `research_rules_v2_royal_conversion`. Master
seed2026112501 is distinct from the pilot, development benchmark and high-R seeds.
Independent demonstration, source and label streams retain their existing
domain-separated hashes. Full initial FEN, all moves, counters and barriers stay
in every root state. Labels remain player-to-move empirical W/D/L; neither
material nor engine-optimal adjudication enters a target.

## Decisions LD01–LD14

LD01. Increase independent source families, not just correlated roots orR. Keep
one uniformly chosen adjacent root pair per source game, selected from prefixes
after ply8 and before its stopping state. Root selection uses source history and
seed only, before any independent root outcomes are sampled.

LD02. Use fixed quotas above. Reject geometry conflicts without replacing their
source families to obtain a favorable outcome mix. Report accepted roots/families
and exclusions by rule, fold and reason; never describe the upper bound as actual
completed data. Standard is a control, not a fourteenth project variant.

LD03. Keep exact radius2 geometry checks: all usable new-demo states versus all
roots, cross-fold query roots, and globally unique position signatures. Also
exclude every new root near any old pilot root or complete demonstration state,
regardless of its old fold. This protects the future test from previously opened
pilot inputs and prevents new training roots from recycling old test positions.
It does not remove every shared opening prefix from full games.

LD04. Build all demonstration/source buckets and freeze the entire admitted root
cohort before labeling or publishing a training subset. Label scheduling then
orders train, validation, test and round-robins rules within each fold. Scheduling
does not change deterministic admission order, root selection or policy streams.
Consequently an early fixed training-prefix manifest cannot change the later
validation/test near-duplicate decisions.

LD05. UseR32 and an initial administrative budget2,560 plies. For unfinished label
continuations and demonstration games, replay the same seeded stream to5,120 and
then10,240 plies, verifying every previous prefix and retaining its digest/timing.
No seed is replaced. Any remaining unfinished sample stays unresolved; a root
with any unresolved replicate has a null target and identification bounds. An
unfinished demonstration is retained as an excluded candidate and never becomes
a draw or partial public demo. Source games keep their initial2,560-ply budget;
their nonterminal past prefixes remain legitimate roots.

LD06. Reconstruct a root once and clone its entire board history for each label
continuation. This avoids redundant repeated history replay, while preserving
repetition, clocks, castling, en passant and barriers. It does not clone an RNG
stream: each replicate still has its distinct indexed seed. The all-rule
benchmark compares every count, probability, seed, move, final FEN, result and
termination against the reference implementation, excluding timing fields.

LD07. Store full traces in atomic gzip-JSONL shards,32 roots per label shard.
Fixed shard order follows the frozen cohort; a bounded window keeps only a small
number of pending/completed root records in memory. Do not build one multi-GB
Python object. Compressing JSON is lossless. Gzip header timestamps/filenames are
fixed; trajectory timing fields themselves remain measured and non-deterministic.

LD08. Resume only from checksum-verified shards with the same config, rule
versions, critical-source hash and historical-input hashes. Completed generation
manifests preserve their original timing; concurrent CLI writers are excluded by
an advisory lock. Temporary/incomplete files are never treated as completed
shards. No edits to a frozen generator are silently mixed into this version.

LD09. Before publishing each label shard, replay every continuation move and
terminal boundary, verify FEN/results/perspective/counts/bounds/SE, regenerate
replicate0 under the documented policy, and check seeds against the oldR32/R256
streams and every prior shard. Every game is also replay-audited at generation.
A final whole-corpus audit repeats these and all global geometry/provenance
checks, including the exact seeded source-pair selection.

LD10. An immutable partial manifest identifies only its completed, audited prefix
of an already fixed cohort. It explicitly marks the final full-corpus audit as
pending. Future training must bind one such manifest and filter both query and
demo folds/rules; it must not watch a live index that grows while training.
Partial snapshots permit early training availability without calling unfinished
generation a complete dataset. The final manifest identifies every shard.

LD11. Keep the new test model-score fold unopened. No predictions, checkpoint
selection or training are run by this generator. Compact summaries disclose test
availability/unresolved counts but omit test W/D/L class totals. Raw test labels
necessarily exist for storage/audit; this is procedural isolation, not encryption.

LD12. Begin with32 benchmark workers and use48 for the main CPU-only generation
after measured feasibility, leaving capacity on the192-logical-CPU shared host.
No GPU is required and no unrelated process is stopped. Wall/worker elapsed
times include sharing/scheduling and are not measurements of CPU core time.

LD13. The larger corpus is a foundation for longer Qwen and~500M scratch work;
approximately10k possible unique roots is not evidence of adequate independent
supervision for a500M model. More epochs or demonstration combinations do not add
independent query labels. A later train-only wave can hold demonstrations and
validation/test roots fixed, generate fresh source families, and reject new train
roots against those fixed pools; adding new demos would require rechecking old
root admissibility. No scratch-model adequacy or scaling result is inferred here.

LD14. Keep code/configs, compact completed benchmarks and final manifests/audits
in Git, while raw traces/shards remain in versioned local deliverables. Keep
existing published snapshots immutable and commit completed stages to main.

## Measured benchmark and present status

Separate benchmark seed2026112500 requests one demo/source family per fold/rule.
It retained77 roots from42 source families:2,464 R32 continuations, zero unresolved,
integrity passed. Full generation/games/selection/audit measured83.32s with32
workers. Its raw JSON was11MB. The optimized same-root label benchmark measured
58.29s including worker startup and exactly reproduced every continuation.
These wall times have different scopes and are not a direct speedup ratio.
The optimized sum of label-worker elapsed times including root replay was817.04s.
The filesystem had approximately2TB free at launch preparation.

The main cohort's accepted counts and actual generation cost are published only
after its corresponding phase completes. Initial cost extrapolation is roughly
one to three hours for full generation and audit at48 workers, subject to current
load, admission losses and unusually long histories. This is an estimate, not a
reported run result. No main model training is launched by this task.

## Paths, streaming and reproduction

Raw output root is `results/large_dataset_v1/corpus/`:

- `design.json`: frozen config, rule versions, source/input hashes and provenance.
- `games/*.jsonl.gz`: complete fixed-quota games and replay-audit markers.
- `roots.jsonl.gz`, `cohort.json`, `exclusions.jsonl.gz`: label-independent fixed
  cohort and admission record.
- `labels/part-*.jsonl.gz`: full query/state andR32 continuation records.
- `snapshots/labels-prefix-*.json`: immutable audited partial manifests.
- `manifest.json`, `audit.json`: final complete data and full-corpus audit.

```bash
python scripts/v2_large_dataset.py generate \
  --config configs/v2_large_dataset_v1.json \
  --output results/large_dataset_v1/corpus
python scripts/v2_large_dataset.py export \
  --output results/large_dataset_v1/corpus \
  --destination results/large_dataset_v1/compatibility.dataset.json
```

Generation automatically runs the final audit. `--skip-audit` is explicitly an
incomplete validation state. The compatibility export streams the legacy JSON
schema, including excluded unfinished demos; it requires the completed audit.
Existing training can read that export. A memory-efficient new training loader
should consume a fixed manifest and gzip-JSONL rows directly:

```python
from chess_icl_v2.large_data import iter_snapshot_queries
for query in iter_snapshot_queries(fixed_prefix_manifest, split="train"):
    ...  # additionally filter exposed rules and null targets as declared
```

All demonstrated games stay complete. No token-length/outcome filter is introduced
to force largerK feasibility; future training must measure actual tokenizer lengths
and reject infeasible episodes without removing the query or silently changingK.
