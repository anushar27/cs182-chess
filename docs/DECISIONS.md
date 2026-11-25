# Decisions and provenance ledger

Scope: the user's current request takes precedence over proposals/instructions in
the attachment. The attachment is planning evidence, not authorization to report
old results as new or to launch an ungated full experiment. Dates in this report
use the client's America/Los_Angeles calendar, November 24, 2026; machine logs use UTC.

| ID | Choice | Reason and consequence |
|---|---|---|
| D01 | Target order W/D/L, root-player perspective | Labels are outcomes of a fixed policy in the exact simulator, not optimal values or strength estimates. Demo outcomes remain White-perspective |
| D02 | Separate new `chess_icl_v2` package | Preserve all old experiments and their artifacts for provenance. Legacy material-label code remains legacy and is never imported by v2 |
| D03 | Two policies, never mixed within a dataset | Uniform legal and immediate-terminal-win-else-uniform. Sorted UCI legal moves and per-continuation hash-derived seeds make replay reproducible across worker schedules |
| D04 | Administrative horizon means unresolved | Retain R, W/D/L counts and unresolved count U. Identification bounds for each probability are [count/R,(count+U)/R]. Upper endpoints are marginal and cannot all hold jointly. Never renormalize completed outcomes |
| D05 | No training target unless U=0 | Pilot mode can explicitly log and skip incomplete roots; main training uses strict coverage and must not do complete-case selection. Extend the same streams/roots before main training. Existing complete labels are retained, not resampled |
| D06 | Optional draw claims disabled | No claim action is part of either declared policy. Fivefold/75-move draws are genuine terminal draws; the 320-ply budget is not |
| D07 | Full replay state, including history, is visible | Needed to preserve repetition and future outcome distributions. This makes K=0 partially rule-informative; acknowledge this in the paper |
| D08 | Independent demonstration, source-game and continuation streams | Source families are assigned before outcomes, 70/15/15 by stable hash. Label replicates are never demonstrations. Siblings/adjacent roots remain in the same fold |
| D09 | Both adjacent sides sampled per source | One outcome-independent adjacent root pair from eligible prefixes (at least 8 plies, before final termination). Near-duplicate removal can remove one side; a main gate checks side coverage rather than pretending it is guaranteed |
| D10 | Conservative near-duplicate exclusion | Across all demonstration states vs query roots, and query roots across folds: same side, ≤2 differing occupied/piece squares. Also reject exact query position signatures (pieces, side, castling, EP), even if counters/history differ. Different barrier maps may still be excluded conservatively. Shared opening prefixes between otherwise distinct games are not globally forbidden |
| D11 | All requested K must fit in full | Serialization ends with QUERY and PREDICT_WDL. Overflow raises a structured exclusion; never truncate query or silently lower K. Training default strict coverage prevents hidden K/length selection. Evaluation uses one common root cohort for all requested K/conditions/draws |
| D12 | UCI for the versioned main input | Unambiguous and available for every custom movement rule. SAN is measured only where the reference library supplies correct notation. No claim of a global UCI-vs-SAN tokenizer optimum |
| D13 | Qwen2.5-7B-Instruct pinned to a09a35458c702b33eeacc393d103063234e8bc28 | Downloaded tokenizer/config only locally. Configuration's native context must be read, not inferred from model family naming |
| D14 | LoRA r=8, alpha=16 on q/k/v/o in last 4 blocks; FP32 3-way head | Frozen BF16 base, causal attention, query-only soft cross entropy. No material labels, centipawns, dense demo loss, custom Qwen embeddings, or generated text labels. Rank16/alpha32 is a future preregistered sweep, not a completed result |
| D15 | Trainable-only checkpoints plus pinned base revision | Avoid saving frozen 7B weights; test exact reload and inference without updates. Checkpoints initialize later warmup branches; mid-run optimizer/RNG resume is not implemented |
| D16 | Common standard warmup before each main condition | Main launch requires an explicit parent checkpoint. Parent architecture and hash are logged; researchers must use the same warmup parent across matched conditions for a seed |
| D17 | Small RoPE causal Transformer as pipeline pilot | Fixed 64-square piece tokens, state metadata, from/to move tokens, outcome tokens. It is standard parameterization, not a validated μP scaling family. No scaling-law or contamination-free transfer result is claimed until a real multi-seed training/evaluation study is run |
| D18 | Mismatch matched in K and demo token length within 10% | Search 64 deterministic candidates; fail if no match. Match demo length after subtracting query, so long queries cannot hide bad controls. Not a match on outcome histogram; the histogram baseline measures that shortcut |
| D19 | Corruption controls are explicit interventions | Outcomes changed to a different W/D/L token; trajectory moves shuffled, potentially illegal. Such a trajectory control measures sensitivity to invalid/shuffled input and does not by itself isolate legal rule inference |
| D20 | Proper probability scores and paired gains | NLL is soft-target cross entropy, not hard-label accuracy. Report vector squared Brier and expected categorical Brier, macro by variant; W/D/L ties and class counts accompany argmax accuracy. Equal-material uses conventional piece weights only as an analysis slice, never as labels |
| D21 | Uncertainty: seeds, demo draws, source families | Hierarchical bootstrap reports Monte Carlo label noise separately. Plug-in binomial SE can be zero at boundaries; Jeffreys Dirichlet marginal intervals are also available. Finite-R labels are noisy; no high-sample accuracy result exists until fresh high-R labels are generated |
| D22 | Familiar-rule adaptation separated from held-out-rule transfer | Original four movement variants form the candidate training set; remaining nine held out, with setup/visible-geometry controls reported separately. Pretrained Qwen can have seen named variants, so do not claim uncontaminated transfer |
| D23 | No empirical rule marginal is called an ICL ceiling | A rigorous rule-information gain bound requires aligned common-root counterfactual distributions and its assumptions. Different source distributions or noisy per-variant marginals cannot establish that bound. This analysis is deferred |
| D24 | Preliminary gates are conservative | Main requires all rule adapters, passing audits, complete labels, at least 32 replicates, both sides in each fold, measured positive pilot gains over kNN/mismatch, and actual H100 profiling. These are necessary, not sufficient, for preregistration approval |
| D25 | Stage0 is exploratory, including its structural `test` fold | No Stage0 fold is the paper's sealed final test. New source-family seed and protocol freeze are required for confirmatory data. No selection on confirmatory test metrics |
| D26 | No remote desktop interface is available in this chat | Tool discovery found no desktop control; app reports no attached terminal. Local nvidia-smi fails and CPU Torch reports no CUDA. Scripts are prepared; cluster execution is not fabricated |

Rule-specific ambiguities, including castling in diagonal-rook/mobile-king,
conversion-promotions, dark-square kings and the Long Board rename, are frozen in
[RULES.md](RULES.md). Any later change requires a new rules_version and regenerated
labels, with an explicit revision to this ledger.

## Development incidents retained for the paper's audit trail

1. Initial workspace had no source or Python scientific dependencies. Built the
shared core and a standard/antichess preliminary sample. User then supplied the
GitHub branch; cloned it and integrated the new package without altering old code.
2. Initial symmetry fuzz treated rank/color reflection as invariant for dark-square
immunity. Replay failed after 15 plies. Corrected the *test assumption*, documented
the geometric limitation, and made that transformation reject this rule.
3. Initial pyffish forward-capture pawn config omitted its custom initial double
step: reference missed all eight starting two-step moves. Corrected Betza definition;
a separate 300-position stream then matched.
4. First all-rule feasibility attempt aborted during replay of demonstrations:
python-chess rewrote a mobile king's e8a8 slide as castling notation. Overrode both
UCI conversion directions, added a regression fixture, and regenerated from the
same seeds. No partial failed dataset is presented as a completed result.
5. Initial PEFT smoke test failed (1 failed, 32 passed): layers_pattern did not
match bare AutoModel layer names. Switched to explicit last-four-layer projection
names and tested real injection/backprop using a locally constructed tiny Qwen2.
6. Changed CPU workers to multiprocessing spawn after observing warnings about
forking a multithreaded PyTorch process. Seed determinism tests cover both worker
counts. The ongoing first all-rule job had already loaded the earlier worker code;
its trace seeds remain deterministic, but provenance records its source snapshot.

Final executable checks and measured outcomes are linked in RESULTS.md. Failures
above are development findings, not research experiment outcomes.

7. The first 1,280-ply extension correctly failed the existing near-duplicate
check: later demonstration suffixes newly overlapped fixed query roots. Revised
extension to exclude entire conflicting demonstration games, retain their raw
traces in `excluded_demonstrations`, and record exclusions without changing query
roots or continuation seeds. Added a regression test. This can reduce feasible K;
the evaluator still rejects missing/full-context controls. The failed extension
was recomputed; its wall time was not separately instrumented, so it is not
included in a claimed exact total CPU cost.

8. The first scratch training attempt reached its first validation at update10
but report serialization rejected a NumPy int64 tie count. Preserved its partial
checkpoint/log in `results/tiny_pilot_failed_serialization*`, converted the count
to a Python int, added a JSON serialization regression assertion, and reran from
the same initialization. No partial run is treated as a completed checkpoint.
9. Final review preserves model-seed and original demo-draw identifiers separately
from the mismatch sampler's candidate index, and propagates all gradient-exposed
rules through warmup checkpoint lineage. This prevents mislabeled paired controls
or a false held-out claim caused by an exposed parent checkpoint. Standard chess
is scored as its own control group, not folded into the nine-rule transfer set.

## Cluster-access correction and next development stage

D27. The user correctly identified this host as the GPU cluster. A check outside
the restricted execution sandbox on host `wnckcaina01` found **eight H10080GB GPUs,
driver580.95.05 and192 logical CPUs**. Earlier statements that the host lacked
GPUs were wrong: the default sandbox hid device access. The initial CPU-only
PyTorch environment also could not test CUDA. Preserve the earlier logs as an
access-diagnosis failure, and use approved host execution plus CUDA2.6.0+cu124 for
actual GPU runs. Do not require the user to supply another remote desktop.

D28. The original one-seed scratch negative pilot remains a result, not a stopping
rule for all exploratory work. Proceed with an explicitly exploratory7B experiment
and stronger development dataset, as the user requested. Positive context gain is
a criterion for a positive research claim, not a reason to suppress a negative
7B experiment. Confirmatory protocol/seals remain separate from exploratory runs.

D29. Freeze the next **development** policy as terminal_aware_v1, based on the
retained small same-root probe and its piece-value-free interpretation. New seed
2026112402, R32,2560-ply administrative budget,8 demo/source families per fold and
rule, standard plus all13 variants. All policies remain in separate datasets.
The budget is increased to finish games; it never changes the simulator's draws.
If any continuation remains unresolved it stays unresolved and blocks strict
training, rather than being resampled or relabeled.

D30. Scale generation using48 persistent CPU workers and an exact compact Hamming
index. The index splits64 squares into r+1 disjoint interleaved subsets: at Hamming
distance≤r, one subset must match exactly. Candidates are checked by exact Hamming
distance, so this is an acceleration of the same filter, not approximate retrieval.
Keep the serial generator as a differential reference. Use newly generated family
folds; do not promote the already inspected Stage0 test fold to a final test.

D31. Actual H100 profile passed all4k/8k/16k/24k/32k lengths and exact save/reload
predictions. At32k the measured optimizer update took5.69s and peaked at32.31GB
allocated (decimal GB), with41.65GB reserved. Use the pinned32,768 native limit,
no RoPE scaling. SDPA emits a sliding-window warning; inspection shows
use_sliding_window=false and sliding_window131072 exceeds the native context, so
no within-window sliding cutoff is applied. Retain the warning log.

D32. Freeze an exploratory7B matrix before examining its model results: three
seeds42/123/456, common same-seed standard warmup30 updates, then four familiar
rules200 updates, accumulation4, K mixture{0,1}, validation every40 updates,
LR1e-4 and20-update warmup. Validation and inference batch8 with right padding and
length sorting. Main comparison K0 versus K1 relevant/mismatch/corruptions, five
demo draws. These are measured-scale exploratory runs, not the previously
illustrative2,000-update full matrix or a μP scaling study. Keep all negative
results. A later K sweep is an explicitly named extrapolation beyond trainingK1.

D33. Before opening the new test fold's model scores, save a frozen exploratory
protocol and bind each validation-selected checkpoint to it. Exploratory seals
require every data/simulator/hardware gate but may retain a failed positive-pilot
gain flag; they explicitly record `confirmatory_gates_passed=false`. They cannot
bypass unfinished labels, insufficient R, bad splits or missing GPU measurements.
The evaluator checks the sealed K set and demo draw count, not only hashes.

D34. Independently label the first two test family hashes per rule, keeping all
retained siblings, with R256, budget2560 and master seed2026112410. Root selection
uses IDs only. Save that design before generation; these fresh labels are not
used to train or choose checkpoints. Assess Monte Carlo error separately from
seed/demo/source-family uncertainty. Do not silently replace R32 target vectors
inside a published result without naming the high-R analysis.

D35. Cache deterministic tokenization of identical full episode text during
control matching. This changes runtime, not text, tokens, root cohorts, K or the
10% exact-token tolerance. All admitted controls still use actual complete encoded
inputs; no approximate length is substituted. Batched inference is compared with
identically formed repeated batches; different batch/padding shapes need not be
bitwise equal in BF16. All checkpoints are frozen during evaluation.

D36. Initial Qwen validation with the64-candidate mismatch sampler excluded67
roots. Before opening test scores, replace K1 matching with deterministic search
over all available foreign-rule full games, ranked by cached demo-block length
and a query/draw/family hash tie break. Admission checks the **complete** encoded
input against the exact10% demo-length tolerance; block lengths only rank search
order. Retain the original validation artifacts as `*_matching_v1` and do not
retune checkpoints. K>1 retains its separately documented candidate sampler. This
changes control construction, not the trained models or target policy. If a rule
still lacks a valid full-game control, record it explicitly rather than weakening
length matching or truncating games.

D37. Exhaustive K1 matching still found no suitable foreign-game lengths for some
validation demonstrations (notably dark-square immunity). Before opening test
scores, collect a fixed24 additional independent evaluation-fold demos per rule
with seed2026112403; no additional training demos. Keep all original query states,
R32 labels and trained checkpoints fixed. Reject whole new demos that are
unfinished or geometrically near any query and retain their traces/reasons. This
is an evaluation-dataset extension with its own parent hash; it does not overwrite
the training dataset or alter checkpoint selection. Do not manufacture a match by
padding, concatenating extra games atK1, truncating trajectories or weakening the
10% tolerance. Any remaining unsupported controls must be reported.

D38. Report two explicitly separate evaluation cohorts: (1) every root with full
K0/relevantK1 input, measuring probability prediction across all13 rules, and
(2) the strict common-root, length-matched control cohort used for relevance/ICL
attribution. A missing foreign-game length must not erase all-root prediction
results or be hidden as a successful match. Coverage-only reports carry their
own cohort label and cannot be mixed with matched-control reports in aggregation.
Use independent high-R labels for both cohorts where their preselected IDs appear.
The same frozen checkpoints and K/draw protocol apply; no test-based retraining.

D39. Source-root sampling is uniform over eligible adjacent prefix pairs after
the source game's realized length is known. This induces a game-length-dependent
root distribution. Only past state/history is exposed to the model; the source
future trajectory stays private, and targets come from independent continuations.
This is part of the declared data distribution, not uniform sampling over all
legal positions. Extra demo collection is not a causal policy-effect experiment.

D40. Final state review added preservation of promoted-piece markers in arbitrary
initial FENs and their mirrored states. All generated games start from unpromoted
standard/variant setups; their full histories already reconstruct every promotion,
so this does not change any retained dataset IDs, model inputs or labels. The
scratch vocabularyv1 explicitly rejects an initially promoted setup it cannot
encode losslessly; it does not silently discard those flags. The Qwen public-FEN
serializer preserves the initial marker and full history. Add a regression test.

Piece-conversion clarification: a royal king cannot capture a nonking if conversion
would remove the king; such a transition fails royal-king safety. The renamed
8×8 initial-pawn approximation has an effective maximum five-square multi-push,
since its nominal sixth step reaches the excluded multi-push promotion rank.
These follow the frozen transition rules; no labels are changed by this wording.

D41. A final adversarial regression review found a real bug in piece_capture:
the castling exception checked the target's type but not its color, allowing a
king capture of an enemy rook to bypass conversion. Correct this by checking the
pre-move target color. Freeze that rule as `research_rules_v2_royal_conversion`.
Regenerate its source games, demos, roots, R32 continuations and independent R256
subset; preserve all other roots/labels. Retain superseded data and the old all-rule
reference as invalid-for-v2 artifacts. Retrain the all-rule exposure reference.
The three familiar-rule models and standard warmups never saw piece_capture
positions or demos during gradients/checkpoint selection, so their training is
unaffected; all their evaluations must use corrected data/controls. No test-model
scores had been opened. Add king-capture tests for all five nonking victim types
and a castling regression. Do not relabel the buggy transition as an intended rule.

D42. Learned non-ICL baselines use training-fold roots from every gradient-exposed
rule, including the common standard-chess warmup, rather than only the later
four-rule mixture. This makes their available labeled data at least as broad as
the model's declared supervised exposure. They never fit validation/test labels.
Context kNN/prior baselines use only the actual selected demonstrations.

D43. The hash has70/15/15 eligible buckets, but pilot generation requests8 source
families **per fold per rule**. Thus this pilot has roughly equal retained fold
sizes after exclusions, not a final70/15/15 train/validation/test count. Report the
actual counts. Family assignment is stable and independent of outcomes; quotas
were fixed before generation.

D44. Publish completed-stage snapshots to Git, keeping raw datasets, live status,
rollout/prediction traces and binary checkpoints in local deliverable artifacts.
Commit hashes, sizes, configurations and completed audits/training metrics in the
snapshot. Respect HF_HOME in cluster scripts (default repository work/hf-cache),
cache the tokenizer during profiling, and avoid hardcoding this chat's generated
folder layout into published launchers. These are operational changes, not changes
to the measured policy, model weights or statistical protocol.


D45. After the user merged the validated-core branch, fetch and work on main.
Archive the legacy material/centipawn package, scripts, configs, tests, dependencies,
figures and results under archive/legacy_material_labels. Move the historical
Stage0 report and obsolete proposal/control configs to archive/development_stage0.
Keep the current W/D/L runtime, actual configs, tests, decisions and completed
snapshots active. Preserve history rather than deleting evidence. Future completed
stages are committed and pushed to main as requested.
