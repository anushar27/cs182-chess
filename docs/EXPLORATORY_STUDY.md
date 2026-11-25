# Fixed-policy chess-variant probability prediction: exploratory study

## Objective and scope

The target is the player-to-move W/D/L distribution when both players follow
`terminal_aware_v1`: choose uniformly among legal immediate wins if any exist,
otherwise uniformly among all legal moves, sorted by UCI before seeded sampling.
The same policy is used for full demonstrations, source games and independent
label continuations. No material evaluation or optimal-play labels enter targets.
The experiment covers thirteen project variants and a separate standard control.

This is a completed small exploratory experiment, not a confirmatory paper result.
The negative scratch pilot was retained; the subsequent H100 experiment explicitly
bypassed only its positive-gain gate under the user's instruction to run the study.
All simulator, split, complete-target, R32, side and hardware gates passed. The
test protocol and selected checkpoints were sealed before test-model comparisons.
The structural test folds had already appeared in development feasibility records.

## Methods and reproducibility

See [exact rules](RULES.md), [implementation contracts](IMPLEMENTATION.md),
[decisions and incidents](DECISIONS.md), [cluster measurements](CLUSTER_RESULTS.md),
[artifact validity](ARTIFACT_STATUS.md), and [runbook](REPRODUCING.md).
Rules were reconstructed from the original repository and the design attachment;
ambiguous choices were frozen and recorded, including royal king behavior,
castling, promotion/capture conversion, color-dependent immunity and draw claims.
The long-board proposal is explicitly an 8×8 initial-pawn-extension rule.

Full replay states retain initial position, every past move, castling, en passant,
clocks, promotion information and fixed barriers. Inputs contain complete terminal
demonstration trajectories and their White outcomes, then the full query and a
prediction marker. No rule description/name or private family/label information is
included. Overflow rejects the episode; neither the query nor a demo is trimmed.
Query history and visible geometry may themselves reveal a rule at K=0.

Training uses pinned Qwen2.5-7B-Instruct, last-four-layer attention LoRA r8/alpha16,
and a three-way FP32 head: 731,651 trainable parameters. Three seeds (42,123,456)
receive 30 standard warmup updates, then 200 updates on diagonal rook, straight
bishop, weakened queen and extended pawn, with K=0/1, accumulation4, LR1e-4 and
20-update main LR warmup. Checkpoints minimize macro validation soft NLL over
the four training rules; no checkpoint was selected using test scores. The other
nine variant rules receive no gradients in these models. The one-seed all-rule
reference starts from the same standard warmup and receives 200 variant updates.
Its exposure and number of distinct labeled roots differ; it is not a causal
estimate of exposure alone or an upper bound.

The corrected pool contains 614 roots (206 train,197 validation,211 test), each
with 32 independent continuations, and 1,000 full demonstration games after
evaluation-only additions. The pools and query-source families are separate;
cross-fold roots and query-vs-demo states pass the stated Hamming-distance audit.
All 19,648 R32 continuations terminated within 2,560 plies. Unfinished runs would
produce null targets and marginal identification bounds, never automatic draws.
Eight source/demo families were requested per fold/rule; hash eligibility70/15/15
does not imply that the final quotas have a70/15/15 ratio.

Primary test comparisons use the same206 roots across K0, relevantK1, foreign-rule
K1 and both corruption controls, with five draws per context. Foreign demos must
match complete encoded demonstration length within10%. Five roots fail matching;
the separate coverage cohort retains all211 roots for K0/relevantK1. Weights stay
frozen. After discovering batch-dependent BF16 probabilities (up to about0.044),
we rescored exactly the saved inputs individually without padding. Original input
hashes, root cohorts, matching, targets and selected weights are unchanged; old
padded outputs remain diagnostics. Identical token sequences use the same output.
Validation checkpoint selection retains the original padded procedure. This
numerical analysis amendment was recorded after the first test pass, before final
comparative claims, and must not be described as preregistered.

Metrics are soft-target NLL and squared-vector Brier, averaged within rule then
equally across rules. Positive gain means lower NLL with relevant context. The
1,000-replicate, seed0 hierarchy resamples model seeds, demo draws and source
families within fixed rules. Its intervals condition on empirical labels; Monte
Carlo label uncertainty is assessed separately. Only three model seeds and few
families per rule limit inference. All intervals and ablations are exploratory;
no multiplicity-adjusted significance claim is made.

## Frozen matched-cohort results

| Rule set | Roots/families | K0 NLL | Relevant K1 NLL | Mismatched K1 NLL | Relevant Brier |
|---|---:|---:|---:|---:|---:|
| familiar_rules | 61/31 | 1.069144 | 1.103794 | 1.106581 | 0.131821 |
| heldout_rules | 129/66 | 1.075007 | 1.088554 | 1.095761 | 0.222578 |
| standard_control | 16/8 | 1.077991 | 1.101167 | 1.102852 | 0.096765 |

| Rule set | Relevant gain over K0 [95% interval] | Relevance advantage over mismatch [95% interval] |
|---|---|---|
| familiar_rules | -0.034650 [-0.058684, -0.015994] | 0.002787 [-0.003923, 0.009566] |
| heldout_rules | -0.013547 [-0.071814, 0.023449] | 0.007207 [-0.001399, 0.015828] |
| standard_control | -0.023176 [-0.058242, 0.005585] | 0.001686 [-0.013996, 0.016943] |

The primary group has four familiar rules; transfer has nine gradient-held-out rules. Standard is a separate warmup control. Raw per-seed, per-rule, side, equal-material and calibration records are retained in the machine-readable analysis.

## Baselines and interventions

The completed study does not demonstrate beneficial rule-specific ICL. Relevant context worsens familiar-rule predictions; the held-out estimate is negative but inconclusive. Neither rule group has a resolved relevance advantage over matched foreign-rule context. This is a negative result for this training/data scale, not evidence that the task is impossible.

Learned baselines use only 73 training roots from exposed rules, including standard warmup. They receive no rule IDs. Baselines cover frequency/majority, material-count regression, length/outcome histograms, demo frequency/majority and demo-state kNN. K1 demo frequency and majority coincide because there is only one game.

| Baseline, relevant K1 | Familiar macro NLL | Held-out macro NLL |
|---|---:|---:|
| demo_frequency | 1.134578 | 1.129651 |
| demo_majority | 1.134578 | 1.129651 |
| knn_demo_states | 1.428764 | 1.413973 |
| length_histogram | 1.101572 | 1.123976 |
| query_material | 1.054037 | 1.062355 |
| training_frequency | 1.097150 | 1.127126 |
| training_majority | 1.130638 | 1.168924 |

| Rule set | Corruption | NLL(corrupt) − NLL(relevant) [95% interval] |
|---|---|---|
| familiar_rules | corrupt_outcomes | -0.004863 [-0.013737, 0.004738] |
| familiar_rules | corrupt_trajectories | 0.000551 [-0.002604, 0.003746] |
| heldout_rules | corrupt_outcomes | 0.003962 [-0.009186, 0.017785] |
| heldout_rules | corrupt_trajectories | 0.001416 [-0.001745, 0.004610] |
| standard_control | corrupt_outcomes | 0.005137 [-0.009376, 0.020831] |
| standard_control | corrupt_trajectories | -0.000750 [-0.005358, 0.004611] |

Shuffled trajectories are generally illegal and shift the input distribution; they are a sensitivity intervention, not a valid alternate policy.

## Independent high-R sensitivity analysis

The first two retained test-family hashes per rule, all siblings, select 56 roots from 28 families for 14,336 fresh R256 continuations. None is unresolved and no R32 random stream is reused. Mean R32–R256 vector L1 difference is 0.158343; maximum class difference is 0.218750. Per-root counts, plug-in SE and marginal Wilson intervals are retained. R256 still estimates policy probabilities; it is not ground truth.

| Rule set | High-R matched roots/families | Relevant NLL at R256 | Relevant gain over K0 [95% interval] |
|---|---|---:|---|
| familiar_rules | 16/8 | 1.074874 | -0.028122 [-0.064999, -0.000892] |
| heldout_rules | 33/17 | 1.078401 | -0.005289 [-0.073070, 0.039464] |
| standard_control | 4/2 | 1.107879 | -0.031940 [-0.063034, -0.007157] |

The all-root high-R analysis and baseline/reference re-scores are retained separately. Bootstrap intervals do not propagate Monte Carlo label uncertainty; the independent re-estimate is a sensitivity check on only two families per rule.

## All-rule exposure reference

| Primary model rule group | Seed42 primary relevant NLL | All-rule reference relevant NLL |
|---|---:|---:|
| familiar_rules | 1.092461 | 1.078241 |
| heldout_rules | 1.054465 | 0.990916 |
| standard_control | 1.062617 | 1.069621 |

The reference has gradients from all thirteen variants; its rows on the primary model's held-out rule set do not measure held-out transfer. One reference seed does not support an across-seed exposure-effect claim.

## Numerical inference audit

The maximum probability change from the original padded test outputs to the canonical policy was 0.077508; mean absolute changes per model were 0.010756, 0.011234, 0.010123, 0.009428. These are floating-point inference-policy differences, not changes to labels or weights.

## Interpretation and limitations

Context improvement over K0 alone cannot establish inference of a rule: compare
the matched foreign-rule context, shortcut baselines and corruptions. The tables
report all controls irrespective of direction. A gain interval spanning zero is
inconclusive at this sample size; it does not prove no effect. Even a positive
relevance advantage requires larger independent families and a newly frozen
confirmatory test before a broad rule-learning claim. Named variants may have
appeared in Qwen pretraining. Simulator audits are sampled, with independent
engine comparisons for ten rule/control adapters and golden/property tests for
four in-house rules; these are not correctness proofs.

The study measures neither engine strength nor optimal-play outcomes. No larger-K
learning curve, scratch scaling law, μP validation, zero-context-only training
ablation or fully powered confirmatory study was run. Those are future experiments,
not missing rows to fill with inferred numbers. Retained failed stages and the
capture-rule correction are part of the scientific record.

Generation cost is reported per variant in `tables/generation_cost.csv`: actual
continuation lengths and worker elapsed seconds, with a clearly labeled linear
worker-hour extrapolation. Elapsed time includes sharing/scheduling and is not a
measurement of CPU core time. The original217.97-second48-worker generation phase
precedes audit, capture repair and evaluation-demo additions; it is not the total
end-to-end cost of this corrected study. GPU loop and canonical-rescore times are
retained per run rather than interpreted as dedicated-device throughput.

## Tables, figures and retained evidence

CSV and LaTeX tables are under `tables/`; SVG, PNG and PDF figures under `figures/`.
`results/final_analysis.json` includes per-seed metrics, all slices, intervention
intervals, paired baseline comparisons and the complete exclusion list.
`results/high_r_analysis.json` carries all independent-noise records and re-scores.
The final publication snapshot contains compact results and exact raw-artifact
hashes. Full datasets, continuation traces and checkpoint binaries are stored in
the local deliverable archive rather than Git. Reproduction uses the pinned model
revision and environments; numerical kernels can change exact floating-point
values across different hardware/software versions.
