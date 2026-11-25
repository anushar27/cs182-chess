# Authoritative and superseded artifacts

Do not pool counts across files that copy or extend the same random streams.
Each dataset has explicit hashes and lineage. The final report uses the corrected
rules and corrected evaluation controls listed below.

| Artifact | Status and use |
|---|---|
| terminal_policy_rules_v2.dataset.json | Authoritative training/reference pool:614 roots, R32; corrected piece_capture rule; other root states/labels unchanged |
| terminal_evaluation_rules_v2.dataset.json | Authoritative evaluation pool: identical614 roots/labels,1000 full demonstrations including independent evaluation-only additions |
| high_r_test_labels_rules_v2.json | Authoritative independent R256 subset:56 preselected test roots; corrected capture subset; no unresolved continuations |
| simulator_audit_rules_v2.json | Latest full audit after the capture correction; read beside the regression test report |
| qwen_warmup_seed42/123/456 | Valid standard-only warmup parents; no capture-variant gradients |
| qwen_familiar_seed42/123/456 | Valid four-movement-rule models. Their input/label subsets and checkpoint selection are unchanged by the capture repair. The original full training-file hash is retained honestly |
| qwen_direct_exposure_seed42 | Corrected all-rule reference, retrained from the same standard parent; no held-out transfer claim |
| qwen_validation/test/test_coverage_seed*.json | Final corrected-data evaluations when present; separate matched-control and all-root prediction cohorts |
| seals/ | Frozen checkpoint/data/protocol bindings; explicitly exploratory, with all data/hardware gates and the negative pilot flag preserved |
| cluster_environment.json, h100_profile.json | Actual CUDA/H100 evidence; supersedes the restricted shell's apparent missing-device diagnosis |

## Retained history, not final evidence for the corrected capture rule

- `terminal_policy.dataset.json` and `terminal_evaluation.dataset.json` contain the
old capture-conversion implementation. Unchanged non-capture subsets were copied
and verified into the corrected artifacts; do not use their old capture row.
- `high_r_test_labels.json` has the superseded capture subset; use the versioned
replacement, which retains every other high-R result.
- `qwen_direct_exposure_piece_capture_v1_seed42` learned the superseded rule and
is not the corrected all-rule reference.
- `qwen_direct_exposure_failed_rules_version*` records a launcher retry that still
pointed at the old file. Strict coverage rejected the mismatched rule versions
before any optimizer updates; no trained checkpoint resulted.
- `qwen_validation_matching_v1_seed*` records the first64-candidate sampler and
its67 excluded roots. Subsequent matching feasibility files document the control
repair before any final test scores were opened.
- Uniform Stage0 and the original scratch pilot are development history. Their
capture-variant labels/controls are superseded. Standard/antichess preliminary
results and the scratch model's four-rule gradients remain valid for their stated
scope; they are not a paper headline or confirmatory transfer result.
- Earlier simulator audits sampled reachable positions without exercising the
enemy-rook/king-conversion corner case. Keep them as evidence of coverage limits,
not proof that the bug was absent. The final golden suite adds the missed cases.

The final code and ledger retain unsuccessful feasibility stages, failed runs,
control exclusions, and measured limitations. No superseded artifact is silently
substituted into a favorable final metric.
