# Completed exploratory fixed-policy W/D/L test snapshot

All three primary model seeds and the one-seed all-rule reference are complete.
The final test uses canonical unpadded batch1 inference of the exact saved inputs,
with frozen weights and identical results across overlapping cohorts. Validation
retains the original padded procedure. The final audit and 51 tests pass.

The publication contains compact metrics, intervals, independent R256 sensitivity,
all exclusions, numerical diagnostics and exact raw-artifact/checkpoint hashes.
Read `docs/EXPLORATORY_STUDY.md` for methods, results and limitations, with CSV and
LaTeX exports under `tables/`. The older validated-core snapshot is unchanged.

Full prediction rows, all R32/R256 continuation traces, legacy/failed-stage evidence
and trainable checkpoint binaries remain in the separately generated local ZIP.
Pinned Qwen base weights are downloaded separately and are not bundled. No claim
of optimal play, confirmatory significance, scaling laws or contamination-free
rule learning is made. Preserve the negative controls and development incidents.
