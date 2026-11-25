# Validated core and completed training snapshot

This commit contains the completed corrected simulator/data pipeline, audits,
actual H100 profile, exact dataset hashes and all completed Qwen training records.
The three familiar-rule runs use valid four-rule/standard gradient subsets; the
capture-conversion repair regenerated only that held-out rule. The all-rule
reference was retrained on corrected data. See docs/ARTIFACT_STATUS.md.

The final multi-seed test aggregate is **not part of this snapshot**. Evaluations
were still running when this snapshot was created. No provisional test result is
called final. The failed positive-pilot gate is preserved; these runs are explicitly
exploratory, with all data/simulator/hardware gates passing.

Raw data, individual continuation traces, live logs and binary checkpoints remain
in the local deliverable archive. Their file/checkpoint hashes and reproduction
scripts are provided. Published manifest paths are normalized to repository-relative
paths; the original manifest file hash is recorded alongside each copy.
