# Archived approaches and development records

The active project is the fixed-policy W/D/L pipeline described in the root
[README](../README.md). This archive preserves history; it is not an alternate
source of labels or current research claims.

## legacy_material_labels/

Original `chess_icl` package, original training/evaluation and cluster scripts,
legacy configuration files/tests/dependencies, old figures and results, and the
original README from commit `e6aa1fece489ac23552e51dbed67985fc60c9f5a`.
These experiments used material/centipawn-derived labels and other targets that
were replaced by empirical fixed-policy outcomes. Their reported accuracy and
scaling results must not be reused as W/D/L-policy results.

## development_stage0/

Historical CPU/uniform-policy report, its report-generation script, proposed
unexecuted large-run configurations, and the superseded control-matching protocol.
The report explicitly identifies superseded capture-rule results. These files
remain for the research decision trail; active commands live in `scripts/v2_*`
and active protocol/configuration files live in `configs/`.

Raw rollout traces, checkpoints and live experiment logs are not added to Git.
Their authoritative status and hashes are documented in
[ARTIFACT_STATUS.md](../docs/ARTIFACT_STATUS.md) and completed snapshots in
[results/published](../results/published/). Consult historical Git revisions when
reproducing an archived script's original filesystem layout.
