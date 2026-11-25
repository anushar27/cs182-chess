# Experiment artifacts

`published/` contains immutable, compact snapshots of completed stages committed
to Git: configuration and code hashes, audits, GPU profiles, dataset validation,
training/checkpoint records and summaries. Read the snapshot's README for scope.

Large raw datasets, individual rollout traces, prediction tables, live job logs,
model checkpoints and superseded development artifacts remain in this local
`results/` directory and are excluded from Git. The exact generation/evaluation
scripts and file hashes make their provenance reviewable. The final deliverable
archive contains the raw artifacts; do not infer that an absent Git file is an
unexecuted run. `docs/ARTIFACT_STATUS.md` distinguishes authoritative and invalidated
artifacts. No unfinished aggregate is promoted into a published result.
