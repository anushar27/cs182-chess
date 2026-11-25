# Fixed fresh validation panel, ready

This immutable panel contains448 validation roots,16 retained source families
and32 roots per rule/control, across13 variants plus standard. Every included
R32 trace passed replay/status/FEN/result/seed/count checks; history disjointness
checks passed and0 samples are unresolved. No model predictions are published.

The panel uses the exact indexed streams which later appear in the full larger
corpus. It adds zero independent samples to that corpus; final exact equivalence
verification remains pending until full generation reaches these validation roots.
The parent root cohort and all geometry decisions were already frozen. The main
48-worker generator was not interrupted. This16-worker side job only supplied
validation early, after approval review rejected a proposed checkpoint-stop.

Bind the raw manifest at results/large_dataset_v1/validation_panel/snapshots/validation-panel.json.
The raw shards remain local. Filter exposed rules for each experiment's checkpoint
selection. The original test/pilot records and completed benchmark snapshots stay
unchanged. This is not a completed large corpus or proof of adequate500M training
sample size. Read docs/LARGE_DATASET.md and the parent frozen cohort counts here.
