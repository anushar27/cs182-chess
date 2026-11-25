# Figure definitions and captions

All values come from retained corrected artifacts. The script `scripts/v2_figures.py`
exports SVG for Git and PDF/PNG for the local research bundle. It uses no
illustrative outcome values.

- **policy_outcome_diversity:** Mean empirical root-player W/D/L proportions for
  training and validation roots under terminal_aware_v1, R32. Standard is a separate
  control. These different root populations do not identify a causal policy effect.
- **h100_context_memory:** Measured peak allocated/reserved memory for one actual
  Qwen-7B LoRA optimizer update at each synthetic context length. Reserved memory
  includes allocator cache. These probes establish feasibility, not chess quality
  or dedicated-device throughput.
- **test_context_gain:** Macro paired NLL(K0)−NLL(K1), three seeds and five context
  draws, same206 test roots/105 source families across all conditions. Familiar:
  four rules/61 roots/31 families; held-out: nine rules/129 roots/66 families;
  standard:16 roots/eight families. Bars are percentile95% intervals from the
  declared1,000-replicate, seed0 model/draw/family hierarchy, conditional on R32
  labels. Positive means context improves predictions. Rules have equal weight.
- **test_relevance_gain:** NLL(mismatchedK1)−NLL(relevantK1) on those same paired
  inputs, with foreign-demo length matched within10%. Same intervals and label-noise
  limitation. Positive indicates an advantage for relevant context.
- **test_calibration_ensemble:** Descriptive classwise reliability, split by side
  to move, on all211 test roots. Average probabilities across three model seeds
  and, at K1, five draws before binning into ten equal-width bins. Each point
  compares mean predicted and empirical R32 probability among roots in that bin.
  Counts are in the published `calibration_ensemble_bins.json`; bins can contain
  only one root. No uncertainty bars or calibration-improvement inference is
  implied. This ensemble differs from the primary mean-of-model NLL estimand.

Both full test figures condition on empirical R32 labels. Independent R256 scoring
uses56 preselected roots/28 families (53 roots/27 families for matched controls) and
is documented in `docs/EXPLORATORY_STUDY.md`, rather than silently replacing labels
in these figures.
