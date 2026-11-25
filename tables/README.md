# Measured tables

`primary_results.csv` and `.tex` are the compact primary table: K0 versus relevant
K1 on the same roots, averaged within rule and then equally across rules, with
three model seeds and five context draws. Positive gain is better. The95% intervals
use the declared1,000-replicate seed0 hierarchy and condition on empirical R32
labels; see the study's uncertainty limitations and separate R256 results.

The other CSV/LaTeX pairs retain complete metrics, rule-level results, matched
comparisons, seven baselines, the one-seed all-rule exposure reference and actual
rollout timing. Full metric tables are wide appendix tables; split columns or use
landscape/`graphicx` scaling when typesetting. They are not hypothetical values.
The reference's group labels refer to the primary model's rule groups: the
reference itself trains all thirteen variants.

The per-rule and overall tables average predictions as scores, not probabilities.
This differs from the explicitly labeled descriptive calibration ensemble.
See `docs/EXPLORATORY_STUDY.md` for exclusion counts, direction conventions,
independent high-R sensitivity and the negative interpretation.
