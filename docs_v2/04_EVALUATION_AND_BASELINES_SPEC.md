# Chess-Variant ICL v2: Unified Evaluation, Baselines, Controls & Uncertainty Specification

**Date:** 2026-11-25  
**Package:** `chess_icl_v2.eval`  
**Modules:** `metrics.py`, `baselines.py`, `controls.py`, `uncertainty.py`

---

## 1. Outcome Representation & Perspective Convention

Every query state $s_q$ has a soft empirical outcome target vector $y \in \Delta^2$ estimated from $R$ independent simulator continuations from $s_q$ under $(T_v, \pi)$ up to horizon $H$:
$$y = (y_W, y_D, y_L) \in [0, 1]^3, \qquad y_W + y_D + y_L = 1,$$
indexed as `0 = win`, `1 = draw`, `2 = loss` **from the query side-to-move's ($c_q \in \{\text{White}, \text{Black}\}$) perspective**.

By contrast, each demonstration game $D_k$ records its realized outcome $o_k \in \{W, D, L\}$ **from White's perspective**. Any baseline that maps demonstration outcomes to query predictions must apply the perspective involution $\Phi_{c_q}$:
$$\Phi_{\text{White}}(p_W, p_D, p_L) = (p_W, p_D, p_L), \qquad \Phi_{\text{Black}}(p_W, p_D, p_L) = (p_L, p_D, p_W).$$

---

## 2. Proper Scoring Metrics, Calibration, Context Gain & MC Floor (`metrics.py`)

Let $\hat{p}_i = (\hat{p}_{i,0}, \hat{p}_{i,1}, \hat{p}_{i,2}) \in \Delta^2$ and $y_i = (y_{i,0}, y_{i,1}, y_{i,2}) \in \Delta^2$ for $i = 1, \dots, N$.

### 2.1 Soft-Target Cross-Entropy / Negative Log-Likelihood (`compute_nll`)
$$\mathrm{NLL}_i = -\sum_{c=0}^{2} y_{i,c} \log \mathrm{clip}(\hat{p}_{i,c}, \varepsilon, 1.0), \qquad \overline{\mathrm{NLL}} = \frac{1}{N} \sum_{i=1}^{N} \mathrm{NLL}_i,$$
with default clipping $\varepsilon = 10^{-7}$. Returns both `mean_nll` and the per-sample vector `per_sample_nll`.

### 2.2 Multiclass Brier Score (`compute_brier`)
$$\mathrm{Brier} = \frac{1}{N} \sum_{i=1}^{N} \sum_{c=0}^{2} (\hat{p}_{i,c} - y_{i,c})^2 \in [0, 2].$$

### 2.3 L1 Error and Total Variation Distance (`compute_l1_tv`)
$$\mathrm{L1}_i = \sum_{c=0}^{2} |\hat{p}_{i,c} - y_{i,c}|, \qquad \mathrm{TV}_i = \frac{1}{2} \sum_{c=0}^{2} |\hat{p}_{i,c} - y_{i,c}| \in [0, 1].$$

### 2.4 Argmax Accuracy & Class Distribution (`compute_argmax_accuracy`)
With $\hat{c}_i = \arg\max_c \hat{p}_{i,c}$ and $c_i^\star = \arg\max_c y_{i,c}$:
$$\mathrm{Acc} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[\hat{c}_i = c_i^\star],$$
reported alongside `pred_counts`, `true_counts`, and `per_class_accuracy` for `{win, draw, loss}` to surface degenerate majority-class collapse.

### 2.5 Expected Calibration Error & Side-to-Move Calibration (`compute_ece`, `compute_side_calibration`)
For soft probability targets $y_i$, let $\hat{c}_i = \arg\max_c \hat{p}_{i,c}$ and $\mathrm{conf}_i = \hat{p}_{i, \hat{c}_i}$. The expected accuracy under $y_i$ of predicting $\hat{c}_i$ is $a_i = y_{i, \hat{c}_i}$. Partitioning $[0, 1]$ into $M = 10$ uniform bins $B_1, \dots, B_M$:
$$\mathrm{ECE}_{\text{top}} = \sum_{b=1}^{M} \frac{|B_b|}{N} \left| \frac{1}{|B_b|}\sum_{i \in B_b} a_i - \frac{1}{|B_b|}\sum_{i \in B_b} \mathrm{conf}_i \right|.$$
We also compute **classwise ECE** averaged over $c \in \{0, 1, 2\}$:
$$\mathrm{ECE}_{\text{class}} = \frac{1}{3} \sum_{c=0}^{2} \sum_{b=1}^{M} \frac{|B_{b,c}|}{N} \left| \overline{y}_{B_{b,c}, c} - \overline{p}_{B_{b,c}, c} \right|.$$
`compute_side_calibration` partitions samples into White-to-move ($N_W$) and Black-to-move ($N_B$) slices and reports separate NLL, Brier, top-label ECE, classwise ECE, mean predicted $\overline{\hat{p}}$, mean true $\overline{y}$, and `side_nll_gap = |NLL_W - NLL_B|`. This directly diagnoses perspective-inversion bugs.

### 2.6 Context Gain (`compute_context_gain`)
Evaluated strictly on **identical query roots** across $K=0$ and $K>0$:
$$\Delta \mathrm{NLL}_i(K) = \mathrm{NLL}_i(0) - \mathrm{NLL}_i(K).$$
Positive values signify improvement from in-context demonstrations. Reports overall mean, sample std (`ddof=1`), standard error, per-variant breakdown, and unweighted **macro-average** across evaluation variants:
$$\overline{\Delta \mathrm{NLL}}(K) = \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} \frac{1}{|I_v|} \sum_{i \in I_v} \Delta \mathrm{NLL}_i(K).$$

### 2.7 Monte Carlo Rollout Noise Floor (`compute_mc_noise_floor`)
When $y_i$ is estimated from $R$ i.i.d. multinomial rollouts $\hat{y}_i \sim \frac{1}{R}\mathrm{Multinomial}(R, p_i)$:
1. **Irreducible Entropy Floor:**
   $$H(y) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c: y_{i,c}>0} y_{i,c} \log y_{i,c}.$$
2. **Miller-Madow Finite-Sample Bias:** With effective support $k_{\mathrm{eff}, i} = |\{c : y_{i,c} > 0\}|$:
   $$\mathrm{Bias}_R = \frac{1}{N} \sum_{i=1}^{N} \frac{k_{\mathrm{eff}, i} - 1}{2R}, \qquad H_{\text{debiased}} = H(y) + \mathrm{Bias}_R.$$
3. **Rollout NLL Variance per Root:**
   $$\mathrm{Var}_{\text{rollout}}(\mathrm{NLL}_i) = \frac{1}{R}\left( \sum_{c: y_{i,c}>0} y_{i,c} (\log y_{i,c})^2 - H(y_i)^2 \right).$$
4. **Bayes Brier Floor & Multinomial Target Variance:**
   $$\mathrm{Brier}_{\text{Bayes}} = \frac{1}{N}\sum_{i=1}^N \left(1 - \sum_{c=0}^2 y_{i,c}^2\right), \qquad \mathrm{Var}_{\text{MC}}(\hat{y}) = \frac{1}{N}\sum_{i=1}^N \frac{1 - \sum_{c=0}^2 y_{i,c}^2}{R}.$$

---

## 3. Non-Neural Baselines & Slicing (`baselines.py`)

All predictors implement the `Predictor` protocol: `predict_proba(episodes: list[dict]) -> np.ndarray` returning valid simplex rows of shape `(N, 3)`.

1. **`UniformBaseline`:** Predicts $(1/3, 1/3, 1/3)$ for all queries.
2. **`TrainingPriorBaseline`:** Fits marginal W/D/L distribution $\pi_{\text{train}} = \frac{\sum_i y_i + \alpha}{\sum_i \mathbf{1} + 3\alpha}$ on training query targets.
3. **`DemoMajorityBaseline`:** Computes Laplace-smoothed empirical White-perspective distribution over the $K$ demos with smoothing $\alpha$ and prior $p_0$ (default uniform):
   $$p_{\text{White}} = \frac{\mathbf{c}_{\text{White}} + 3\alpha p_0}{K + 3\alpha}, \qquad \hat{p} = \Phi_{c_q}(p_{\text{White}}).$$
   Falls back to $p_0$ when $K = 0$.
4. **`LengthHistogramBaseline`:** Fits a convex L2-regularized multinomial logistic regression (pure PyTorch L-BFGS with strong Wolfe line search on standardized features, `l2_reg=1e-4`) on the 7 episode summary features:
   $$x_{\text{len}} = \big[\mu_{\text{len}},\ \sigma_{\text{len}},\ f_W,\ f_D,\ f_L,\ \mathbf{1}[c_q=\text{White}],\ K\big].$$
5. **`QueryMaterialBaseline`:** Fits a convex L2-regularized multinomial logistic regression on the 9 static query board features (using standard piece values $P=1, N=3, B=3, R=5, Q=9$, from query side-to-move's perspective):
   $$x_{\text{mat}} = \big[\Delta M_{\text{stm}},\ M_{\text{own}},\ M_{\text{opp}},\ \Delta P,\ \Delta N,\ \Delta B,\ \Delta R,\ \Delta Q,\ \mathbf{1}[c_q=\text{White}]\big].$$
6. **`BoardKNNBaseline`:** Encodes every board state into a $(64 \times 13 + 1) = 833$-dimensional vector: 12 piece channels (`P..K`, `p..k`) + 1 empty channel (`.`) per square (blocked squares `X` remain all-zero across the 13 channels so they remain distinct from both pieces and empty), plus 1 side-to-move bit. Reconstructs demo states either along full trajectories (`state_mode="trajectories"`) or final states (`state_mode="final"`) using variant-safe UCI mailbox execution (without standard chess legality rejection), retrieves the top-$k$ nearest demo states by cosine or normalized Hamming similarity $s_j \in [0, 1]$, forms the similarity-weighted White-perspective distribution
   $$p_{\text{White}} = \frac{\sum_{j \in \text{top-}k} w_j \mathbf{e}_{o_j} + 3\alpha p_0}{\sum_{j \in \text{top-}k} w_j + 3\alpha},$$
   and maps $\hat{p} = \Phi_{c_q}(p_{\text{White}})$.
7. **`EqualMaterialSlice` / `filter_equal_material`:** Selects query positions satisfying $|M_{\text{White}} - M_{\text{Black}}| \le \text{threshold}$ (default `threshold=2`) to isolate rule-learning effects from trivial material imbalances.

---

## 4. Scientific Controls (`controls.py`)

All control builders deep-copy input episodes and never mutate inputs:

1. **`make_mismatched_context_episodes(episodes, other_variant_demos_by_variant, seed)`:** Replaces each episode's $K$ demonstrations with $K$ demonstrations from a different variant $v' \neq v$ (preferring a single mismatched variant with $\ge K$ demos), greedily matched without replacement to be closest in token/move length to the original demonstrations. Isolates rule-specific ICL from generic length/format conditioning.
2. **`make_corrupted_outcome_episodes(episodes, mode, seed)`:** Supports:
   - `mode="inverted"`: swaps White `win <-> loss` while preserving `draw` (tests sensitivity to outcome-rule polarity).
   - `mode="all_draw"`: replaces all demo outcomes with `draw`.
   - `mode="scrambled"`: randomly permutes demo outcomes across the episode batch (or samples uniformly from `{win, draw, loss}` if all input outcomes are identical), breaking move-to-outcome alignment while preserving marginal outcome frequencies.
3. **`make_corrupted_trajectory_episodes(episodes, seed, mode="shuffled")`:** Preserves each demo's exact move count and White-perspective outcome while corrupting the move sequence (`mode="shuffled"` permutes moves within each demo; `mode="mismatched"` substitutes moves sliced from another demo/variant) and clears cached `fens`/`states`/`final_fen`/`tokens` fields. Tests whether models genuinely parse move trajectories rather than relying solely on demo length and outcome counts.

---

## 5. Hierarchical Uncertainty Quantification (`uncertainty.py`)

Evaluation spans multiple levels of stochasticity: `model_seeds` ($S$), `demo_draws` ($D$, e.g. 5 draws per $K$), correlated query roots within `source_game_ids` ($G$ games), and evaluation `variants` ($|\mathcal{V}|$).

1. **Cluster Bootstrap by `source_game_id` (`bootstrap_by_source_game`):**
   Let clusters $g \in \{1, \dots, G\}$ have metric sums $S_g = \sum_{i \in g} x_i$ and sizes $m_g = |g|$. Each bootstrap replicate draws $G$ clusters uniformly with replacement (cluster multiplicities $w_g$) and computes:
   $$\bar{x}^{(b)} = \frac{\sum_{g=1}^{G} w_g S_g}{\sum_{g=1}^{G} w_g m_g}, \qquad b = 1, \dots, B.$$
   Reports point mean, bootstrap std, and percentile $1-\alpha$ confidence interval $[\mathrm{q}_{\alpha/2}, \mathrm{q}_{1-\alpha/2}]$.
2. **Seed & Draw Hierarchy (`aggregate_seed_draw_hierarchy`):**
   Given tensor $X \in \mathbb{R}^{S \times D \times N_{\text{roots}}}$:
   - Averages over $(S, D)$ per root for cluster-bootstrap game sampling uncertainty $\hat{\sigma}_{\text{game}}$.
   - Reports per-seed means and across-seed standard error $\hat{\sigma}_{\text{seed}} / \sqrt{S}$.
   - Reports within-seed across-draw standard deviation $\hat{\sigma}_{\text{draw}}$ and $\hat{\sigma}_{\text{draw}} / \sqrt{D}$.
   - Reports combined standard error $\mathrm{SE}_{\text{comb}} = \sqrt{\hat{\sigma}_{\text{game}}^2 + \hat{\sigma}_{\text{seed}}^2 / S}$.
3. **Variant Macro-Averaging (`macro_average_variants`, `aggregate_evaluation_records`):**
   Runs stratified cluster bootstrap independently within each variant $v \in \mathcal{V}$ and forms the unweighted macro-average bootstrap distribution $\overline{x}^{(b)} = \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} \bar{x}_v^{(b)}$, yielding per-variant and aggregate macro-average 95% CIs.

---

## 6. Verification

Verified with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest tests_v2/test_eval_v2.py -v` covering:
- Soft-target NLL (with `1e-7` clipping), Brier score, L1/TV, argmax accuracy + class counts, top-label/classwise ECE, side-to-move calibration, context gain (array and mapping APIs), and MC noise floor.
- All baselines (`UniformBaseline`, `TrainingPriorBaseline`, `DemoMajorityBaseline` with White->Black perspective flip & `K=0` fallback, `LengthHistogramBaseline`, `QueryMaterialBaseline`, `BoardKNNBaseline` with `64*13 + 1` encoding including `'X'` blocked squares, cosine/Hamming metrics, trajectory/final modes, and perspective flip, plus `EqualMaterialSlice` / `filter_equal_material`).
- All three non-mutating control builders (`make_mismatched_context_episodes`, `make_corrupted_outcome_episodes`, `make_corrupted_trajectory_episodes`).
- Cluster bootstrap by `source_game_id`, seed/draw hierarchy, and macro-averaging across variants.

