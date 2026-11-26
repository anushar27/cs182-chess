# Paper Draft Sections: Introduction, Background, Related Work, Method Math & Hyperparameter Justification

**Target Venue Standard:** CS 182/282 NeurIPS-format Research Paper  
**Scope:** Self-contained methodology and literature positioning for `chess_icl_v2`. Empirical protocol cells dependent on Stage 0 rollout profiling are marked `[TBD Stage 0]`.

---

## 1. Introduction & Novelty Statement

In-context learning (ICL) allows sequence models to adapt their predictions to novel tasks purely from input-output demonstrations in the prompt, without gradient updates. Yet most controlled ICL benchmarks evaluate continuous regression, synthetic linear operators, or discrete classification where the latent mapping acts directly on surface features. Real-world reasoning frequently requires inferring **unobserved transition dynamics and terminal rules** from behavioral trajectories alone—for example, deducing how pieces move, capture, or win by watching $K$ unlabeled games before evaluating a new board position.

**Novelty Statement (1 Sentence):**  
> *We present the first controlled empirical-outcome study of rule-based in-context learning in custom chess variants—directly comparing pretrained causal LLM adaptation (`Qwen2.5-7B-Instruct` LoRA) against contamination-free $\mu$P-parameterized from-scratch transformers across four parameter scales (`1.40M`–`206.36M`)—backed by formally audited variant simulators, strictly paired query evaluation, causal intervention controls, and hierarchical cluster-bootstrap uncertainty.*

To isolate genuine rule inference from surface heuristics, our benchmark makes four methodological advances over prior game-ICL setups:
1. **Empirical Rollout Targets Replace Static Heuristics:** Rather than labeling query boards via standard-chess material differences (which are blind to movement, capture, and board-barrier changes), ground-truth labels $y(s_q, v; \pi, H) \in \Delta^2$ are estimated directly from $R$ independent continuations under each variant's audited simulator $(T_v, O_v)$.
2. **Formally Audited Variant Engine:** A unified `VariantBoard(chess.Board)` suite covering `standard`, $4$ movement-rule training variants, $9$ held-out single variants (spanning movement, capture, board barriers, piece setup, and objective overrides), and $3$ compositional variants, verified via $300$-position differential fuzzing and pure-Python reference checks.
3. **Dual-Track Contamination-Free Comparison:** Track 1 adapts `Qwen2.5-7B-Instruct` (LoRA + 3-way head) to test whether pretrained sequence priors aid rule induction; Track 2 trains from-scratch decoder-only transformers (`TINY` `1.40M` to `LARGE` `206.36M`) under Maximal Update Parametrization ($\mu$P) with zero prior exposure to held-out chess variants.
4. **Paired Context Gain & Causal Interventions:** All evaluations measure calibrated proper scoring rules (soft-target NLL, Brier, ECE) and Context Gain $\Delta \mathrm{NLL}(K) = \mathrm{NLL}(0) - \mathrm{NLL}(K)$ on **identical query roots**, validated against mismatched-rule contexts, inverted/scrambled outcomes, shuffled trajectories, a setup-only null control (`knight_army`), and an equal-material slice ($|\Delta M| \le 2$).

---

## 2. Background

### 2.1 Variant Games as Latent-Rule Markov Processes
Each chess variant $v \in \mathcal{V}$ defines a deterministic transition kernel $T_v(s, a) \mapsto s'$ over legal actions $a \in \mathcal{A}_v(s)$ and a terminal outcome function $O_v(s) \in \{W, D, L, \bot\}$ (White-perspective Win, Draw, Loss, or non-terminal $\bot$). Under a fixed rollout policy $\pi(a \mid s; v)$ and horizon $H$, any board state $s_q$ induces a well-defined outcome distribution over trajectories $\tau = (s_0 = s_q, a_0, \dots, s_T)$ with $T \le H$. Because two variants $v \neq v'$ can share identical board syntax while differing in legal move generation, check detection, capture rules, or win conditions, evaluating $s_q$ without knowing $v$ requires inferring $(T_v, O_v)$ from demonstrations $D_{1:K}$.

### 2.2 In-Context Learning as Latent Rule Posterior Prediction
In Bayesian ICL framing, a sequence model parameterized by $\theta$ observes $K$ complete demonstration games $D_1, \dots, D_K \sim \mathcal{D}^{\text{demo}}_v$ (without any rule name or natural-language rule description) followed by query state $s_q$, and outputs calibrated probabilities $\hat{q}_\theta(s_q \mid D_{1:K}) \in \Delta^2$. An ideal Bayesian predictor marginalizes over the posterior $P(v \mid D_{1:K})$:
$$q^\star(s_q \mid D_{1:K}) = \sum_{v \in \mathcal{V}} P(v \mid D_{1:K})\, y(s_q, v; \pi, H).$$
As $K$ increases, demonstrations disambiguate $(T_v, O_v)$, concentrating $P(v \mid D_{1:K})$ and monotonically reducing expected cross-entropy NLL relative to the $K=0$ prior.

### 2.3 Maximal Update Parametrization ($\mu$P) and RoPE
In standard transformer parameterization, scaling model width $d_{\text{model}}$ causes attention logits and residual updates to drift or explode at fixed learning rates, confounding scaling laws with hyperparameter mistuning. Maximal Update Parametrization ($\mu$P; Yang et al., 2022) rescales initialization variance, attention temperature, learning rates, and readout magnitude as functions of width multiplier $r = d_{\text{model}} / d_{\text{base}}$ so internal feature updates remain $\Theta(1)$ across all widths, enabling direct transfer of learning rate $\eta_{\text{base}}$ from `TINY` ($1.40\text{M}$) to `LARGE` ($206.36\text{M}$). Combining $\mu$P with Rotary Position Embeddings (RoPE; Su et al., 2021) enables length-generalizable relative attention over long $K=16$ multi-game contexts up to $8{,}192$ tokens.

---

## 3. Related Work (Compare & Contrast)

| Work | Setting & Objective | How Our Work Differs |
|---|---|---|
| **Brown et al. (2020); Garg et al. (2022); Akyürek et al. (2023)** | *In-context learning of functions:* GPT-3 few-shot NLP; transformers trained from scratch to regress linear/nonlinear functions $x \mapsto f(x)$ or algorithmic tasks from $(x_i, f(x_i))$ pairs. | Prior synthetic ICL benchmarks provide explicit input-output function pairs over flat vectors. Our setting requires inducing **hidden combinatorial transition & check/terminal rules** $(T_v, O_v)$ from unlabeled-rule game trajectories $D_{1:K}$ and applying them to evaluate a separate query board $s_q$. |
| **Ruoss et al. (2024)** (*Grandmaster-Level Chess Without Search*) | *Amortized search-free chess:* Trains transformers on millions of Stockfish-labeled positions under fixed, known FIDE chess rules to predict action-values and win probabilities without search. | Ruoss et al. amortize evaluation for a **single known rule set**. We ask whether transformers can adapt their position evaluation **in context** when the rules themselves change at test time across unseen held-out variants. |
| **Silver et al. (2018)** (*AlphaZero*); **Schrittwieser et al. (2020)** (*MuZero*) | *Self-play RL + MCTS:* Learns policy/value networks (and latent dynamics in MuZero) via millions of self-play games and tree search per game. | AlphaZero/MuZero require gradient-based RL retraining and test-time search for each new game. We study **zero-search, zero-gradient few-shot adaptation** from $K \le 16$ observed games. |
| **Toshniwal et al. (2022); Li et al. (2023)** (*Othello-GPT*); **Karvonen (2024)** | *Board world models:* Trains sequence models on move logs in standard Othello/chess to probe emergent internal board-state representations and next-move legality. | World-model probing evaluates whether a model tracks state under a **fixed known rule set**. We evaluate **cross-rule generalization**: inferring altered movement, capture, barrier, and win-condition rules purely from context and measuring calibrated value shifts. |
| **Yang et al. (2022)** (*Tensor Programs V / $\mu$P*) | *Width-invariant hyperparameter transfer:* Derives scaling laws for initialization, attention, and AdamW learning rates across transformer widths. | We apply $\mu$P directly to from-scratch chess-variant ICL (`1.40M`–`206.36M`) and verify $O(1)$ 1-step logit/feature coordinate stability across an $8\times$ width expansion. |

---

## 4. Self-Contained Method & Mathematical Formulation

### 4.1 Empirical-Outcome Ground Truth & Perspective Convention
Let $\mathcal{V}$ be the set of chess variants, where each variant $v \in \mathcal{V}$ specifies transition function $T_v$ and terminal outcome mapping $O_v$. Given a query state $s_q$ with side-to-move $c_q \in \{\text{White}, \text{Black}\}$, declared rollout policy $\pi$, and horizon $H$, we roll out $R$ independent continuations $\tau_1, \dots, \tau_R$ from $s_q$ under $(T_v, \pi)$ up to $H$ plies. The ground-truth target distribution from $c_q$'s perspective is:
$$y(s_q, v; \pi, H) = \big(y_W, y_D, y_L\big) = \frac{1}{R}\sum_{r=1}^{R} \mathbf{e}_{O_v^{(c_q)}(\tau_r)} \in \Delta^2,$$
where index $0$ denotes Win, $1$ denotes Draw, and $2$ denotes Loss for $c_q$.

Each demonstration game $D_k = (s_{k,0}, a_{k,1}, \dots, a_{k,L_k}, o_k)$ records initial state $s_{k,0}$ (including barrier squares rendered explicitly as `'X'`), UCI move sequence, and realized **White-perspective** outcome $o_k \in \{W, D, L\}$. Any predictor mapping White-perspective counts to query predictions applies the side-to-move involution:
$$\Phi_{\text{White}}(p_W, p_D, p_L) = (p_W, p_D, p_L), \qquad \Phi_{\text{Black}}(p_W, p_D, p_L) = (p_L, p_D, p_W).$$

### 4.2 Proper Scoring Rules, Calibration & Context Gain
For $N$ evaluation queries with predictions $\hat{p}_i \in \Delta^2$ and targets $y_i \in \Delta^2$:

1. **Soft-Target Cross-Entropy (NLL):**
   $$\mathrm{NLL}_i = -\sum_{c=0}^{2} y_{i,c}\,\log\,\mathrm{clip}(\hat{p}_{i,c}, \varepsilon, 1.0), \qquad \overline{\mathrm{NLL}} = \frac{1}{N}\sum_{i=1}^{N}\mathrm{NLL}_i \quad (\varepsilon = 10^{-7}).$$
2. **Multiclass Brier Score:**
   $$\mathrm{Brier} = \frac{1}{N}\sum_{i=1}^{N}\sum_{c=0}^{2}(\hat{p}_{i,c} - y_{i,c})^2 \in [0, 2].$$
3. **Top-Label & Classwise ECE:** With $\hat{c}_i = \arg\max_c \hat{p}_{i,c}$, $\mathrm{conf}_i = \hat{p}_{i,\hat{c}_i}$, $a_i = y_{i,\hat{c}_i}$, and $M=10$ uniform bins $B_1, \dots, B_M$ over $[0, 1]$:
   $$\mathrm{ECE}_{\text{top}} = \sum_{b=1}^{M}\frac{|B_b|}{N}\left|\frac{1}{|B_b|}\sum_{i\in B_b} a_i - \frac{1}{|B_b|}\sum_{i\in B_b}\mathrm{conf}_i\right|, \qquad \mathrm{ECE}_{\text{class}} = \frac{1}{3}\sum_{c=0}^{2}\sum_{b=1}^{M}\frac{|B_{b,c}|}{N}\left|\overline{y}_{B_{b,c},c} - \overline{p}_{B_{b,c},c}\right|.$$
4. **Context Gain (Headline Metric):** Evaluated strictly on **identical query roots** across $K=0$ and $K>0$:
   $$\Delta \mathrm{NLL}_i(K) = \mathrm{NLL}_i(0) - \mathrm{NLL}_i(K), \qquad \overline{\Delta\mathrm{NLL}}(K) = \frac{1}{|\mathcal{V}_{\text{eval}}|}\sum_{v \in \mathcal{V}_{\text{eval}}}\frac{1}{|I_v|}\sum_{i \in I_v}\Delta\mathrm{NLL}_i(K).$$
5. **Irreducible Monte Carlo Noise Floor:** For $R$-rollout empirical targets $y_i$ with effective support $k_{\mathrm{eff},i} = |\{c : y_{i,c} > 0\}|$:
   $$H(y) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c: y_{i,c}>0} y_{i,c}\log y_{i,c}, \qquad H_{\text{debiased}} = H(y) + \frac{1}{N}\sum_{i=1}^{N}\frac{k_{\mathrm{eff},i}-1}{2R}.$$

### 4.3 Track 1: Pretrained LLM Adaptation (`Qwen2.5-7B-Instruct`)
Track 1 freezes the BF16 `Qwen2.5-7B-Instruct` backbone (`hidden_size = 3584`, `28` layers, GQA `28/4` heads) and attaches low-rank LoRA adapters (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`) plus a trainable FP32 3-way linear readout head $W_{\text{head}} \in \mathbb{R}^{3 \times 3584}$ at the final non-pad token hidden state $h_{\text{last}}$. Prompts serialize $D_{1:K}$ (initial 8x8 ASCII grid with `'X'` barriers, moves, White-perspective outcome) followed by $s_q$ without variant names or rule descriptions, guaranteeing the query block is never truncated when length exceeds `max_length`. Training minimizes soft-target cross-entropy over the 3-way head:
$$\mathcal{L}(\theta) = -\sum_{c=0}^{2} y_c \log \mathrm{softmax}\big(W_{\text{head}} h_{\text{last}}(\theta)\big)_c.$$

### 4.4 Track 2: Compact Tokenizer & From-Scratch $\mu$P-RoPE Transformer
To eliminate pretraining contamination and evaluate $K=16$ full-game contexts inside `max_seq_len = 8192`, Track 2 introduces:

1. **Compact Vocabulary ($V = 4{,}705$):**
   - **Fixed 70-token board state:** $64$ fixed square tokens in FEN order (`a8..h1`) from `{., X, P, N, B, R, Q, K, p, n, b, r, q, k}`, $1$ side-to-move token (`SIDE_W`/`SIDE_B`), $4$ castling-flag slots, and $1$ en-passant target token.
   - **1-token UCI moves ($4{,}608$ tokens):** All $64 \times 64 = 4{,}096$ `(from, to)` pairs plus all $8 \times 8 \times 2 \times 4 = 512$ pawn promotions (`7->8`, `2->1` for `{q, r, b, n}`).
   - **No-query-truncation episode layout:** `[CLS]` (seg $0$), each demo's `[DEMO_START] [STATE] <70> [MOVES] <moves>` (seg $2k-1$), each demo's `[OUT_*] [DEMO_END]` (seg $2k$), and `[QUERY] [STATE] <70> [SEP]` (seg $2K+1$).

2. **Rotary Position Embeddings (RoPE, $\theta = 10{,}000$):**
   For head dimension $d_{\text{head}} = d_{\text{model}} / n_{\text{heads}}$ and frequencies $\theta_i = 10000^{-2i/d_{\text{head}}}$, queries and keys at token position $t$ are rotated in `float32`:
   $$\tilde{q}_t = q_t \odot \cos(t\theta) + \mathrm{rotate\_half}(q_t) \odot \sin(t\theta), \qquad \tilde{k}_t = k_t \odot \cos(t\theta) + \mathrm{rotate\_half}(k_t) \odot \sin(t\theta).$$

3. **Exact $\mu$P Scaling ($d_{\text{base}} = 128$, $d_{\text{base,head}} = 32$, $r = d_{\text{model}} / d_{\text{base}}$):**
   - Attention logit scale: $\alpha_{\text{attn}} = \sqrt{d_{\text{base,head}}} / d_{\text{head}}$.
   - Initialization: `token_emb`, `head` $\sim \mathcal{N}(0, 0.02^2)$; hidden linear weights inside transformer blocks $\sim \mathcal{N}(0, (0.02/\sqrt{r})^2)$.
   - Readout multiplier at final non-pad `[SEP]` hidden state: $\hat{z} = \frac{1}{r} W_{\text{head}}\,\mathrm{LN}_f(h_{\text{last}})$.
   - AdamW learning rates: $\eta_{\text{hidden}} = \eta_{\text{base}} / r$; $\eta_{\text{emb}} = \eta_{\text{LN}} = \eta_{\text{head}} = \eta_{\text{base}}$.

4. **Attention Modes (Causal vs. Segment-Bidirectional Ablation):**
   Let $v_j \in \{0, 1\}$ be the non-pad mask and $s_i$ be token $i$'s segment index:
   - `causal`: $i$ attends to $j$ iff $j \le i \land v_j = 1$.
   - `segment_bidirectional`: $i$ attends to $j$ iff $v_j = 1 \land \big((s_j = s_i) \lor (s_j < s_i \land j < i)\big)$, allowing bidirectional encoding within each board/demo segment while strictly preventing outcome tokens (`[OUT_*]`) or query tokens from leaking into earlier segments.

5. **Two-Stage Training Curriculum & Model Selection:**
   Models first undergo **standard chess warmup** (`mode="warmup"`) on `standard` chess roots across $K \in \{0, 1, 2, 4, 8, 16\}$, then **variant-mixture continuation** (`mode="continue"`) over the $4$ movement training variants (`diagonal_rook`, `straight_bishop`, `weakened_queen`, `extended_pawn`). Checkpoints (`best.pt`) are selected strictly by lowest macro validation NLL.

### 4.5 Hierarchical Uncertainty Quantification
Because multiple query roots $s_q$ can be sampled from the same underlying source game $g = \mathrm{source\_game\_id}(s_q)$, treating roots as i.i.d. underestimates variance. For each variant $v$, we average per-root metrics across $S$ model seeds and $M=5$ demonstration draws to obtain $\bar{m}_{v, r}$, group roots into clusters $\{C_{v, g}\}_{g=1}^{G_v}$ by `source_game_id`, and run a **stratified cluster bootstrap** ($B = 1{,}000$ replicates resampling $G_v$ clusters with replacement per variant) to form $95\%$ percentile CIs for both per-variant and macro-averaged NLL and paired Context Gain $\overline{\Delta\mathrm{NLL}}(K)$. Combined standard error propagates both game-sampling and model-seed variance: $\mathrm{SE}_{\text{combined}} = \sqrt{\mathrm{SE}_{\text{boot}}^2 + \mathrm{SE}_{\text{seed}}^2}$.

---

## 5. Hyperparameter Justification Table

| Component | Parameter | Chosen Value / Options | Justification |
|---|---|---|---|
| **Rollout Protocol** | Policy $\pi$ | `[TBD Stage 0]` (candidate: `uniform_legal_v1`, `terminal_aware_v1`, `search_d1_v1`) | Selected in Stage 0 to maximize outcome-distribution diversity without degenerate draw saturation at tractable CPU cost. |
| **Rollout Protocol** | Horizon $H$ & Rollouts $R$ | $H = 320$ plies, $R = \text{[TBD Stage 0]}$ | $H=320$ bounds endgame cost; $R$ balances empirical label variance ($\propto 1/R$) against generation budget. |
| **Eval Protocol** | Context sizes $K$ | $K \in \{0, 1, 2, 4, 8, 16\}$ | Spans zero-shot baseline ($K=0$) through multi-game rule disambiguation ($K=16$) within 8k/32k context windows. |
| **Eval Protocol** | Demo draws & Bootstrap | $5$ demo draws/$K$, $B = 1{,}000$ cluster bootstrap | Averages out demonstration draw luck; $B=1{,}000$ yields stable $95\%$ CIs clustered by `source_game_id`. |
| **Equal-Mat Slice** | Material threshold | $|\Delta M_{\text{White}}| \le 2$ | Isolates positions where static piece-count heuristics are uninformative, forcing reliance on variant rules. |
| **Track 1 (Qwen)** | Base model | `Qwen/Qwen2.5-7B-Instruct` (BF16, rev `a09a3545...`) | Locally cached, $32\text{k}$ native context window, strong sequence priors, fits comfortably on 1x H100 (`29.6 GB` at `seq_len=24576`). |
| **Track 1 (Qwen)** | LoRA config | `r=16, lora_alpha=32`, all 28 blocks (`q/k/v/o/gate/up/down`) (`handoff` mode: `r=8`, last 4 blocks) | Standard high-capacity adaptation of attention + MLP projections with $<0.6\%$ trainable parameters. |
| **Track 1 (Qwen)** | Move notation | `[TBD Stage 0]` (`UCI` vs `SAN`) | Chosen in Stage 0 by comparing tokenized game length and token regularity under Qwen BPE. |
| **Track 2 (From-Scratch)** | Model ladder | `TINY` (1.40M, $d=128, L=4$), `SMALL` (7.52M, $d=256, L=8$), `MEDIUM` (23.10M, $d=384, L=12$), `LARGE` (206.36M, $d=1024, L=16$) | Spans $>140\times$ parameter range cleanly with $d_{\text{head}} \in \{32, 48, 64\}$ (even for RoPE). |
| **Track 2 (From-Scratch)** | $\mu$P & RoPE | $d_{\text{base}}=128$, $d_{\text{base,head}}=32$, $\theta=10{,}000$, `max_seq_len=8192` | Verified $O(1)$ 1-step logit updates (`0.7912`–`0.8164`) across $8\times$ width expansion; fits $K=16$ full games. |
| **Track 2 (From-Scratch)** | Optimizer & Schedule | AdamW (`wd=0.01`, grad clip `1.0`), linear warmup + cosine decay (`0.1` floor), $\eta_{\text{base}} = \text{[TBD Stage 0]}$ | Standard stable transformer optimization with $\mu$P $\eta_{\text{base}}/r$ hidden-layer scaling. |
