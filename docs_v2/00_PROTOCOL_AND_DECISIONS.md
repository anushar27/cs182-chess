# Chess-Variant ICL v2: Protocol, Architecture & Open Decisions

**Date:** 2026-11-25  
**Package:** `chess_icl_v2` (leaves v1 `chess_icl` untouched)  
**Target Standard:** CS 182/282 NeurIPS-format Research Paper

---

## 1. Research Questions

1. **Track 1 (Pretrained LLM Adaptation):** Can a pretrained causal LLM (`Qwen2.5-7B-Instruct`, LoRA-adapted with a 3-way probability head) infer unfamiliar chess-variant rules solely from $K$ full-game demonstrations and improve its empirical W/D/L probability predictions on identical query roots (measured by macro context gain $\Delta \mathrm{NLL}(K) = \mathrm{NLL}(0) - \mathrm{NLL}(K)$)?
2. **Track 2 (Contamination-Free Emergence & Scaling):** Does in-context rule learning emerge from scratch in $\mu$P-parameterized decoder-only transformers (`TINY` 1.3M through `LARGE` 205M) after standard-chess warmup + variant mixture training, and how does context gain scale with model width/depth on strictly uncontaminated held-out variants?

---

## 2. Why v2 Replaces v1 (Post-Mortem of v1 Flaws)

| Flaw in v1 | Impact on Scientific Validity | Fix in v2 |
|---|---|---|
| Heuristic static material labels (`>= +3` = win) | Labels did not reflect variant rules (e.g., `diagonal_rook`, `blocked_squares`, `dark_square_immunity` all had identical material labels to standard chess!) | Empirical W/D/L outcome probability vector $p \in \Delta^2$ estimated from $R$ fresh simulator continuations under declared policy $\pi$ and horizon $H$. |
| Standard `is_into_check` / `is_game_over` inside variants | `DiagonalRook`, `StraightBishop`, `WeakenedQueen`, `MobileKing`, `ShortKnight`, `ForwardPawnCapture`, `BlockedSquares`, `DarkSquareImmunity` detected checks using standard movement rules. | `VariantBoard(chess.Board)` overrides `attackers_mask`, `checkers_mask`, `is_into_check`, `generate_legal_moves`, `outcome`, and passes differential fuzzing + property audits. |
| `KnightArmyBoard` overwrote any passed FEN | Loading midgame positions silently reset the board to 15 knights. | Custom starting army applied ONLY when `fen is None`. |
| `BlockedSquaresBoard` seed drift & invisible barriers | Blocked squares changed when loading midgame FENs and were absent from FEN strings. | Blocked squares sampled deterministically over ranks 3–6 (`a3..h6`), preserved in canonical state keys and rendered as `X` in state representations. |
| `ShortKnight` offset mismatch | `ShortKnightBoard` used ferz+dabbaba while compositional `ShortKnightAntichessBoard` used ferz only. | Frozen ferz+dabbaba specification shared everywhere. |
| Demo outcomes only + 40-move truncation + argmax accuracy | Accuracy hides calibration; 40-move truncation hides endgame rules; no mismatched-rule or kNN controls in main eval. | Full-game demos, soft-target cross-entropy (NLL), Brier score, Context Gain $\Delta \mathrm{NLL}$ on identical roots, and all 11 merged baselines/controls. |

---

## 3. Formal Problem Formulation

Let $v \in \mathcal{V}$ denote a chess variant with transition function $T_v$ and terminal outcome mapping $O_v$. Given a query state $s_q$ with player-to-move $c_q \in \{\text{White}, \text{Black}\}$, a declared rollout policy $\pi$, and horizon $H$, the ground-truth label is the empirical outcome distribution from $c_q$'s perspective:
$$y(s_q, v; \pi, H) = \left( \hat{p}_W, \hat{p}_D, \hat{p}_L \right) = \frac{1}{R} \sum_{r=1}^{R} \mathbf{e}_{O_v^{(c_q)}(\tau_r)}$$
where $\tau_1, \dots, \tau_R$ are $R$ independent continuations rolled out from $s_q$ under $(T_v, \pi)$ up to $H$ plies.

An ICL episode for context size $K$ consists of $K$ complete demonstration games $D_1, \dots, D_K \sim \mathcal{D}^{\text{demo}}_v$, each formatted as initial state, move sequence, and White-perspective realized outcome $o_k \in \{W, D, L\}$, followed by query state $s_q$. The model outputs calibrated probabilities $\hat{q}_\theta(s_q \mid D_{1:K}) \in \Delta^2$ trained via soft-target cross-entropy:
$$\mathcal{L}(\theta) = -\sum_{j \in \{W, D, L\}} y_j \log \hat{q}_{\theta, j}(s_q \mid D_{1:K}).$$

**Headline Metric (Context Gain):** For identical query roots $\{s_q\}$ across context sizes:
$$\Delta \mathrm{NLL}_v(K) = \mathrm{NLL}_v(0) - \mathrm{NLL}_v(K), \qquad \overline{\Delta \mathrm{NLL}}(K) = \frac{1}{|\mathcal{V}_{\text{eval}}|} \sum_{v \in \mathcal{V}_{\text{eval}}} \Delta \mathrm{NLL}_v(K).$$

---

## 4. Open Decisions Tracker (All 7 Frozen Prior to Main Runs)

| # | Decision | Options Tested / Considered | Frozen Choice & Justification | Status |
|---|---|---|---|---|
| 1 | Main rollout policy $\pi$ | `uniform_legal_v1`, `terminal_aware_v1`, `search_d1_v1` | **`terminal_aware_v1`**: Highest decisive rate (`58.8%` vs `17.2%` uniform / `21.6%` search_d1), highest outcome variance (`std(p_win)=0.203`), fastest Stage 0 runtime (`116.1 s` on 64 cores), balanced shared-cohort sensitivity (`0.058–0.413 nats`). | **FROZEN (Stage 0)** |
| 2 | Horizon $H=320$ handling | Finite-horizon draw vs. censored/unresolved | **Finite-horizon draw** (with `label_probs_censored` saved for sensitivity analysis). Mean horizon hit rate is only `7.4%` (`<= 8.7%` on 12/14 variants) under `terminal_aware_v1`. | **FROZEN (Stage 0)** |
| 3 | Track 2 paper role | Paper claim (contamination-free scaling + emergence) vs debug only | **Both:** Pilot gate before Qwen AND primary contamination-free scaling & emergence claim in paper. | **FROZEN** |
| 4 | Train / Holdout split | 4 Move train / 9 Test (`knight_army` setup-only null control) | **4-to-9 split** (all 17 variants passed differential & property audits in `01_VARIANT_SPECS_AND_AUDITS.md`). `knight_army` (null control) & visibly identifiable (`blocked_squares`, `knight_army`) analyzed separately. | **FROZEN** |
| 5 | Long Board (`long_board`) | 8x20 custom board vs frozen 8x8 Extended Opening Pawn Push approximation | **Frozen 8x8 Extended Opening Pawn Push approximation** so all 13 variants share the exact 64-square state/token space for controlled comparison. | **FROZEN** |
| 6 | Track 1 move notation | UCI vs SAN | **`SAN`** for Track 1 (`29.8%` shorter than UCI: `504.1` vs `718.5` tokens/game; `K=16` mean `8,304`, max `14,134` tokens, fitting in `16,384` without truncation). Track 2 uses 1-token-per-move UCI (`3,644` mean tokens at `K=16`). | **FROZEN (Stage 0)** |
| 7 | Track 1 base model | `Qwen/Qwen2.5-7B-Instruct` (rev `a09a35458c702b33eeacc393d103063234e8bc28`) | **Pinned `Qwen2.5-7B-Instruct`** (local snapshot `a09a35458c702b33eeacc393d103063234e8bc28`, 32,768 native context, BF16, `sdpa`). | **FROZEN** |
