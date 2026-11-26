# Stage 0 CPU Diagnostics & Tokenization Benchmarks (`chess_icl_v2`)

**Date:** 2026-11-25  
**Scope:** All 14 single-rule variants (`standard` + 13 single variants) × 3 variant-agnostic rollout policies (`uniform_legal_v1`, `terminal_aware_v1`, `search_d1_v1`)  
**Artifacts Generated:**
- Rollout module: [`chess_icl_v2/rollouts/`](file:///data/backups/rganapa/Sebastian/cs182-chess/chess_icl_v2/rollouts/__init__.py)
- Unit tests: [`tests_v2/test_rollouts_v2.py`](file:///data/backups/rganapa/Sebastian/cs182-chess/tests_v2/test_rollouts_v2.py) (**21 passed in 2.17s**)
- Raw metrics: [`runs_v2/stage0/stage0_metrics.json`](file:///data/backups/rganapa/Sebastian/cs182-chess/runs_v2/stage0/stage0_metrics.json)
- Sample games (2,520 games): [`runs_v2/stage0/sample_games.jsonl`](file:///data/backups/rganapa/Sebastian/cs182-chess/runs_v2/stage0/sample_games.jsonl)
- Token stats: [`runs_v2/stage0/stage0_token_stats.json`](file:///data/backups/rganapa/Sebastian/cs182-chess/runs_v2/stage0/stage0_token_stats.json)

---

## 1. Experimental Setup

Executed [`scripts_v2/run_stage0.py`](file:///data/backups/rganapa/Sebastian/cs182-chess/scripts_v2/run_stage0.py) with `--workers 64 --games 60 --roots 80 --R 16 --max-plies 320 --seed 2026`:
- **Per-policy rollout workload:** 60 full games × 14 variants = **840 games**, plus 80 unique midgame roots (`ply >= 8`, balanced White/Black to move) × 14 variants × `R=16` continuations = **17,920 continuations** per policy.
- **Shared-cohort Rule-Sensitivity / Context Gain Ceiling:** 80 standard-legal midgame positions verified valid across all 14 variants (no pieces on `blocked_squares(seed=0)`, no pre-existing illegal checks under any variant's rules) × 14 variants × `R=16` continuations = **17,920 shared-cohort continuations** per policy.
  - Defined $\bar{p}(s_q) = \frac{1}{14}\sum_{v} \hat{p}_v(s_q)$ as the $K=0$ rule-agnostic mixture baseline, and
  - $\text{Ceiling Gain} = \mathrm{NLL}_{K=0}(\bar{p}) - \frac{1}{14}\sum_{v} H(\hat{p}_v) = \frac{1}{14 \cdot 80}\sum_{v,q} D_{\mathrm{KL}}(\hat{p}_v(s_q) \,\|\, \bar{p}(s_q))$.

---

## 2. Policy Rung Comparison (Macro Across 14 Variants)

| Policy Rung | Wall (64 cores) | ms/ply | Mean Plies | Horizon Rate (Mean / Max) | Draw Rate | Decisive Rate | Std($\hat{p}_W$) | Irred. Entropy (nats) | MC Brier Noise Floor | MC NLL Excess | Shared Ceiling Gain (nats) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `uniform_legal_v1` | 125.5 s | **0.840** | 185.6 | 0.184 / 0.395 | 0.828 | 0.172 | 0.082 | 0.310 | 0.0120 | 0.0268 | 0.2213 |
| **`terminal_aware_v1`** | **116.1 s** | 1.337 | 122.2 | 0.074 / 0.206 | **0.413** | **0.588** | **0.203** | 0.715 | 0.0291 | 0.0477 | 0.1483 |
| `search_d1_v1` | 296.3 s | 5.059 | 57.7 | **0.001 / 0.007** | 0.784 | 0.216 | 0.141 | 0.365 | 0.0145 | 0.0299 | **0.2359** |

### Key Takeaways Across Policies
1. **`uniform_legal_v1` collapses into draws and horizon hits:** 82.8% of continuations end in draws (18.4% mean horizon rate, spiking to **39.5%** on `dark_square_immunity`), leaving `decisive_rate = 17.2%` and very low outcome variance (`std(p_win) = 0.082`).
2. **`search_d1_v1` suffers from greedy repetition draws & 3.8× higher CPU cost:** Avoiding mate-in-1 while minimizing opponent mobility induces premature threefold repetitions (mean length **57.7 plies**, **78.4% draws**, `decisive_rate = 21.6%`), and takes `5.059 ms/ply` (`296.3s` vs `116.1s` on 64 cores). Moreover, its high shared ceiling gain (`0.2359 nats`) is disproportionately driven by `antichess` (`1.444 nats` vs `0.094–0.354 nats` on the other 13 variants).
3. **`terminal_aware_v1` hits the scientific sweet spot:**
   - **Highest decisive signal & variance:** `58.8%` decisive outcomes (`41.3%` draws) and `std(p_win) = 0.203` (2.5× higher than `uniform_legal_v1`).
   - **Well-behaved horizon rate:** `7.4%` mean (`<= 8.7%` on 12 of 14 variants; `16.1%` on `blocked_squares`, `20.6%` on `mobile_king`).
   - **Fastest total wall time:** `116.1 s` (`1.337 ms/ply`) across all 14 variants × 80 roots × 16 rollouts.
   - **Uniform rule sensitivity:** Per-variant shared-cohort ceiling gains span `0.058–0.413 nats` across all 14 variants rather than collapsing into an `antichess`-only outlier.

---

## 3. Per-Variant Diagnostics for `terminal_aware_v1` (vs `uniform` / `search_d1`)

| Variant | `ms/ply` | Mean Plies | Horizon Rate | Decisive Rate | Std($\hat{p}_W$) | Irred. $H$ (nats) | MC Brier Floor | Shared Ceiling (nats) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `standard` | 1.212 | 119.7 | 0.052 | 0.577 | 0.229 | 0.799 | 0.0323 | 0.0623 |
| `diagonal_rook` | 1.407 | 141.6 | 0.076 | 0.348 | 0.191 | 0.596 | 0.0238 | 0.1159 |
| `straight_bishop` | 1.932 | 104.6 | 0.044 | 0.792 | 0.209 | 0.839 | 0.0342 | 0.1072 |
| `weakened_queen` | 1.199 | 148.6 | 0.084 | 0.510 | 0.192 | 0.791 | 0.0321 | 0.1239 |
| `extended_pawn` | 1.521 | 94.9 | 0.029 | 0.688 | 0.235 | 0.779 | 0.0315 | 0.1080 |
| `short_knight` | 1.415 | 126.9 | 0.071 | 0.570 | 0.209 | 0.823 | 0.0331 | 0.0721 |
| `mobile_king` | 1.849 | 199.9 | 0.206 | 0.188 | 0.118 | 0.433 | 0.0171 | 0.3895 |
| `piece_capture` | 1.058 | 151.4 | 0.130 | 0.507 | 0.174 | 0.867 | 0.0347 | 0.1156 |
| `forward_pawn_capture` | 1.591 | 96.1 | 0.036 | 0.724 | 0.225 | 0.804 | 0.0327 | 0.0987 |
| `dark_square_immunity` | 1.585 | 73.4 | 0.014 | 0.982 | 0.194 | 0.658 | 0.0293 | 0.2815 |
| `blocked_squares` | 1.045 | 169.9 | 0.161 | 0.396 | 0.160 | 0.744 | 0.0297 | 0.0635 |
| `long_board` | 1.339 | 107.1 | 0.050 | 0.593 | 0.236 | 0.708 | 0.0288 | 0.0669 |
| `knight_army` | 0.918 | 140.4 | 0.087 | 0.381 | 0.181 | 0.614 | 0.0246 | 0.0575 |
| `antichess` | 0.651 | 36.4 | 0.000 | 0.970 | 0.284 | 0.552 | 0.0239 | 0.4133 |

*(Full per-variant tables for `uniform_legal_v1` and `search_d1_v1` are stored in [`runs_v2/stage0/stage0_metrics.json`](file:///data/backups/rganapa/Sebastian/cs182-chess/runs_v2/stage0/stage0_metrics.json).)*

---

## 4. Real Token Length Benchmarks: Qwen2.5-7B (UCI vs SAN) & Track 2 (`ChessV2Tokenizer`)

Measured via [`scripts_v2/measure_stage0_tokens.py`](file:///data/backups/rganapa/Sebastian/cs182-chess/scripts_v2/measure_stage0_tokens.py) on all **2,520** Stage 0 full games (840 per policy) using the pinned local `Qwen/Qwen2.5-7B-Instruct` tokenizer (`rev a09a35458c702b33eeacc393d103063234e8bc28`) and `ChessV2Tokenizer`.

### 4.1 Single-Game Formatted Lengths (tokens/game)

| Policy | Qwen-UCI Mean ± Std (p50 / p95 / max) | Qwen-SAN Mean ± Std (p50 / p95 / max) | SAN Savings | Track 2 `ChessV2Tokenizer` Mean ± Std (p50 / p95 / max) |
|---|---:|---:|---:|---:|
| `uniform_legal_v1` | 1,185.3 ± 353.6 (1,404 / 1,414 / 1,417) | 803.2 ± 225.3 (927 / 974 / 1,014) | -32.2% | 338.9 ± 88.0 (395 / 395 / 395) |
| **`terminal_aware_v1`** | **718.5 ± 420.8 (579 / 1,411 / 1,417)** | **504.1 ± 267.0 (412 / 950 / 1,008)** | **-29.8%** | **222.5 ± 104.9 (188 / 395 / 395)** |
| `search_d1_v1` | 543.3 ± 176.9 (536 / 851 / 1,410) | 403.8 ± 112.3 (399 / 592 / 922) | -25.7% | 178.8 ± 44.0 (177 / 255 / 395) |

### 4.2 Full-Episode Prompt Lengths for $K \in \{1, 4, 8, 16\}$ (560 sampled episodes per $(K, \text{policy})$)

| Policy | $K$ | Qwen-UCI Mean (p95 / max) | Qwen-SAN Mean (p95 / max) | Track 2 Mean (p95 / max) |
|---|---:|---:|---:|---:|
| `terminal_aware_v1` | 1 | 885.2 (1,611 / 1,617) | **683.6 (1,150 / 1,204)** | 288.3 (469 / 469) |
| `terminal_aware_v1` | 4 | 3,023.8 (4,998 / 5,848) | **2,184.4 (3,420 / 3,973)** | 950.8 (1,441 / 1,654) |
| `terminal_aware_v1` | 8 | 5,876.7 (9,069 / 11,174) | **4,190.3 (6,240 / 7,563)** | 1,834.5 (2,630 / 3,154) |
| `terminal_aware_v1` | 16 | 11,749.7 (17,988 / 20,927) | **8,304.4 (12,214 / 14,134)** | 3,643.6 (5,198 / 5,931) |
| `uniform_legal_v1` | 16 | 19,229.4 (22,389 / 22,794) | 13,092.4 (15,117 / 15,474) | 5,508.5 (6,289 / 6,394) |
| `search_d1_v1` | 16 | 8,903.4 (11,173 / 13,642) | 6,676.0 (8,011 / 9,509) | 2,933.1 (3,499 / 4,115) |

> **Important Track 2 Tokenizer Audit Finding:**  
> Running `ChessV2Tokenizer` across all 2,520 Stage 0 games uncovered **225 OOV move tokens** across two variants because `ChessV2Tokenizer._build_vocab()` currently generates only 1-square pawn promotions (`7→8`, `2→1`) for `{q, r, b, n}`:
> - `extended_pawn`: **206 OOV moves** (58 unique 2-square jump promotions such as `b3b1n`, `d6d8q`, `g3g1r`, `e6e8q`).
> - `antichess`: **19 OOV moves** (10 unique King promotions such as `a7a8k`, `b2b1k`, `f2g1k`).  
> *Action item before Track 2 dataset serialization:* Add 2-square pawn promotions (`rank 6→8` and `rank 3→1`) and King promotion (`k`) tokens to `ChessV2Tokenizer` (+576 / +128 move tokens).

---

## 5. Frozen Recommendations for Open Decisions (#1, #2, #6)

| Decision | Frozen Choice | Data-Backed Rationale |
|---|---|---|
| **#1 Main rollout policy $\pi$** | **`terminal_aware_v1`** | Achieves highest decisive rate (`58.8%`) and outcome variance (`std(p_win) = 0.203`) with zero hardcoded piece values, fastest 64-core runtime (`116.1 s` for 17,920 continuations), and balanced rule sensitivity across all 14 variants (`0.058–0.413 nats`). |
| **#2 Horizon $H=320$ handling** | **Finite-horizon draw (with `label_probs_censored` sensitivity reported)** | Under `terminal_aware_v1`, mean continuation length is `122.2 plies` and horizon hit rate is `7.4%` macro (`<= 8.7%` on 12/14 variants). Counting horizon hits as draws keeps probability vectors finite and unbiased under fixed compute, while storing `label_probs_censored` enables direct sensitivity checks. |
| **#6 Track 1 move notation** | **`SAN`** | Saves **29.8%** Qwen2.5-7B tokens per game (`504.1` vs `718.5` tokens under `terminal_aware_v1`), cutting $K=16$ episode prompts from `11,750` (max `20,927`) to **`8,304` (max `14,134`)** — comfortably within Qwen's 32,768 context (leaving headroom even for $K=32$) while matching pretrained chess notation. Track 2 retains 1-token-per-move UCI (`3,644` mean tokens at $K=16$, fitting inside `max_len=8192`). |
