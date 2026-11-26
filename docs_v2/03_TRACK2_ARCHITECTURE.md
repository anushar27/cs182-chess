# Track 2 Architecture: Compact Tokenizer (`ChessV2Tokenizer`) & $\mu$P-RoPE Transformer (`ChessTransformerV2`)

**Date:** 2026-11-25  
**Modules:**
- `chess_icl_v2/data/tokenizer_v2.py` (`ChessV2Tokenizer`, `collate_episodes`)
- `chess_icl_v2/models/small_transformer.py` (`ChessTransformerV2`, `ModelConfigV2`, `RotaryEmbedding`, `TINY`/`SMALL`/`MEDIUM`/`LARGE`)
- `tests_v2/test_models_v2.py`

---

## 1. Compact Vocabulary & Episode Serialization (`ChessV2Tokenizer`)

### 1.1 Vocabulary Breakdown (`V = 4,705`)

| Group | Count | Tokens | Notes |
|---|---:|---|---|
| Special tokens | 11 | `[PAD]` (0), `[CLS]`, `[SEP]`, `[DEMO_START]`, `[DEMO_END]`, `[STATE]`, `[MOVES]`, `[OUT_W]`, `[OUT_D]`, `[OUT_L]`, `[QUERY]` | `[PAD]` fixed at ID `0`. |
| Square occupancy | 14 | `.`, `X`, `P`, `N`, `B`, `R`, `Q`, `K`, `p`, `n`, `b`, `r`, `q`, `k` | `X` explicitly marks impassable blocked squares. |
| Side-to-move | 2 | `SIDE_W`, `SIDE_B` | Separate from piece `'b'` (black bishop) for lossless decoding. |
| Castling flags | 5 | `CASTLE_NONE`, `CASTLE_K`, `CASTLE_Q`, `CASTLE_k`, `CASTLE_q` | Encoded in 4 fixed positional slots. |
| En-passant target | 65 | `EP_NONE`, `EP_a1`..`EP_h8` | Supports standard and variant pawn pushes. |
| 1-token UCI moves | 4,608 | `MOVE_{from}{to}` (4,096) + `MOVE_{f7}{t8}{q,r,b,n}` (256) + `MOVE_{f2}{t1}{q,r,b,n}` (256) | Every UCI move is strictly **1 token**. |
| **Total** | **4,705** | | |

### 1.2 Fixed-Length State Encoding (`70` Tokens)

`encode_state(state, blocked_squares=None)` always produces **exactly 70 tokens**:
1. **Indices `0..63`**: 64 square tokens in FEN traversal order (`a8..h8`, `a7..h7`, ..., `a1..h1`). Any square in `blocked_squares` (or `'X'` in the FEN placement) is encoded as `'X'`.
2. **Index `64`**: Side-to-move (`SIDE_W` or `SIDE_B`).
3. **Indices `65..68`**: 4 castling flag slots (`CASTLE_K`/`CASTLE_NONE`, `CASTLE_Q`/`CASTLE_NONE`, `CASTLE_k`/`CASTLE_NONE`, `CASTLE_q`/`CASTLE_NONE`).
4. **Index `69`**: En-passant target (`EP_NONE` or `EP_<sq>`).

### 1.3 Episode Layout, Segment Assignment, and No-Query-Truncation Guarantee

For $K$ demonstration games $D_1, \dots, D_K$ and query state $s_q$:

```text
[CLS]                                                         (segment 0)
[DEMO_START] [STATE] <70 tokens> [MOVES] <m_1 ... m_{L_1}>     (segment 1)
[OUT_*] [DEMO_END]                                            (segment 2)
...
[DEMO_START] [STATE] <70 tokens> [MOVES] <m_1 ... m_{L_K}>     (segment 2K-1)
[OUT_*] [DEMO_END]                                            (segment 2K)
[QUERY] [STATE] <70 tokens> [SEP]                             (segment 2K+1)
```

- **Sequence efficiency:** A game of $80$ plies takes $75 + 80 = 155$ tokens; $K=16$ full 80-ply games + query take $\sim 2{,}555$ tokens, and even $K=16$ full $320$-ply horizon games take $6{,}394 \le 8{,}192$ tokens.
- **No-query-truncation safety:** `[CLS]` + query (`74` tokens total) are reserved first. If total length exceeds `max_len`, `encode_episode` drops oldest demos (or skips oversized demos) while **never** dropping or truncating the query state. Demo move prefixes are trimmed **only** when `allow_trim_demo_prefix=True`, and `max_len < 74` raises `ValueError`.

---

## 2. Model Architecture (`ChessTransformerV2`)

### 2.1 Model Scaling Ladder (`d_base = 128`, `d_base_head = 32`, `max_seq_len = 8192`)

| Config | $d_{\text{model}}$ | $n_{\text{heads}}$ | $d_{\text{head}}$ | $n_{\text{layers}}$ | $d_{\text{ff}}$ | Width Mult $r$ | Exact Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|
| `TINY` | 128 | 4 | 32 | 4 | 512 | 1.0 | 1,395,971 |
| `SMALL` | 256 | 8 | 32 | 8 | 1,024 | 2.0 | 7,523,843 |
| `MEDIUM` | 384 | 8 | 48 | 12 | 1,536 | 3.0 | 23,102,211 |
| `LARGE` | 1,024 | 16 | 64 | 16 | 4,096 | 8.0 | 206,362,627 |

### 2.2 Rotary Position Embeddings (RoPE)

Learned absolute `pos_emb` is removed and replaced with RoPE ($\theta = 10000.0$) applied to per-head queries $q$ and keys $k$ inside every self-attention layer:
$$\theta_i = 10000^{-2i / d_{\text{head}}}, \quad i \in \{0, \dots, d_{\text{head}}/2 - 1\},$$
$$\mathrm{RoPE}(x, t) = x \odot \cos(t\theta) + \mathrm{rotate\_half}(x) \odot \sin(t\theta),$$
computed in `float32` and cast back to activation dtype for `bfloat16` numerical stability.

### 2.3 Maximal Update Parametrization ($\mu$P) Equations

With width multiplier $r = d_{\text{model}} / d_{\text{base}}$ ($d_{\text{base}} = 128$, $d_{\text{base,head}} = 32$):

1. **Attention logit scale:**
   $$\alpha_{\text{attn}} = \frac{\sqrt{d_{\text{base,head}}}}{d_{\text{head}}}$$
   (replacing $1/\sqrt{d_{\text{head}}}$ so attention entropy stays invariant across head dimensions `32, 48, 64`).
2. **Initialization:**
   - `token_emb`, `head`: $\mathcal{N}(0, 0.02^2)$
   - Hidden block linear layers (`c_attn`, `c_proj`, MLP `ff`): $\mathcal{N}\!\left(0, \left(\frac{0.02}{\sqrt{r}}\right)^2\right)$
   - Biases initialized to $0$; LayerNorm weights to $1$.
3. **Readout multiplier (last non-pad token):**
   $$\text{logits} = \frac{1}{r} W_{\text{head}}\,\mathrm{LN}_f(h_{T_{\text{last}}}) = \frac{d_{\text{base}}}{d_{\text{model}}} W_{\text{head}}\,\mathrm{LN}_f(h_{T_{\text{last}}}).$$
4. **AdamW learning rate scaling (`get_optimizer(base_lr)`):**
   - `token_emb`, `ln_*`, `head`: $\eta = \eta_{\text{base}}$
   - Hidden weights & biases inside transformer blocks: $\eta = \eta_{\text{base}} / r$.

### 2.4 Attention Modes (`causal` vs. `segment_bidirectional`)

Let $v_j \in \{0, 1\}$ denote the non-pad mask (`input_ids != pad_token_id` or `attention_mask`), and $s_i$ denote `segment_ids[b, i]`:

- **`causal` (default):** Position $i$ attends to $j$ iff $j \le i \land v_j = 1$ (fast `is_causal=True` SDPA path when no padding is present).
- **`segment_bidirectional` (ablation):** Position $i$ attends to $j$ iff
  $$v_j = 1 \;\land\; \big((s_j = s_i) \;\lor\; (s_j < s_i \land j < i)\big).$$
  Because each demo's `[OUT_*] [DEMO_END]` occupies its own isolated segment *after* the demo's board+moves segment, outcomes **cannot** leak backward into the demo's state/moves tokens, and query tokens **cannot** leak into earlier demos. For `[PAD]` query rows ($v_i = 0$), diagonal self-attention $(i = j)$ is enabled to avoid all-masked `NaN` rows in `softmax`.

### 2.5 Soft-Target Cross-Entropy

For empirical target probabilities $y = (\hat{p}_W, \hat{p}_D, \hat{p}_L) \in \Delta^2$ and readout logits $z \in \mathbb{R}^3$ at the final non-pad token (`[SEP]`):
$$\mathcal{L}(z, y) = -\sum_{c \in \{W, D, L\}} y_c \log \mathrm{softmax}(z)_c.$$

---

## 3. Observed Verification Results

### 3.1 $\mu$P Coordinate Check Across Widths (`base_lr = 1e-3`, `T = 48`, `B = 6`)

| Config | $r$ | $\mu$P `init_std` | $\mu$P `dh_1step` | $\mu$P `dlogit_1step` | Standard (`use_mup=False`) `dlogit_1step` |
|---|---:|---:|---:|---:|---:|
| `TINY` | 1.0 | 0.2347 | 0.8224 | **0.8156** | 0.8974 |
| `SMALL` | 2.0 | 0.1773 | 0.9601 | **0.8164** | 1.4763 |
| `MEDIUM` | 3.0 | 0.1519 | 1.0192 | **0.8101** | 2.4873 |
| `LARGE` | 8.0 | 0.0631 | 0.9972 | **0.7912** | 7.6407 |

Under $\mu$P, 1-step logit updates remain nearly identical (`0.7912`–`0.8164`) across an $8\times$ width increase (`1.40M` to `206.36M` params), whereas standard parameterization grows nearly linearly with width (`0.8974` $\to$ `7.6407`).

### 3.2 Unit & GPU Test Suite (`tests_v2/test_models_v2.py`)

Command:
```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest tests_v2/test_models_v2.py -v
```
Covers:
1. Tokenizer vocab (`4,705` tokens, `4,608` 1-token moves) and special tokens.
2. Lossless 70-token state round-trips including `X` barrier encoding (`blocked_squares`).
3. 1-token-per-move UCI round-trips (including promotions).
4. Episode segment isolation and $K=16$ full 80-ply game capacity inside `8192`.
5. No-query-truncation guarantee on tight `max_len` budgets.
6. Batch collation (`collate_episodes`) and padding invariance.
7. RoPE relative-position inner-product invariance.
8. Model shapes, `predict_proba` simplex normalization, and exact parameter counts.
9. Causal & `segment_bidirectional` leak tests (exact `0.0` diff on earlier segments when mutating demo outcome or query tokens; positive within-segment bidirectional influence).
10. Soft-target cross-entropy loss, gradient finiteness, and overfitting convergence.
11. $\mu$P coordinate check and 1-step stability across `TINY`/`SMALL`/`MEDIUM`/`LARGE`.
12. Checkpoint `save`/`load` round-trip.
13. CUDA `bfloat16` forward/backward smoke test on `SMALL` (`segment_bidirectional`).
