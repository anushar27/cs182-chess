# Chess-Variant ICL v2: Dataset Splitting, Deduplication & Leakage Audit

**Date:** 2026-11-25  
**Module:** `chess_icl_v2/data/splits.py`  
**Tests:** `tests_v2/test_splits_v2.py`

---

## 1. Strictly Disjoint Game-Pool Separation (`split_games_into_pools`)

All demonstration games and query roots originate from full simulated trajectories partitioned at the **source-game level** (`game_id` / `source_game_id`) prior to root sampling:

| Pool | Default Fraction | Output Key | Downstream Artifact |
|---|---:|---|---|
| Demonstration Pool | `0.40` | `demo_games` | `games_demo.jsonl` |
| Train Root Sources | `0.30` | `train_source_games` | `positions_train.jsonl` |
| Validation Root Sources | `0.15` | `val_source_games` | `positions_val.jsonl` |
| Test Root Sources | `0.15` | `test_source_games` | `positions_test.jsonl` |

Games sharing a `game_id` / `source_game_id` are grouped together before deterministic permutation (`np.random.default_rng(seed)`), guaranteeing zero `source_game_id` overlap across `demo_games`, `train`, `val`, and `test`.

---

## 2. Canonical & Near-Duplicate Filtering (`deduplicate_and_filter_roots`)

Even when source games are disjoint, early openings or transpositions can produce identical or near-identical board states across splits or between query roots and demonstration games (which would inflate `BoardKNNBaseline` or leak train/val/test states). Filtering proceeds in priority order **`test` $\succ$ `val` $\succ$ `train`**:

1. **Opening Ply Exclusion (`min_ply = 8`):** Any candidate root explicitly carrying `ply` / `move_number` / `halfmove` $< 8$ is excluded (`opening_ply_removed`), because early plies share $> 90\%$ square identity with the starting position and across games.
2. **Exact Canonical Deduplication (`canonical_state_key`):** Using `VariantBoard.canonical_state_key()` (which encodes variant, board placement with `'X'` barriers, side-to-move, castling rights, and legal en-passant target), any exact repeat across or within splits is removed (`exact_removed`), retaining the highest-priority split (`test` > `val` > `train`).
3. **Vectorized Hamming Near-Duplicate Filter (`board_similarity_signature`, `max_similarity = 0.90`):**
   - Each board state is encoded as $(\mathbf{b}, t) \in \{0, \dots, 13\}^{64} \times \{0, 1\}$ (`int8` square codes for `.`, `X`, `P..K`, `p..k` plus side-to-move bit $t$).
   - Similarity against any state $(\mathbf{b}', t')$ with matching turn ($t = t'$) is:
     $$\mathrm{sim}(\mathbf{b}, \mathbf{b}') = \frac{1}{64} \sum_{s=0}^{63} \mathbf{1}[b_s = b'_s].$$
   - Because $57/64 = 0.8906 < 0.90 < 58/64 = 0.9063$, rejecting $\mathrm{sim} \ge 0.90$ enforces Hamming distance $\ge 7$ differing squares out of 64 against:
     - **(a) All states along `demo_games` trajectories** (`neardup_vs_demos_removed`), preventing kNN memorization of demo trajectories; and
     - **(b) All accepted roots in higher-priority splits** (`neardup_cross_split_removed`), preventing near-duplicate leakage from `train`/`val` into `test`.

---

## 3. Dataset Integrity, Prompt Leakage & Balance Audit (`audit_dataset_integrity`)

For each variant directory under `data_dir`, `audit_dataset_integrity` verifies:

1. **Source-Game Disjointness:** Zero intersection of `source_game_id` across `games_demo.jsonl`, `positions_train.jsonl`, `positions_val.jsonl`, and `positions_test.jsonl`.
2. **Canonical State Disjointness:** Zero intersection of `canonical_state_key` across `positions_train.jsonl`, `positions_val.jsonl`, and `positions_test.jsonl`.
3. **Zero-Leakage Formatting Audit:**
   - Formats sampled episodes through **`QwenPromptFormatter`** (Track 1) and checks via `check_prompt_leakage` that prompts contain **no variant names** (`ALL_VARIANT_NAMES`), **no rule descriptions**, **no future rollout trajectories**, and **no query labels or identifiers** (`root_id`, `source_game_id`, `y_true` in `[Query]`).
   - Encodes sampled episodes through **`ChessV2Tokenizer`** (Track 2) and verifies the final `[QUERY]` segment has exact length `73` (`[QUERY]`, `[STATE]`, 70 state tokens, `[SEP]`) with zero `[OUT_*]`, `[MOVES]`, or `[DEMO_*]` tokens.
4. **Side & Outcome Balance Reporting:** Reports `n_positions`, White/Black side-to-move fractions (`white_frac`, `black_frac`), `equal_material_frac` ($|\Delta M| \le 2$), and mean/std `[p_win, p_draw, p_loss]` per variant and split.
