# Track 1: Qwen2.5-7B-Instruct Architecture, CUDA Smoke Test & Profiling

**Date:** 2026-11-25  
**Model:** `Qwen/Qwen2.5-7B-Instruct` (revision `a09a35458c702b33eeacc393d103063234e8bc28`)  
**Hardware:** 1x NVIDIA H100 80GB HBM3 (`81079.1 MB` reported by PyTorch)  
**Software:** `torch 2.15.0+cu132`, `transformers 5.24.0`, `peft 0.22.0`, `accelerate 1.16.0`, `safetensors 0.8.0` (`flash_attn` not installed -> `sdpa`)

---

## 1. Implementation & Design Decisions

Implemented in [`chess_icl_v2/models/qwen_model.py`](../chess_icl_v2/models/qwen_model.py), tested on CPU in [`tests_v2/test_qwen_v2.py`](../tests_v2/test_qwen_v2.py), and profiled on CUDA via [`scripts_v2/profile_qwen.py`](../scripts_v2/profile_qwen.py):

1. **Pinned Offline Model Loading:**
   - `MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"`, `REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"`.
   - `resolve_qwen_model_path()` resolves directly to the locally cached snapshot `/data/backups/rganapa/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28` when present and always passes `local_files_only=True`.
2. **Zero-Leakage, Query-Preserving Prompt Formatting (`QwenPromptFormatter`):**
   - Formats $K$ full-game demonstrations (`DemoGame`: initial 8x8 board state, move list in `uci` or `san`, White-perspective realized outcome `win`/`draw`/`loss`) followed by the query board state (`QueryPosition`: explicit 8x8 ASCII grid with `.` for empty squares and `X` for blocked squares, `Side to move: White/Black`, `Castling`, `En passant`).
   - Strictly ignores any `variant`, `rule`, `description`, `label`, or `target_probs` attributes/keys on demos and queries.
   - Guarantees the query block is **never truncated**: when token length exceeds `max_length`, binary-searches to drop the oldest demonstrations (`demos[0], demos[1], ...`) first; drops the static header before touching the query if $K=0$ still exceeds `max_length`; raises `ValueError` if the query block alone exceeds `max_length`.
3. **LoRA + 3-Way Probability Head (`QwenICLClassifier`):**
   - Frozen BF16 `Qwen2ForCausalLM` backbone with `attn_implementation="sdpa"`.
   - Calls `Qwen2Model` (`get_base_model().model(...)`) directly rather than the full causal LM head, avoiding materializing the 152,064-vocab logit tensor at every token.
   - Two LoRA presets targeting `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`:
     - **Mode A (`handoff`):** `r=8, lora_alpha=16`, last 4 blocks (`layers 24..27` of 28) -> `2,883,584` LoRA params.
     - **Mode B (`full_lora`):** `r=16, lora_alpha=32`, all 28 blocks.
   - Trainable FP32 linear head `nn.Linear(3584, 3)` (`10,755` params) applied to the **last non-pad token hidden state** (mask-aware indexing supporting both right- and left-padding).
   - Trained with soft-target cross-entropy `-(target_probs * log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()`.
   - Supports `gradient_checkpointing_enable()` (with `use_cache=False` and `enable_input_require_grads()`).

---

## 2. Verified Model Configuration

Observed directly from loaded `Qwen2ForCausalLM.config` on H100:

| Field | Observed Value |
|---|---|
| `architectures` | `["Qwen2ForCausalLM"]` |
| `max_position_embeddings` | `32768` |
| `hidden_size` | `3584` |
| `num_hidden_layers` | `28` |
| `num_attention_heads` | `28` |
| `num_key_value_heads` | `4` |
| `vocab_size` | `152064` |
| `attn_implementation` | `sdpa` |
| Cold load time (BF16) | `36.83 s` |
| Mode A trainable LoRA params | `2,883,584` |
| Trainable 3-way head params | `10,755` |
| Total trainable / total params | `2,894,339 / 7,618,510,851` (`0.0380%`) |

---

## 3. CUDA Smoke Test & Reload Determinism

Run via `CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 scripts_v2/profile_qwen.py`:

- **Status:** **PASSED**
- **1-step forward + backward + AdamW step:**
  - `smoke_loss`: `2.692578`
  - `nonzero_lora_grad_tensors_step1`: `28` (all 28 `lora_B` matrices across 4 layers x 7 target modules; `lora_A` is zero-grad on step 1 due to standard `B=0` LoRA init and becomes nonzero from step 2 onward)
  - `head_grad_norm_step1`: `220.769241`
- **Save & Fresh-Instance Reload Determinism (`save_adapter_and_head` -> `from_adapter_and_head`):**
  - `probs_before_reload`: `[[0.851526, 0.136661, 0.011812], [0.825550, 0.154108, 0.020342]]`
  - `probs_after_reload`: `[[0.851526, 0.136661, 0.011812], [0.825550, 0.154108, 0.020342]]`
  - `reload_max_abs_diff`: **`0.0`** (criterion `< 1e-5`)

---

## 4. Memory & Latency Profile (`batch_size=1`, `gradient_checkpointing=True`, Mode A `handoff`)

Measured over 1 warmup step + 3 timed `forward + backward + AdamW.step()` iterations per sequence length on 1x H100 80GB HBM3:

| Sequence Length | Status | Mean Step (ms) | Min / Max (ms) | Throughput (tok/s) | Peak Allocated (MB) | Peak Reserved (MB) |
|---:|---|---:|---:|---:|---:|---:|
| `4096` | `ok` | `401.66` | `388.90 / 415.66` | `10197.6` | `17105.1` | `17526.0` |
| `8192` | `ok` | `815.36` | `812.71 / 817.86` | `10047.2` | `19609.4` | `20318.0` |
| `16384` | `ok` | `1812.84` | `1810.97 / 1813.89` | `9037.7` | `24616.3` | `26234.0` |
| `24576` | `ok` | `3061.38` | `3056.97 / 3067.96` | `8027.7` | `29621.7` | `32084.0` |

**Takeaway:** At `seq_len=24576` with `batch_size=1` and gradient checkpointing, peak allocated HBM3 is `29.6 GB` (`32.1 GB` reserved) out of `81.1 GB`, leaving ~`49 GB` of headroom on a single H100 GPU.

---

## 5. UCI vs. SAN Tokenization Efficiency (20 Random 200-Ply Games)

Sampled 20 standard chess games reaching exactly 200 plies under uniform random legal moves (`seed=42`, 25 total games sampled to obtain 20 full 200-ply games) and tokenized with `Qwen2.5-7B-Instruct` tokenizer (`add_special_tokens=False`):

| Representation | Mean Tokens / Game | Stdev | Min / Max | Tokens / Ply | SAN / UCI Ratio |
|---|---:|---:|---:|---:|---:|
| Raw move sequence (`UCI`) | `801.25` | `1.07` | `800 / 803` | `4.006` | — |
| Raw move sequence (`SAN`) | `513.70` | `9.84` | `496 / 529` | `2.569` | **`0.6411`** (-35.9%) |
| Full formatted `DemoGame` (`UCI`) | `929.25` | `1.07` | `928 / 931` | — | — |
| Full formatted `DemoGame` (`SAN`) | `641.65` | `9.82` | `624 / 657` | — | **`0.6905`** (-30.9%) |

**Context Budget Implications (for 200-ply games):**
- `SAN` reduces full-demo token length by **30.9%** (`641.65` vs `929.25` tokens/game): $K=16$ full 200-ply demos fit in ~`10.3k` tokens (vs ~`14.9k` for `UCI`), and $K=32$ full 200-ply demos fit in ~`20.5k` tokens (under `24,576`, whereas `UCI` reaches ~`29.7k` tokens).
- `UCI` has lower length variance (`stdev = 1.07` vs `9.82`) and requires no variant-specific SAN check/disambiguation rules, while `SAN` maximizes $K$ capacity under the `24k`/`32k` context budget.

---

## 6. Unit Test Verification

Ran sandboxed CPU tests (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest tests_v2/test_qwen_v2.py -v`):
- **Result:** `9 passed in 18.46s` (0 failures).

Ran sandboxed pipeline tests (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest tests_v2/test_qwen_pipeline.py -v`):
- **Result:** `2 passed in 18.09s` (0 failures).

---

## 7. Real 7B E2E Pipeline Verification

Verified [`scripts_v2/train_qwen.py`](../scripts_v2/train_qwen.py) and [`scripts_v2/eval_qwen.py`](../scripts_v2/eval_qwen.py) end-to-end on 1x NVIDIA H100 80GB HBM3 (`CUDA_VISIBLE_DEVICES=1`) using the real cached `Qwen/Qwen2.5-7B-Instruct` (`a09a35458c702b33eeacc393d103063234e8bc28`) weights over a temporary synthetic v2 dataset (`standard` + `diagonal_rook`, 10 demo games and 6 train / 4 val / 4 test positions each, cleaned up after verification):

| Stage | Command Highlights | Wall Time (s) | Peak GPU VRAM (`nvidia-smi`) | Key Observed Outputs |
|---|---|---:|---:|---|
| 1. Warmup Train | `--mode warmup --lora-mode handoff --notation san --max-steps 3 --eval-interval 3 --batch-size 1 --grad-accum 2 --k-values 0 2 --val-k-values 0 2 --max-seq-len 4096` | `24.76` | `15889 MiB` | `history.json`: `step=3`, `train_loss=2.166425`, `last_step_loss=1.596104`, `step_ms=586.74`, `val_nll=1.555895`, `val_brier=0.180262`, `val_acc=0.375`, `best_step=3`, saved `best_adapter/` |
| 2. Continue Train | `--mode continue --init-adapter-dir .../warmup/best_adapter --variants diagonal_rook --max-steps 3 --eval-interval 3 --batch-size 1 --grad-accum 2 --k-values 0 2 --val-k-values 0 2 --max-seq-len 4096` | `18.49` | `15895 MiB` | `history.json`: `step=3`, `train_loss=1.351069`, `last_step_loss=1.220167`, `step_ms=791.45`, `val_nll=1.232451`, `val_brier=0.117040`, `val_acc=0.125`, `best_step=3`, saved `best_adapter/` |
| 3. Full Eval | `--adapter-dir .../continue/best_adapter --variants diagonal_rook --k-values 0 2 --num-draws 2 --max-seq-len 4096 --notation san --n-boot 50` | `17.40` | `16219 MiB` | `eval_test.json`: `40` `per_root_predictions`; `clean` & `equal_material` summaries at `K=0, 2`; all 3 controls (`mismatched_context`, `corrupted_outcomes`, `corrupted_trajectories`) at `K=2`; `clean_k0_nll=1.240817`, `clean_k2_nll=1.247153`, `clean_k2_context_gain=-0.006336` |

No code fixes were needed in `qwen_model.py`, `train_qwen.py`, or `eval_qwen.py` during the real 7B run.
