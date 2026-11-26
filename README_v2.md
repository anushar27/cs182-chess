# Chess-Variant In-Context Learning v2 (`chess_icl_v2`)

**CS 182/282 Reproducibility & Architecture Guide**  
`chess_icl_v2` is a clean-slate redesign of the chess-variant ICL benchmark under an empirical rollout protocol, formally audited variant simulators, dual-track neural evaluation (Track 1 `Qwen2.5-7B-Instruct` LoRA vs. Track 2 contamination-free $\mu$P-RoPE transformers), non-neural baselines, causal intervention controls, and hierarchical cluster-bootstrap uncertainty. Legacy v1 modules (`chess_icl/`, `scripts/`, `data/`, `runs/`) remain untouched.

---

## 1. Verified Environment & Installation

All v2 tests and reproduction runs were verified on:

| Component | Verified Version / Hardware |
|---|---|
| Python | `3.12.7` |
| PyTorch | `2.15.0+cu132` (CUDA `13.2`) |
| GPU (verified) | NVIDIA H100 80GB HBM3 |
| `python-chess` | `1.11.2` |
| `transformers` | `5.24.0` |
| `peft` | `0.22.0` |
| `numpy` | `2.5.5` |
| `matplotlib` | `3.11.2` |
| `pytest` | `9.2.0` |

Install base dependencies (plus `transformers`, `peft`, `accelerate`, `safetensors` for Track 1 Qwen experiments):

```bash
pip install -r requirements.txt
pip install transformers peft accelerate safetensors pytest
```

For Track 1 offline loading of the pinned `Qwen/Qwen2.5-7B-Instruct` revision (`a09a35458c702b33eeacc393d103063234e8bc28`), ensure the snapshot is cached locally and export:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

---

## 2. Running the Test Suite

Set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` on multi-core hosts to avoid CPU thread oversubscription on small tensors:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest tests_v2/ -v
```

**Observed output:** `70 passed, 2 warnings in 74.77s (0:01:14)` across:
- `tests_v2/test_variants_v2.py` (19 tests: variant specs, 300-position differential fuzzing vs. `python-chess`, pure-Python reference & property audits)
- `tests_v2/test_splits_v2.py` (4 tests: game-disjoint splits, deduplication, Hamming/leakage integrity audits)
- `tests_v2/test_models_v2.py` (13 tests: `ChessV2Tokenizer`, RoPE, segment-bidirectional leak isolation, soft-target CE, $\mu$P coordinate check across `TINY`/`SMALL`/`MEDIUM`/`LARGE`, CUDA `bfloat16` smoke)
- `tests_v2/test_track2_pipeline.py` (5 tests: `EpisodeDatasetV2`, `EvalEpisodeBuilder`, Track 2 warmup $\to$ continue $\to$ eval end-to-end in `causal` and `segment_bidirectional` modes)
- `tests_v2/test_qwen_v2.py` & `tests_v2/test_qwen_pipeline.py` (11 tests: `QwenPromptFormatter` zero-leakage & no-query-truncation, LoRA modes, mock warmup/continue/eval pipeline)
- `tests_v2/test_eval_v2.py` & `tests_v2/test_baselines_and_plots.py` (18 tests: metrics, MC noise floor, 6 non-neural baselines, 3 causal controls, hierarchical cluster bootstrap, paper figure/LaTeX generation)

---

## 3. Tiny End-to-End Reproduction (`< 30s`)

Run the self-contained reproduction script to generate a deterministic simulated dataset under `data_v2_tiny/` (`standard`, `diagonal_rook`, `short_knight` with `20` demo games, `24` train, `12` val, and `12` test positions each), run Track 2 `TINY` warmup $\to$ continuation, evaluate with 5 demo draws per $K$ and causal controls, run all 6 non-neural baselines, and generate PDF/PNG figures and LaTeX tables in `runs_v2_tiny/paper_figures/`:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts_v2/run_tiny_repro.py
```

Pass `--gpu -1` to force CPU-only execution.

**Measured wall-clock timing (observed on `cuda:0`, H100 80GB):**
- `data_gen_sec`: `4.619s`
- `train_warmup_sec`: `4.936s`
- `train_continue_sec`: `0.440s`
- `eval_track2_sec`: `3.443s`
- `run_baselines_sec`: `0.949s`
- `make_paper_figures_sec`: `5.587s`
- **Total end-to-end time:** **`22.54s`** (`22.538s`), generating `8` figure files (`fig1_training_curves`, `fig2_context_gain_vs_k`, `fig4_causal_controls`, `fig5_side_calibration_and_variants` as 300-DPI `.png` + `.pdf`) and `3` LaTeX files (`table1_main_results.tex`, `table2_hyperparameters.tex`, `captions.tex`). (`fig3_mup_scaling` is skipped in tiny repro because only `TINY` is trained.)

---

## 4. Directory Layout

```text
cs182-chess/
├── README_v2.md                              # Reproducibility guide (this file)
├── chess_icl_v2/
│   ├── __init__.py
│   ├── variants/
│   │   ├── specs.py                          # VariantSpec definitions & splits (standard + 13 single + 3 compositional)
│   │   ├── board.py                          # VariantBoard(chess.Board) bitboard rule hooks & canonical state keys
│   │   ├── variants_impl.py                  # Concrete variant subclasses + get_board()
│   │   └── audit.py                          # Differential fuzzing, pure-Python verifier, property audits
│   ├── data/
│   │   ├── tokenizer_v2.py                   # ChessV2Tokenizer (V=4,705: 70-tok state + 1-tok UCI moves, no-query-truncation)
│   │   ├── dataset_v2.py                     # EpisodeDatasetV2, EvalEpisodeBuilder, collate_episode_batch
│   │   └── splits.py                         # Disjoint game splits & anti-leakage board similarity audits
│   ├── models/
│   │   ├── small_transformer.py              # ChessTransformerV2 (RoPE, muP, causal & segment_bidirectional)
│   │   └── qwen_model.py                     # QwenPromptFormatter & QwenICLClassifier (Qwen2.5-7B LoRA + 3-way head)
│   └── eval/
│       ├── metrics.py                        # Soft-target NLL, Brier, L1/TV, ECE, Side calibration, Context Gain, MC floor
│       ├── baselines.py                      # Uniform, TrainingPrior, DemoMajority, LengthHistogram, QueryMaterial, BoardKNN
│       ├── controls.py                       # Mismatched-context, corrupted-outcome, corrupted-trajectory builders
│       └── uncertainty.py                    # Stratified cluster bootstrap over source_game_id + hierarchical aggregation
├── scripts_v2/
│   ├── run_tiny_repro.py                     # <30s self-contained end-to-end reproduction script
│   ├── train_track2.py                       # Track 2 warmup & variant-mixture continuation trainer
│   ├── eval_track2.py                        # Track 2 evaluation with 5 draws/K, controls, equal-material slice & CIs
│   ├── train_qwen.py                         # Track 1 Qwen2.5-7B LoRA warmup & continuation trainer
│   ├── eval_qwen.py                          # Track 1 Qwen2.5-7B LoRA evaluator
│   ├── profile_qwen.py                       # CUDA smoke, memory/latency profile, & UCI vs SAN benchmark
│   ├── run_baselines.py                      # Non-neural baseline runner across K and 5 demo draws
│   └── make_paper_figures.py                 # 300-DPI PDF/PNG figures & LaTeX table generator
├── tests_v2/                                 # Comprehensive unit, audit, and end-to-end pipeline test suite
├── docs_v2/
│   ├── 00_PROTOCOL_AND_DECISIONS.md          # Research questions, v1 post-mortem, frozen protocol tracker
│   ├── 01_VARIANT_SPECS_AND_AUDITS.md        # Formal rules for 17 variants + differential fuzzing results
│   ├── 03_QWEN_CUDA_PROFILE.md               # Qwen2.5-7B H100 profile & tokenization benchmarks
│   ├── 03_TRACK2_ARCHITECTURE.md             # Track 2 tokenizer, RoPE, muP math & coordinate checks
│   ├── 04_EVALUATION_AND_BASELINES_SPEC.md   # Scoring rules, baselines, controls & cluster bootstrap spec
│   └── 05_PAPER_METHOD_AND_RELATED_WORK.md   # Paper Intro, Background, Related Work, Method Math & Hyperparams
└── runs_v2/                                  # Output directory for full-scale experiments (runs_v2_tiny/ for tiny repro)
```
