> **Current experiment:** predict fixed-policy W/D/L probabilities with the audited
> `chess_icl_v2` pipeline. Start with [README_V2.md](README_V2.md) and its results
> ledger. The documentation and results below describe the legacy material-label
> experiments and must not be used as evidence for the new objective.

# In-Context Learning of Unfamiliar Chess Variants

This repository implements the research pipeline for the research question:
**Can a small transformer pretrained on standard chess learn to evaluate positions in unfamiliar chess variants from K in-context demonstration games, without being told the rules?**

## Architecture & Pipeline Overview

- **Task formulation**: Win/Draw/Loss (W/D/L) classification (not centipawn regression)
- **No rules given**: The model receives $K$ demonstration games followed by the query position; no natural language rules are provided.
- **Custom variant engine**: Modular engine built on `python-chess` supporting 13 creative variants across 5 categories.
- **Model**: Decoder-only transformer (`ChessTransformer`) with causal self-attention.

### The 13 Variants

| # | Variant Code | Category | Rule Summary | Split |
|---|--------------|----------|--------------|-------|
| 1 | `diagonal_rook` | Move | Rook moves diagonally (like a bishop) | **Train** |
| 2 | `straight_bishop` | Move | Bishop moves in straight lines (like a rook) | **Train** |
| 3 | `weakened_queen` | Move | Queen loses diagonal movement (moves like rook only) | **Train** |
| 4 | `extended_pawn` | Move | Pawns can always move 1 or 2 squares forward | **Train** |
| 5 | `short_knight` | Move | Knight moves in short L-shape (ferz + dabbaba) | **Test** |
| 6 | `mobile_king` | Move | King moves like a queen | **Test** |
| 7 | `piece_capture` | Capture | Captured piece converts to your color on capture square | **Test** |
| 8 | `forward_pawn_capture` | Capture | Pawns can also capture square directly ahead | **Test** |
| 9 | `dark_square_immunity` | Capture | Pieces on dark squares cannot be captured | **Test** |
| 10 | `blocked_squares` | Board | 8 impassable barrier squares | **Test** |
| 11 | `long_board` | Board | Extended board simulated via multi-square pawn pushes | **Test** |
| 12 | `knight_army` | Pieces | Each side has 15 knights + 1 king | **Test** |
| 13 | `antichess` | Objective | Lose all pieces to win; captures are compulsory | **Test** |

---

## Directory Structure

```
cs182-chess/
├── chess_icl/
│   ├── data.py                    # ChessTokenizer, Game/Position dataclasses, loaders
│   ├── dataset.py                 # PyTorch datasets (Pretrain, Variant, Eval)
│   ├── model.py                   # ChessTransformer (GPT-style causal transformer)
│   ├── evaluate.py                # Metrics, Matteo re-evaluation test, demo quality
│   ├── prepare_data.py            # High-level data preparation wrapper
│   └── variants/                  # Modular variant engine
│       ├── engine.py              # VariantBoard base class
│       ├── move_variants.py       # 6 move variants
│       ├── capture_variants.py    # 3 capture variants
│       ├── board_variants.py      # blocked_squares, long_board
│       ├── piece_variants.py      # knight_army
│       ├── objective_variants.py  # antichess
│       └── registry.py            # Variant registry & get_board factory
├── scripts/
│   ├── generate_data.py           # Self-play game generation & position sampling
│   ├── pretrain.py                # Standard chess pretraining (Phase 2)
│   ├── continue_train.py          # Variant continue-training (Phase 3)
│   ├── run_eval.py                # Full evaluation matrix (Phase 4)
│   └── analyze_results.py         # Result aggregation, figures, LaTeX tables
├── tests/
│   ├── test_variants.py           # Unit tests for variant engine rules
│   ├── test_tokenizer.py          # Tokenizer & sequence roundtrip tests
│   └── test_model.py              # Model shape, gradient, and overfit tests
├── configs/
│   ├── small.yaml                 # Transformer hyperparameters
│   └── experiment.yaml            # Experiment matrix & split specification
├── cluster/
│   ├── generate_data.sh           # SLURM script for data generation
│   ├── pretrain.sh                # SLURM script for pretraining (3 seeds)
│   ├── continue_train.sh          # SLURM script for continue-training (9 jobs)
│   ├── evaluate.sh                # SLURM script for evaluation matrix
│   └── run_all.sh                 # Master pipeline submit script
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Tests
```bash
pytest tests/ -v
```

### 3. Generate Data
```bash
# Generate standard chess and all 13 variants
python scripts/generate_data.py --all-variants --count 2000 --seed 42
```

### 4. Phase 2: Pretraining
```bash
python scripts/pretrain.py --seed 42 --config small --max-steps 20000 --output runs/pretrain/
```

### 5. Phase 3: Continue-Training
```bash
python scripts/continue_train.py \
    --config move_variants \
    --pretrain-checkpoint runs/pretrain/chess_base_seed42.pt \
    --seed 42 \
    --output runs/variant_train/
```

### 6. Phase 4: Evaluation & Analysis
```bash
python scripts/run_eval.py \
    --checkpoint runs/variant_train/move_variants_seed42.pt \
    --output runs/results/move_variants_seed42.json

python scripts/analyze_results.py --results-dir runs/results --output-dir runs/analysis
```
