"""Evaluation metrics, Matteo's re-evaluation test, and demo quality comparison."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence, Union

import torch
from .data import ChessTokenizer, load_games, load_positions
from .model import ChessTransformer


def compute_accuracy(predictions: Sequence[int], labels: Sequence[int]) -> dict[str, Any]:
    """Compute overall and per-class accuracy for W/D/L classification."""
    correct = sum(p == l for p, l in zip(predictions, labels))
    total = len(labels)
    if total == 0:
        return {"accuracy": 0.0, "per_class": {}, "total": 0}

    per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    for p, l in zip(predictions, labels):
        per_class[l]["total"] += 1
        if p == l:
            per_class[l]["correct"] += 1

    label_names = {0: "win", 1: "draw", 2: "loss"}
    per_class_acc = {
        label_names.get(k, str(k)): (v["correct"] / v["total"] if v["total"] > 0 else 0.0)
        for k, v in per_class.items()
    }

    return {
        "accuracy": correct / total,
        "per_class": per_class_acc,
        "total": total,
    }


def position_reevaluation_test(
    model: ChessTransformer,
    tokenizer: ChessTokenizer,
    standard_positions: Sequence[dict],
    variant_positions: Sequence[dict],
    demos: Sequence[dict],
    K: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    """Matteo's idea: Same board position, different evaluation under different rules.

    1. Take positions that are WINS in standard chess.
    2. Find the same positions' labels under the variant rules (may be LOSS, e.g., Antichess).
    3. Give the model K variant demos + the position.
    4. Measure whether the model correctly FLIPS its evaluation.
    """
    model.eval()
    flipped_positions = []

    # Map standard positions by FEN
    std_map = {pos["fen"]: pos.get("label", "draw") for pos in standard_positions}

    for var_pos in variant_positions:
        fen = var_pos["fen"]
        if fen in std_map:
            std_label = std_map[fen]
            var_label = var_pos.get("label", "draw")
            if std_label != var_label:
                flipped_positions.append({
                    "fen": fen,
                    "standard_label": std_label,
                    "variant_label": var_label,
                })

    if not flipped_positions:
        return {"flip_accuracy": None, "count": 0, "correct_flips": 0}

    correct_flips = 0
    label_map = {"win": 0, "draw": 1, "loss": 2}
    batch_size = 64

    with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        for i in range(0, len(flipped_positions), batch_size):
            batch_pos = flipped_positions[i : i + batch_size]
            batch_inps = [tokenizer.encode_input(demos[:K], pos["fen"]) for pos in batch_pos]
            max_len = max(len(inp) for inp in batch_inps)
            padded = [inp + [tokenizer.pad_id] * (max_len - len(inp)) for inp in batch_inps]
            inp_t = torch.tensor(padded, dtype=torch.long, device=device)
            logits = model(inp_t)
            preds = logits.argmax(-1).cpu().tolist()
            targets = [label_map.get(pos["variant_label"], 1) for pos in batch_pos]
            correct_flips += sum(1 for pr, tr in zip(preds, targets) if pr == tr)

    return {
        "flip_accuracy": correct_flips / len(flipped_positions),
        "count": len(flipped_positions),
        "correct_flips": correct_flips,
    }


def demo_quality_comparison(
    model: ChessTransformer,
    tokenizer: ChessTokenizer,
    variant_dir: Union[str, Path],
    K_values: Sequence[int] = (4, 8),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 64,
) -> dict[int, dict[str, float]]:
    """Experiment 3: Compare accuracy with strong (heuristic) vs weak (random) demos."""
    v_dir = Path(variant_dir)
    strong_demos = load_games(v_dir / "games_demo.jsonl")
    weak_demos = load_games(v_dir / "games_demo_weak.jsonl")
    eval_positions = load_positions(v_dir / "positions_eval.jsonl")

    label_map = {"win": 0, "draw": 1, "loss": 2}
    results = {}

    model.eval()
    with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        for K in K_values:
            for demo_type, pool in [("strong", strong_demos), ("weak", weak_demos)]:
                demos = pool[:K]
                correct, total = 0, 0
                for i in range(0, len(eval_positions), batch_size):
                    batch_pos = eval_positions[i : i + batch_size]
                    batch_inps = [tokenizer.encode_input(demos, p["fen"]) for p in batch_pos]
                    max_len = max(len(inp) for inp in batch_inps)
                    padded = [inp + [tokenizer.pad_id] * (max_len - len(inp)) for inp in batch_inps]
                    inp_t = torch.tensor(padded, dtype=torch.long, device=device)
                    logits = model(inp_t)
                    preds = logits.argmax(-1).cpu().tolist()
                    targets = [label_map.get(p.get("label", "draw"), 1) for p in batch_pos]
                    correct += sum(1 for pr, tr in zip(preds, targets) if pr == tr)
                    total += len(batch_pos)

                acc = correct / total if total > 0 else 0.0
                if K not in results:
                    results[K] = {}
                results[K][demo_type] = acc

    return results
