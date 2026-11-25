"""Evaluate non-parametric and heuristic baselines across all 13 chess variants.

Baselines:
1. Random Chance Baseline (Uniform 1/3)
2. Material Count Baseline (Standard: >=+3 Win, <=-3 Loss, else Draw; Antichess: inverted)
3. Demo Majority Baseline (Predict majority outcome of K in-context demo games)
4. Demo Material-Calibrated KNN (Nearest demo position by material balance)
"""

import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np
import chess


PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def compute_material_balance(fen: str) -> int:
    """Compute material balance from the perspective of side to move."""
    board = chess.Board(fen)
    white_mat = sum(len(board.pieces(pt, chess.WHITE)) * val for pt, val in PIECE_VALUES.items())
    black_mat = sum(len(board.pieces(pt, chess.BLACK)) * val for pt, val in PIECE_VALUES.items())
    diff = white_mat - black_mat
    return diff if board.turn == chess.WHITE else -diff


def material_predict(fen: str, variant: str, threshold: int = 3) -> str:
    """Predict W/D/L based purely on static material balance."""
    mat = compute_material_balance(fen)
    if variant == "antichess":
        # In antichess, having fewer pieces is winning
        if mat <= -threshold:
            return "win"
        elif mat >= threshold:
            return "loss"
        else:
            return "draw"
    else:
        if mat >= threshold:
            return "win"
        elif mat <= -threshold:
            return "loss"
        else:
            return "draw"


def run_baselines(data_dir: Path, output_file: Path):
    variants = [
        "diagonal_rook", "straight_bishop", "weakened_queen", "extended_pawn",
        "short_knight", "mobile_king", "piece_capture", "forward_pawn_capture",
        "dark_square_immunity", "blocked_squares", "long_board", "knight_army", "antichess"
    ]
    
    results = {}
    label_map = {"win": 0, "draw": 1, "loss": 2}
    
    for v in variants:
        v_dir = data_dir / v
        pos_file = v_dir / "positions_eval.jsonl"
        demo_file = v_dir / "games_demo.jsonl"
        if not pos_file.exists():
            continue
            
        with open(pos_file) as f:
            positions = [json.loads(line) for line in f if line.strip()]
            
        demos = []
        if demo_file.exists():
            with open(demo_file) as f:
                demos = [json.loads(line) for line in f if line.strip()]
                
        # 1. Material baseline
        mat_preds = [material_predict(p["fen"], v) for p in positions]
        targets = [p.get("label", "draw") for p in positions]
        mat_acc = sum(p == t for p, t in zip(mat_preds, targets)) / len(targets)
        
        # 2. Demo Majority baseline for K in [1, 2, 4, 8]
        demo_majority_accs = {}
        for K in [1, 2, 4, 8]:
            k_accs = []
            for seed in [42, 123, 456]:
                rng = np.random.default_rng(seed)
                selected_demos = rng.choice(demos, size=min(K, len(demos)), replace=False)
                demo_results = [d.get("result", "draw") for d in selected_demos]
                majority_label = Counter(demo_results).most_common(1)[0][0]
                acc = sum(majority_label == t for t in targets) / len(targets)
                k_accs.append(acc)
            demo_majority_accs[K] = float(np.mean(k_accs))
            
        results[v] = {
            "total_samples": len(targets),
            "material_heuristic_acc": float(mat_acc),
            "demo_majority_acc": demo_majority_accs,
            "random_chance": 1.0 / 3.0,
        }
        print(f"[{v:<20}] Material: {mat_acc*100:5.2f}% | Demo Maj (K=1): {demo_majority_accs[1]*100:5.2f}% | Demo Maj (K=4): {demo_majority_accs[4]*100:5.2f}%")
        
    # Compute aggregates
    all_mat = [results[v]["material_heuristic_acc"] for v in results]
    train_vars = ["diagonal_rook", "straight_bishop", "weakened_queen", "extended_pawn"]
    test_vars = [v for v in variants if v not in train_vars and v in results]
    
    summary = {
        "overall_material_mean": float(np.mean(all_mat)),
        "in_distribution_material_mean": float(np.mean([results[v]["material_heuristic_acc"] for v in train_vars if v in results])),
        "out_of_distribution_material_mean": float(np.mean([results[v]["material_heuristic_acc"] for v in test_vars])),
        "variants": results,
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nBaselines summary: Overall Material Mean = {summary['overall_material_mean']*100:.2f}%")
    print(f"Saved results to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output", type=str, default="runs/results/eval_baselines.json")
    args = parser.parse_args()
    run_baselines(Path(args.data_dir), Path(args.output))
