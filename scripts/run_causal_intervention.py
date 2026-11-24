"""Causal Intervention and Demonstration Corruption Study.

Tests whether the transformer causally relies on demonstration outcome tokens:
1. Clean (original demonstrations)
2. Inverted outcomes (Win -> Loss, Loss -> Win)
3. Scrambled outcomes (random outcome assignment)
4. All-Draw outcomes (ablation of outcome signal)
"""

import argparse
import copy
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chess_icl.data import ChessTokenizer, load_positions, load_games
from chess_icl.dataset import ChessEvalDataset
from chess_icl.model import ChessTransformer


def invert_result(res: str) -> str:
    if res == "win":
        return "loss"
    elif res == "loss":
        return "win"
    return "draw"


def run_intervention(
    model: ChessTransformer,
    tokenizer: ChessTokenizer,
    data_dir: Path,
    variants: list[str],
    k_values: list[int] = [1, 2, 4, 8],
    seeds: list[int] = [42, 123, 456],
    batch_size: int = 64,
    device: str = "cuda",
) -> dict:
    model.eval()
    results = {}
    label_map = {"win": 0, "draw": 1, "loss": 2}
    inv_label_map = {0: "win", 1: "draw", 2: "loss"}

    conditions = ["clean", "inverted", "scrambled", "all_draw"]

    for var in variants:
        v_dir = data_dir / var
        if not (v_dir / "positions_eval.jsonl").exists():
            continue

        raw_demos = load_games(v_dir / "games_demo.jsonl")
        positions = load_positions(v_dir / "positions_eval.jsonl")

        var_results = defaultdict(lambda: defaultdict(dict))

        for K in k_values:
            for cond in conditions:
                seed_accs = []
                seed_flip_to_inv = []

                for seed in seeds:
                    rng = np.random.default_rng(seed)
                    # Sample K demo games
                    demo_indices = rng.choice(len(raw_demos), size=min(K, len(raw_demos)), replace=False)
                    selected_demos = [copy.deepcopy(raw_demos[i]) for i in demo_indices]

                    # Apply intervention to demo outcomes
                    for d in selected_demos:
                        orig = d.get("result", "draw")
                        if cond == "inverted":
                            d["result"] = invert_result(orig)
                        elif cond == "scrambled":
                            d["result"] = rng.choice(["win", "draw", "loss"])
                        elif cond == "all_draw":
                            d["result"] = "draw"

                    # Batch encode inputs
                    all_tokens = []
                    all_targets = []
                    for pos in positions:
                        tokens = tokenizer.encode_input(selected_demos, pos["fen"])
                        if len(tokens) <= 2048:
                            all_tokens.append(tokens)
                            all_targets.append(label_map.get(pos.get("label", "draw"), 1))

                    if not all_tokens:
                        continue

                    # Pad and run inference in batches
                    preds = []
                    for i in range(0, len(all_tokens), batch_size):
                        batch = all_tokens[i : i + batch_size]
                        max_len = max(len(t) for t in batch)
                        padded = [t + [tokenizer.pad_id] * (max_len - len(t)) for t in batch]
                        inp = torch.tensor(padded, dtype=torch.long, device=device)
                        with torch.no_grad():
                            logits = model(inp)
                            batch_preds = logits.argmax(dim=-1).cpu().tolist()
                            preds.extend(batch_preds)

                    preds = np.array(preds)
                    targets = np.array(all_targets)
                    acc = float(np.mean(preds == targets))
                    seed_accs.append(acc)

                    # For inverted: check how often model flipped to the inverted target
                    if cond == "inverted":
                        inv_targets = np.array([0 if t == 2 else (2 if t == 0 else 1) for t in targets])
                        flip_rate = float(np.mean(preds == inv_targets))
                        seed_flip_to_inv.append(flip_rate)

                var_results[K][cond]["accuracy"] = float(np.mean(seed_accs))
                if cond == "inverted":
                    var_results[K][cond]["flipped_to_inverted"] = float(np.mean(seed_flip_to_inv))

        results[var] = var_results
        clean_k1 = var_results[1]["clean"]["accuracy"] * 100
        inv_k1 = var_results[1]["inverted"]["accuracy"] * 100
        scram_k1 = var_results[1]["scrambled"]["accuracy"] * 100
        print(f"[{var:<20}] K=1 -> Clean: {clean_k1:5.2f}% | Inverted: {inv_k1:5.2f}% | Scrambled: {scram_k1:5.2f}%")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output", type=str, default="runs/results/eval_causal_intervention.json")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.checkpoint} on {device}...")
    model = ChessTransformer.load(args.checkpoint, map_location=device).to(device)
    tokenizer = ChessTokenizer()
    model.pad_token_id = tokenizer.pad_id

    # Test on key variants: 2 in-dist, 3 OOD
    test_variants = ["diagonal_rook", "straight_bishop", "mobile_king", "piece_capture", "antichess"]

    results = run_intervention(
        model=model,
        tokenizer=tokenizer,
        data_dir=Path(args.data_dir),
        variants=test_variants,
        k_values=[1, 2, 4],
        device=device,
        batch_size=args.batch_size,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "results": results}, f, indent=2)
    print(f"Saved causal intervention results to {out_path}")


if __name__ == "__main__":
    main()
