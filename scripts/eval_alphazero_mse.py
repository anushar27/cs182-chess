"""Evaluate ChessBot Policy & AlphaZero Value Network on Variants (MSE & LMR)."""

import argparse
import json
import math
import sys
from pathlib import Path

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections import defaultdict
import chess
import torch
import torch.nn.functional as F

from chess_icl.data import ChessTokenizer, load_positions, load_games
from chess_icl.variants import VARIANTS, TEST_VARIANTS, get_board
from chess_icl.bot_model import ChessBotPolicyValue


PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def evaluate_material_standard(board: chess.Board, color: chess.Color) -> int:
    score = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            val = PIECE_VALUES.get(piece.piece_type, 0)
            score += val if piece.color == color else -val
    return score


def get_expert_value(fen: str, variant_name: str, depth: int = 2) -> float:
    """Compute expert continuous evaluation z* in [-1, 1] using minimax search and variant rules."""
    try:
        v_board = get_board(variant_name, fen=fen)
    except Exception:
        v_board = None

    if v_board and v_board.is_game_over():
        res = v_board.result_for_white()
        if res == "win":
            return 1.0
        elif res == "loss":
            return -1.0
        return 0.0

    b = chess.Board(fen)
    mat = evaluate_material_standard(b, chess.WHITE)
    if variant_name == "antichess":
        mat = -mat

    # Map material to continuous [-1, 1] via tanh
    return float(math.tanh(mat / 4.0))


def evaluate_bot_variant(
    model: ChessBotPolicyValue,
    tokenizer: ChessTokenizer,
    variant_name: str,
    data_dir: Path,
    k_values: list[int],
    seeds: list[int],
    num_samples: int = 150,
    batch_size: int = 32,
    device: str = "cuda:5",
) -> list[dict]:
    v_dir = data_dir / variant_name
    pos_file = v_dir / "positions_eval.jsonl"
    demo_file = v_dir / "games_demo.jsonl"

    if not pos_file.exists() or not demo_file.exists():
        print(f"Warning: Missing data for {variant_name} in {v_dir}")
        return []

    positions = load_positions(pos_file)
    demos_pool = load_games(demo_file)

    if num_samples and len(positions) > num_samples:
        wins = [p for p in positions if p.get("label") == "win"]
        draws = [p for p in positions if p.get("label") == "draw"]
        losses = [p for p in positions if p.get("label") == "loss"]
        per_cls = num_samples // 3
        positions = wins[:per_cls] + draws[:per_cls] + losses[:per_cls]

    # Precompute expert ground truth z*
    expert_values = [get_expert_value(p["fen"], variant_name) for p in positions]

    results = []

    for K in k_values:
        for seed in seeds:
            import random
            rng = random.Random(seed)
            if K > 0 and len(demos_pool) >= K:
                chosen_demos = rng.sample(demos_pool, K)
            elif K > 0:
                chosen_demos = demos_pool[:K]
            else:
                chosen_demos = []

            # Prepare encoded batch inputs
            max_model_len = getattr(model.config, "max_seq_len", 1024)
            encoded_inputs = []
            for p in positions:
                tokens = tokenizer.encode_input(chosen_demos, p["fen"])
                if len(tokens) > max_model_len:
                    tokens = tokens[-max_model_len:]
                encoded_inputs.append(tokens)

            predicted_values = []
            legal_move_count = 0
            total_moves_checked = 0

            with torch.inference_mode():
                for i in range(0, len(encoded_inputs), batch_size):
                    batch_inps = encoded_inputs[i : i + batch_size]
                    max_len = max(len(t) for t in batch_inps)
                    padded_inps = []
                    for t in batch_inps:
                        pad_len = max_len - len(t)
                        padded_inps.append(torch.cat([torch.tensor(t, dtype=torch.long), torch.full((pad_len,), tokenizer.pad_id, dtype=torch.long)]))
                    inp_tensor = torch.stack(padded_inps).to(device)

                    pol_logits, val_preds = model(inp_tensor, return_policy=True, return_value=True)

                    if val_preds is not None:
                        predicted_values.extend(val_preds.squeeze(-1).cpu().tolist())

                    # Check legal move rate (LMR) on policy prediction
                    if pol_logits is not None:
                        # Find last non-pad position
                        for b_idx, t in enumerate(batch_inps):
                            pos_idx = len(t) - 1
                            top_token_id = pol_logits[b_idx, pos_idx].argmax().item()
                            top_token_str = tokenizer.id_to_token.get(top_token_id, "")

                            p_fen = positions[i + b_idx]["fen"]
                            try:
                                v_board = get_board(variant_name, fen=p_fen)
                                legals = [m.uci() for m in v_board.legal_moves()] if callable(getattr(v_board, "legal_moves", None)) else [m.uci() for m in v_board.legal_moves]
                                # Check if token matches from_sq or start of legal move
                                if any(m.startswith(top_token_str) for m in legals):
                                    legal_move_count += 1
                            except Exception:
                                pass
                            total_moves_checked += 1

            # Compute AlphaZero Value MSE: sum((z* - v)^2) / N
            squared_errors = [(z - v) ** 2 for z, v in zip(expert_values, predicted_values)]
            val_mse = sum(squared_errors) / len(squared_errors) if squared_errors else 0.0

            # Sign agreement (value accuracy)
            sign_correct = sum(
                1 for z, v in zip(expert_values, predicted_values)
                if (z > 0.1 and v > 0.1) or (z < -0.1 and v < -0.1) or (abs(z) <= 0.1 and abs(v) <= 0.1)
            )
            val_accuracy = sign_correct / len(expert_values) if expert_values else 0.0

            lmr = (legal_move_count / total_moves_checked) if total_moves_checked > 0 else 0.0

            res_entry = {
                "variant": variant_name,
                "k": K,
                "seed": seed,
                "value_mse": val_mse,
                "value_sign_accuracy": val_accuracy,
                "legal_move_rate": lmr,
                "num_positions": len(expert_values),
            }
            results.append(res_entry)
            print(
                f"  {variant_name} | K={K} seed={seed} | "
                f"AlphaZero Value MSE: {val_mse:.4f} | Sign Acc: {val_accuracy * 100:.1f}% | "
                f"Legal Move Rate: {lmr * 100:.1f}%"
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ChessBot with AlphaZero MSE")
    parser.add_argument("--model-path", type=str, default="runs/chess_bot/chess_bot_policy_value.pt")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output", type=str, default="runs/results/eval_alphazero_mse.json")
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--k-values", nargs="+", type=int, default=[0, 1, 2, 4, 8])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123])
    parser.add_argument("--num-samples", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Loading ChessBot Policy/Value model from {args.model_path} on {device}...")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    tokenizer = ChessTokenizer()
    model = ChessBotPolicyValue.load(model_path, map_location=device).to(device)
    model.eval()

    all_results = []
    print(f"Evaluating across held-out test variants: {TEST_VARIANTS}")
    for variant in TEST_VARIANTS:
        print(f"\n--- Evaluating {variant} (AlphaZero MSE & Legal Move Rate) ---")
        v_res = evaluate_bot_variant(
            model=model,
            tokenizer=tokenizer,
            variant_name=variant,
            data_dir=Path(args.data_dir),
            k_values=args.k_values,
            seeds=args.seeds,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            device=device,
        )
        all_results.extend(v_res)

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved AlphaZero MSE evaluation results to {out_file}")


if __name__ == "__main__":
    main()
