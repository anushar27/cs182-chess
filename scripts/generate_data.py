"""Data generation CLI for standard chess and chess variants.

Generates self-play games and evaluation positions for ICL experiments.
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Optional

import chess
from chess_icl.variants import VARIANTS, get_board


PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def evaluate_material(board: chess.Board, color: chess.Color) -> int:
    """Evaluate material advantage from the perspective of `color`."""
    score = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            val = PIECE_VALUES.get(piece.piece_type, 0)
            if piece.color == color:
                score += val
            else:
                score -= val
    return score


def choose_move(board, policy: str = "heuristic", rng: Optional[random.Random] = None) -> Optional[chess.Move]:
    """Select a move according to the given policy."""
    if rng is None:
        rng = random.Random()

    legal = list(board.legal_moves()) if callable(getattr(board, "legal_moves", None)) else list(board.legal_moves)
    if not legal:
        return None

    if policy == "random":
        return rng.choice(legal)

    # Fast heuristic: material-greedy policy
    # Prioritizes high-value captures with random tie-breaking among equal-value moves
    inner_board = getattr(board, "board", board)
    best_score = -1
    best_moves = []

    for move in legal:
        target_piece = inner_board.piece_at(move.to_square)
        score = PIECE_VALUES.get(target_piece.piece_type, 0) if target_piece else 0
        if move.promotion:
            score += PIECE_VALUES.get(move.promotion, 0) - 1

        if score > best_score:
            best_score = score
            best_moves = [move]
        elif score == best_score:
            best_moves.append(move)

    return rng.choice(best_moves) if best_moves else rng.choice(legal)


def play_game(
    variant_name: str,
    policy: str = "heuristic",
    max_moves: int = 100,
    rng: Optional[random.Random] = None,
    seed: int = 0,
) -> tuple[dict, list[dict]]:
    """Play a single self-play game and collect positions at move intervals."""
    if rng is None:
        rng = random.Random(seed)

    if variant_name == "standard":
        class StandardBoardWrapper:
            def __init__(self):
                self.board = chess.Board()
                self._move_count = 0
                self.variant_name = "standard"

            @property
            def turn(self):
                return self.board.turn

            def legal_moves(self):
                return list(self.board.legal_moves)

            def push(self, move):
                self.board.push(move)
                self._move_count += 1

            def is_game_over(self):
                return self.board.is_game_over(claim_draw=True) or self._move_count >= max_moves

            def result_for_white(self):
                r = self.board.result(claim_draw=True)
                if r == "1-0":
                    return "win"
                if r == "0-1":
                    return "loss"
                return "draw"

            def fen(self):
                return self.board.fen()

            def copy(self):
                import copy
                return copy.deepcopy(self)

        board = StandardBoardWrapper()
    else:
        board = get_board(variant_name, seed=seed)

    moves_played = []
    positions = []

    move_num = 0
    while not board.is_game_over() and move_num < max_moves:
        # Sample position at intervals (e.g. 10, 20, 30...)
        if move_num > 0 and move_num % 10 == 0:
            positions.append({
                "fen": board.fen(),
                "move_number": move_num,
                "side_to_move": "w" if board.turn == chess.WHITE else "b",
            })

        move = choose_move(board, policy=policy, rng=rng)
        if move is None:
            break

        moves_played.append(move.uci())
        board.push(move)
        move_num += 1

    white_result = board.result_for_white()

    game_record = {
        "game_id": f"{variant_name}_{policy}_{seed}",
        "variant": variant_name,
        "moves": moves_played,
        "result": white_result,
        "policy": policy,
        "move_count": len(moves_played),
    }

    # Label positions from the perspective of side to move
    labeled_positions = []
    for pos in positions:
        side = pos["side_to_move"]
        if white_result == "draw":
            lbl = "draw"
        elif white_result == "win":
            lbl = "win" if side == "w" else "loss"
        else:
            lbl = "loss" if side == "w" else "win"

        labeled_positions.append({
            "fen": pos["fen"],
            "label": lbl,
            "source_game": game_record["game_id"],
            "move_number": pos["move_number"],
        })

    return game_record, labeled_positions


def generate_variant_dataset(
    variant_name: str,
    output_dir: Path,
    heuristic_count: int = 2000,
    random_count: int = 500,
    eval_pos_count: int = 500,
    seed: int = 42,
):
    """Generate demonstration games and evaluation positions for a variant."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    print(f"[{variant_name}] Generating {heuristic_count} heuristic demo games...")
    heur_games = []
    all_eval_positions = []

    for i in range(heuristic_count):
        g_seed = seed + i
        game, positions = play_game(variant_name, policy="heuristic", seed=g_seed, rng=rng)
        heur_games.append(game)
        all_eval_positions.extend(positions)

    # Save heuristic games
    with open(output_dir / "games_demo.jsonl", "w", encoding="utf-8") as f:
        for g in heur_games:
            f.write(json.dumps(g) + "\n")

    print(f"[{variant_name}] Generating {random_count} random demo games...")
    rand_games = []
    for i in range(random_count):
        g_seed = seed + 100000 + i
        game, _ = play_game(variant_name, policy="random", seed=g_seed, rng=rng)
        rand_games.append(game)

    with open(output_dir / "games_demo_weak.jsonl", "w", encoding="utf-8") as f:
        for g in rand_games:
            f.write(json.dumps(g) + "\n")

    # Sample held-out evaluation positions
    if len(all_eval_positions) > eval_pos_count:
        eval_positions = rng.sample(all_eval_positions, eval_pos_count)
    else:
        eval_positions = all_eval_positions

    with open(output_dir / "positions_eval.jsonl", "w", encoding="utf-8") as f:
        for p in eval_positions:
            f.write(json.dumps(p) + "\n")

    metadata = {
        "variant": variant_name,
        "heuristic_games": len(heur_games),
        "random_games": len(rand_games),
        "eval_positions": len(eval_positions),
        "seed": seed,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[{variant_name}] Complete: {len(heur_games)} heur, {len(rand_games)} rand, {len(eval_positions)} eval pos.")


def generate_standard_dataset(
    output_dir: Path,
    count: int = 10000,
    seed: int = 42,
    pgn_path: Optional[str] = None,
):
    """Generate standard chess pretraining and validation data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    all_positions = []
    all_games = []

    if pgn_path and Path(pgn_path).exists():
        print(f"[standard] Parsing PGN from {pgn_path}...")
        with open(pgn_path) as pgn_file:
            game_idx = 0
            while game_idx < count:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                board = game.board()
                result_str = game.headers.get("Result", "*")
                if result_str == "1-0":
                    res = "win"
                elif result_str == "0-1":
                    res = "loss"
                elif result_str == "1/2-1/2":
                    res = "draw"
                else:
                    continue

                moves = []
                move_num = 0
                for move in game.mainline_moves():
                    if move_num > 0 and move_num % 10 == 0:
                        side = "w" if board.turn == chess.WHITE else "b"
                        if res == "draw":
                            lbl = "draw"
                        elif res == "win":
                            lbl = "win" if side == "w" else "loss"
                        else:
                            lbl = "loss" if side == "w" else "win"
                        all_positions.append({
                            "fen": board.fen(),
                            "label": lbl,
                            "source_game": f"lichess_{game_idx}",
                            "move_number": move_num,
                        })
                    board.push(move)
                    moves.append(move.uci())
                    move_num += 1

                all_games.append({
                    "game_id": f"lichess_{game_idx}",
                    "variant": "standard",
                    "moves": moves,
                    "result": res,
                    "policy": "human",
                    "move_count": len(moves),
                })
                game_idx += 1
    else:
        print(f"[standard] Generating {count} standard self-play games (heuristic)...")
        for i in range(count):
            g_seed = seed + i
            game, positions = play_game("standard", policy="heuristic", seed=g_seed, rng=rng)
            all_games.append(game)
            all_positions.extend(positions)

    # Train / Val split (85% train, 15% val)
    rng.shuffle(all_positions)
    split_idx = int(len(all_positions) * 0.85)
    train_pos = all_positions[:split_idx]
    val_pos = all_positions[split_idx:]

    with open(output_dir / "positions_train.jsonl", "w", encoding="utf-8") as f:
        for p in train_pos:
            f.write(json.dumps(p) + "\n")

    with open(output_dir / "positions_val.jsonl", "w", encoding="utf-8") as f:
        for p in val_pos:
            f.write(json.dumps(p) + "\n")

    with open(output_dir / "games_train.jsonl", "w", encoding="utf-8") as f:
        for g in all_games:
            f.write(json.dumps(g) + "\n")

    print(f"[standard] Complete: {len(train_pos)} train pos, {len(val_pos)} val pos, {len(all_games)} games.")


def main():
    parser = argparse.ArgumentParser(description="Generate chess ICL datasets.")
    parser.add_argument("--variant", type=str, default="standard", help="Variant name or 'standard' or 'all'")
    parser.add_argument("--all-variants", action="store_true", help="Generate all 13 variants")
    parser.add_argument("--source", type=str, default="selfplay", choices=["selfplay", "lichess"])
    parser.add_argument("--pgn-path", type=str, default=None, help="Path to PGN file if source=lichess")
    parser.add_argument("--count", type=int, default=2000, help="Number of games to generate")
    parser.add_argument("--policy", type=str, default="heuristic", choices=["heuristic", "random"])
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    out_base = Path(args.output_dir)

    if args.all_variants or args.variant == "all":
        # Generate standard first
        generate_standard_dataset(out_base / "standard", count=max(args.count * 2, 5000), seed=args.seed)
        for var_name in VARIANTS:
            generate_variant_dataset(
                var_name,
                out_base / var_name,
                heuristic_count=args.count,
                random_count=min(500, args.count // 4),
                eval_pos_count=500,
                seed=args.seed,
            )
    elif args.variant == "standard":
        generate_standard_dataset(out_base / "standard", count=args.count, seed=args.seed, pgn_path=args.pgn_path)
    else:
        generate_variant_dataset(
            args.variant,
            out_base / args.variant,
            heuristic_count=args.count,
            random_count=500 if args.policy == "heuristic" else args.count,
            eval_pos_count=500,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
