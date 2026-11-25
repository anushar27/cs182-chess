"""Generate demonstration games and balanced evaluation positions for compositional variants."""

import argparse
import json
import random
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chess_icl.variants.compositional import COMPOSITIONAL_RULES, get_compositional_board
from chess_icl.variants.engine import VariantBoard


def play_game(name: str, seed: int, max_moves: int = 100) -> dict:
    board = get_compositional_board(name, seed=seed)
    rng = random.Random(seed)
    moves = []
    
    while not board.is_game_over() and len(moves) < max_moves:
        legal = board.legal_moves()
        if not legal:
            break
        # Heuristic / random greedy move selection
        scored = []
        for m in legal:
            # Score capture
            cap = board.board.piece_at(m.to_square)
            score = 10 if cap is not None else 0
            scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_score = scored[0][0]
        best_moves = [m for s, m in scored if s == top_score]
        chosen = rng.choice(best_moves)
        
        moves.append(chosen.uci())
        board.push(chosen)
        
    result = board.result_for_white()
    return {
        "game_id": f"{name}_{seed}",
        "variant": name,
        "moves": moves,
        "result": result,
    }


def generate_compositional_data(output_dir: Path, n_demos: int = 600, n_eval: int = 600):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name in COMPOSITIONAL_RULES:
        v_dir = output_dir / name
        v_dir.mkdir(parents=True, exist_ok=True)
        print(f"Generating data for compositional variant: {name}...")
        
        # 1. Generate demo games
        demos = []
        for s in range(n_demos):
            demos.append(play_game(name, seed=s))
            
        with open(v_dir / "games_demo.jsonl", "w") as f:
            for g in demos:
                f.write(json.dumps(g) + "\n")
                
        # 2. Extract balanced positions: 200 Win, 200 Draw, 200 Loss
        target_per_class = n_eval // 3
        positions_by_class = {"win": [], "draw": [], "loss": []}
        
        seed = 10000
        while min(len(positions_by_class[c]) for c in ["win", "draw", "loss"]) < target_per_class:
            board = get_compositional_board(name, seed=seed)
            rng = random.Random(seed)
            seed += 1
            move_count = 0
            while not board.is_game_over() and move_count < 80:
                legal = board.legal_moves()
                if not legal:
                    break
                chosen = rng.choice(legal)
                board.push(chosen)
                move_count += 1
                
                # Check position material outcome
                if move_count >= 8:
                    fen = board.board.fen()
                    # Determine label from side to move
                    mat = 0
                    for sq, p in board.board.piece_map().items():
                        val = 1 if p.piece_type == 1 else (3 if p.piece_type in (2, 3) else (5 if p.piece_type == 4 else 9))
                        mat += val if p.color == board.board.turn else -val
                    
                    if name == "short_knight_and_antichess":
                        label = "win" if mat <= -3 else ("loss" if mat >= 3 else "draw")
                    else:
                        label = "win" if mat >= 3 else ("loss" if mat <= -3 else "draw")
                        
                    if len(positions_by_class[label]) < target_per_class:
                        positions_by_class[label].append({"fen": fen, "label": label})
                        
        eval_positions = []
        for c in ["win", "draw", "loss"]:
            eval_positions.extend(positions_by_class[c])
        rng = random.Random(42)
        rng.shuffle(eval_positions)
        
        with open(v_dir / "positions_eval.jsonl", "w") as f:
            for p in eval_positions:
                f.write(json.dumps(p) + "\n")
                
        print(f"[{name}] Generated {len(demos)} demo games and {len(eval_positions)} balanced eval positions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/compositional")
    parser.add_argument("--n-demos", type=int, default=300)
    parser.add_argument("--n-eval", type=int, default=600)
    args = parser.parse_args()
    generate_compositional_data(Path(args.output_dir), n_demos=args.n_demos, n_eval=args.n_eval)
