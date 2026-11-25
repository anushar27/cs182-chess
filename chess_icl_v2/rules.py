"""Frozen v2 rules; legacy wrappers and material adjudication are never used."""
import chess
import chess.variant
from .simulator import NAMES, custom_board

RULES = {"standard": {"version": "python_chess_1.11.2_auto_draws_v1"},
         "antichess": {"version": "python_chess_1.11.2_antichess_auto_draws_v1"},
         **{name: {"version": "research_rules_v1"} for name in NAMES}}
RULES["piece_capture"]["version"] = "research_rules_v2_royal_conversion"
TRAIN_VARIANTS = ["diagonal_rook", "straight_bishop", "weakened_queen", "extended_pawn"]
HELDOUT_VARIANTS = [name for name in RULES if name not in TRAIN_VARIANTS + ["standard"]]
PENDING = []


def make_board(variant, fen=None, *, extras=None, seed=0):
    if variant not in RULES:
        raise ValueError(f"Unknown rules {variant}; long_board is renamed extended_initial_pawn")
    extras = extras or {}
    if variant in NAMES:
        if set(extras) - {"barriers"} or (extras and variant != "blocked_squares"):
            raise ValueError("Unsupported rule metadata")
        return custom_board(variant, fen, seed=seed, barriers=extras.get("barriers"))
    if extras:
        raise ValueError("Unsupported rule metadata")
    cls = chess.Board if variant == "standard" else chess.variant.AntichessBoard
    return cls(fen) if fen is not None else cls()


def outcome(board):
    return board.outcome(claim_draw=False)


def white_result(board):
    result = outcome(board)
    if result is None:
        return None
    return "D" if result.winner is None else ("W" if result.winner else "L")
