"""Rollout policies for chess_icl_v2 Stage 0 and empirical outcome labeling.

All policies are strictly variant-agnostic (zero hardcoded piece values):
- `uniform_legal_v1`: uniform random selection over legal moves.
- `terminal_aware_v1`: detects immediate win-in-1 moves via push/outcome/pop;
  otherwise falls back to uniform random over legal moves.
- `search_d1_v1`: 1-ply lookahead scoring immediate wins (+1000), opponent
  immediate win threats (-1000), draws (0), and variant-agnostic mobility
  difference 0.001 * (own_moves - opp_moves), breaking ties randomly.
"""

from __future__ import annotations

import random
from typing import Optional

import chess

from chess_icl_v2.variants import VariantBoard

POLICY_IDS: tuple[str, ...] = (
    "uniform_legal_v1",
    "terminal_aware_v1",
    "search_d1_v1",
)


def _is_immediate_win_for_mover(board: VariantBoard, mover: chess.Color) -> bool:
    """Fast exact check whether the move just pushed won immediately for `mover`."""
    if board.is_variant_win():
        # Opponent to move (`board.turn != mover`) just won immediately (e.g. antichess stalemate)
        return False
    if board.is_variant_loss():
        return board.turn != mover
    if not board._no_check_rules and board.is_check():
        return not any(board.generate_legal_moves())
    return False


def _choose_uniform(moves: list[chess.Move], rng: random.Random) -> chess.Move:
    return rng.choice(moves)


def _choose_terminal_aware(
    board: VariantBoard, moves: list[chess.Move], rng: random.Random
) -> chess.Move:
    mover = board.turn
    winning_moves: list[chess.Move] = []
    for move in moves:
        board.push(move)
        try:
            if _is_immediate_win_for_mover(board, mover):
                winning_moves.append(move)
        finally:
            board.pop()
    if winning_moves:
        return rng.choice(winning_moves)
    return rng.choice(moves)


def _choose_search_d1(
    board: VariantBoard, moves: list[chess.Move], rng: random.Random
) -> chess.Move:
    mover = board.turn
    own_moves = len(moves)
    no_check = board._no_check_rules

    best_score = float("-inf")
    best_moves: list[chess.Move] = []

    for move in moves:
        board.push(move)
        try:
            if _is_immediate_win_for_mover(board, mover):
                score = 1000.0
            elif board.is_variant_win():
                score = -1000.0
            elif board.is_variant_draw() or board.is_insufficient_material() or board.is_seventyfive_moves():
                score = 0.0
            else:
                opp_moves = list(board.generate_legal_moves())
                opp_count = len(opp_moves)
                if opp_count == 0:
                    # Antichess stalemate was already handled by is_variant_win() above
                    score = 0.0
                else:
                    opp_has_win = False
                    opp_color = board.turn
                    if not no_check:
                        for opp_move in opp_moves:
                            board.push(opp_move)
                            try:
                                if _is_immediate_win_for_mover(board, opp_color):
                                    opp_has_win = True
                                    break
                            finally:
                                board.pop()
                    if opp_has_win:
                        score = -1000.0
                    else:
                        score = 0.001 * (own_moves - opp_count)
        finally:
            board.pop()

        if score > best_score:
            best_score = score
            best_moves = [move]
        elif score == best_score:
            best_moves.append(move)

    return rng.choice(best_moves)


def choose_move(
    board: VariantBoard,
    policy_id: str,
    rng: random.Random,
) -> Optional[chess.Move]:
    """Select a legal move on `board` according to `policy_id`, or None if no legal moves exist."""
    moves = list(board.generate_legal_moves())
    if not moves:
        return None

    if policy_id == "uniform_legal_v1":
        return _choose_uniform(moves, rng)
    if policy_id == "terminal_aware_v1":
        return _choose_terminal_aware(board, moves, rng)
    if policy_id == "search_d1_v1":
        return _choose_search_d1(board, moves, rng)

    raise ValueError(f"Unknown policy_id: {policy_id!r}. Available: {POLICY_IDS}")
