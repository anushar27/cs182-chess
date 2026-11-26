"""Full-game demo generation and empirical continuation rollouts for chess_icl_v2."""

from __future__ import annotations

import random
from typing import Any, Mapping, Optional

import chess

from chess_icl_v2.variants import VariantBoard, get_board

from .policies import choose_move

RULES_VERSION = "research_rules_v2_royal_conversion"


def _outcome_to_result(
    outcome: Optional[chess.Outcome],
    perspective: chess.Color,
    *,
    unresolved_as_draw: bool = True,
) -> tuple[str, bool, str]:
    """Map chess.Outcome (or None on horizon) to (result_str, hit_horizon, termination_str)."""
    if outcome is None:
        return ("draw" if unresolved_as_draw else "unresolved"), True, "horizon"
    term = outcome.termination.name.lower()
    if outcome.winner is None:
        return "draw", False, term
    return ("win" if outcome.winner == perspective else "loss"), False, term


def _resolve_budget_limits(
    initial_plies: int,
    extend_step_plies: int,
    max_total_plies: Optional[int],
    max_extensions: int,
    extend_until_terminal: bool = False,
) -> tuple[int, int]:
    """Compute (effective_max_total_plies, effective_max_extensions) for same-stream budgets."""
    if initial_plies <= 0:
        raise ValueError(f"max_plies must be positive, got {initial_plies}")
    step = max(0, int(extend_step_plies))
    ext_cap = max(0, int(max_extensions))

    if not extend_until_terminal and ext_cap == 0:
        return initial_plies, 0

    if max_total_plies is not None:
        total_cap = max(initial_plies, int(max_total_plies))
        return total_cap, ext_cap

    # Default extension horizon when max_total_plies is omitted
    effective_ext = ext_cap if ext_cap > 0 else 3
    if step <= 0:
        return initial_plies, 0
    return initial_plies + effective_ext * step, ext_cap


def _try_extend_budget(
    ply: int,
    current_budget: int,
    extensions_used: int,
    effective_max_total: int,
    effective_max_extensions: int,
    extend_step_plies: int,
    extend_until_terminal: bool,
) -> Optional[tuple[int, int]]:
    """Return updated (new_budget, new_extensions_used) if an administrative extension is allowed."""
    if not extend_until_terminal and effective_max_extensions <= 0:
        return None
    if ply >= effective_max_total:
        return None
    if effective_max_extensions > 0 and extensions_used >= effective_max_extensions:
        return None
    step = max(1, int(extend_step_plies))
    new_budget = min(effective_max_total, current_budget + step)
    if new_budget <= current_budget:
        return None
    return new_budget, extensions_used + 1


def play_full_game(
    variant_name: str,
    policy_id: str,
    max_plies: int = 320,
    seed: int = 0,
    collect_san: bool = True,
    collect_states: bool = False,
    min_sample_ply: int = 8,
    fen: Optional[str] = None,
    blocked: Optional[list[int]] = None,
    extend_until_terminal: bool = False,
    extend_step_plies: int = 2560,
    max_total_plies: Optional[int] = None,
    max_extensions: int = 0,
    unresolved_as_draw: bool = True,
    rules_version: str = RULES_VERSION,
) -> dict[str, Any]:
    """Play a complete game from the starting position (or `fen`) up to `max_plies`.

    Supports same-stream administrative budget extensions when `extend_until_terminal=True`
    or `max_extensions > 0`, continuing on the exact same `board` and `rng` state.
    """
    board = get_board(variant_name, fen=fen, seed=seed, blocked=blocked)
    init_fen = board.fen()
    blocked_list = sorted(board.blocked)
    rng = random.Random(seed)

    effective_max_total, effective_max_ext = _resolve_budget_limits(
        initial_plies=max_plies,
        extend_step_plies=extend_step_plies,
        max_total_plies=max_total_plies,
        max_extensions=max_extensions,
        extend_until_terminal=extend_until_terminal,
    )
    current_budget = max_plies
    extensions_used = 0

    moves_uci: list[str] = []
    moves_san: list[str] = []
    states_at_plies: list[tuple[int, dict[str, Any]]] = []

    outcome: Optional[chess.Outcome] = None
    ply = 0
    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            break

        if ply >= current_budget:
            extended = _try_extend_budget(
                ply=ply,
                current_budget=current_budget,
                extensions_used=extensions_used,
                effective_max_total=effective_max_total,
                effective_max_extensions=effective_max_ext,
                extend_step_plies=extend_step_plies,
                extend_until_terminal=extend_until_terminal,
            )
            if extended is None:
                break
            current_budget, extensions_used = extended

        if collect_states and ply >= min_sample_ply:
            states_at_plies.append((ply, board.to_state_dict()))

        move = choose_move(board, policy_id, rng)
        if move is None:
            outcome = board.outcome(claim_draw=True)
            break

        if collect_san:
            try:
                moves_san.append(board.san(move))
            except Exception:
                moves_san.append(move.uci())

        moves_uci.append(move.uci())
        board.push(move)
        ply += 1

    if outcome is None:
        outcome = board.outcome(claim_draw=True)

    is_terminal = outcome is not None
    result, hit_horizon, termination = _outcome_to_result(
        outcome, chess.WHITE, unresolved_as_draw=unresolved_as_draw
    )

    out: dict[str, Any] = {
        "variant": variant_name,
        "policy": policy_id,
        "rules_version": rules_version,
        "seed": seed,
        "init_fen": init_fen,
        "initial_fen": init_fen,
        "blocked_squares": blocked_list,
        "moves": moves_uci,
        "moves_uci": moves_uci,
        "moves_san": moves_san,
        "result": result,
        "outcome": result,
        "hit_horizon": hit_horizon,
        "is_terminal": is_terminal,
        "termination": termination,
        "move_count": len(moves_uci),
        "plies_budget_used": int(current_budget),
        "extensions_used": int(extensions_used),
    }
    if collect_states:
        out["states_at_plies"] = states_at_plies
    return out


def rollout_continuations(
    variant_name: str,
    state_dict: Mapping[str, Any],
    policy_id: str,
    R: int = 16,
    max_plies: int = 320,
    seed: int = 0,
    unresolved_as_draw: bool = True,
    allow_unresolved: Optional[bool] = None,
    extend_step_plies: int = 2560,
    max_total_plies: Optional[int] = None,
    max_extensions: int = 0,
    rules_version: str = RULES_VERSION,
) -> dict[str, Any]:
    """Run `R` independent continuations from `state_dict` with optional same-stream extensions.

    In legacy mode (`unresolved_as_draw=True`, default), horizon hits count as draws in
    `label_probs` and are excluded from `label_probs_censored`.
    In long-study mode (`unresolved_as_draw=False` or `allow_unresolved=True`), unfinished
    continuations increment `n_unresolved` and never become draws or renormalized targets:
    if `n_unresolved > 0`, `label_probs` is `None` and `target_bounds` gives exact
    worst-case lower/upper simplex bounds.
    """
    if R <= 0:
        raise ValueError(f"R must be positive, got {R}")

    if allow_unresolved is not None:
        if not unresolved_as_draw and not allow_unresolved:
            raise ValueError("Conflicting unresolved_as_draw=False and allow_unresolved=False")
        as_draw = not bool(allow_unresolved)
    else:
        as_draw = bool(unresolved_as_draw)

    effective_max_total, effective_max_ext = _resolve_budget_limits(
        initial_plies=max_plies,
        extend_step_plies=extend_step_plies,
        max_total_plies=max_total_plies,
        max_extensions=max_extensions,
        extend_until_terminal=(max_extensions > 0),
    )

    sd = dict(state_dict)
    if sd.get("variant") != variant_name:
        sd["variant"] = variant_name
    root_board = VariantBoard.from_state_dict(sd)
    root_turn = root_board.turn

    wins = 0
    draws = 0
    losses = 0
    non_h_wins = 0
    non_h_draws = 0
    non_h_losses = 0
    horizon_hits = 0
    n_unresolved = 0
    total_plies = 0
    max_budget_used = max_plies
    total_extensions = 0

    for r in range(R):
        board = root_board.copy(stack=False)
        rng = random.Random(seed * 100003 + r * 977 + 17)

        current_budget = max_plies
        extensions_used = 0
        outcome: Optional[chess.Outcome] = None
        plies = 0

        while True:
            outcome = board.outcome(claim_draw=True)
            if outcome is not None:
                break

            if plies >= current_budget:
                extended = _try_extend_budget(
                    ply=plies,
                    current_budget=current_budget,
                    extensions_used=extensions_used,
                    effective_max_total=effective_max_total,
                    effective_max_extensions=effective_max_ext,
                    extend_step_plies=extend_step_plies,
                    extend_until_terminal=(effective_max_ext > 0),
                )
                if extended is None:
                    break
                current_budget, extensions_used = extended

            move = choose_move(board, policy_id, rng)
            if move is None:
                outcome = board.outcome(claim_draw=True)
                break
            board.push(move)
            plies += 1

        if outcome is None:
            outcome = board.outcome(claim_draw=True)

        total_plies += plies
        if current_budget > max_budget_used:
            max_budget_used = current_budget
        total_extensions += extensions_used

        res, hit_h, _ = _outcome_to_result(
            outcome, root_turn, unresolved_as_draw=as_draw
        )
        if hit_h:
            horizon_hits += 1
            n_unresolved += 1
            if as_draw:
                draws += 1
        else:
            if res == "win":
                wins += 1
                non_h_wins += 1
            elif res == "loss":
                losses += 1
                non_h_losses += 1
            else:
                draws += 1
                non_h_draws += 1

    non_h_total = non_h_wins + non_h_draws + non_h_losses
    if as_draw:
        label_probs: Optional[list[float]] = [wins / R, draws / R, losses / R]
        label_probs_censored: Optional[list[float]] = (
            [non_h_wins / non_h_total, non_h_draws / non_h_total, non_h_losses / non_h_total]
            if non_h_total > 0
            else None
        )
        target_bounds = {
            "lower": [non_h_wins / R, non_h_draws / R, non_h_losses / R],
            "upper": [
                (non_h_wins + n_unresolved) / R,
                (non_h_draws + n_unresolved) / R,
                (non_h_losses + n_unresolved) / R,
            ],
        }
    else:
        lower = [wins / R, draws / R, losses / R]
        if n_unresolved > 0:
            label_probs = None
            label_probs_censored = None
            upper = [
                (wins + n_unresolved) / R,
                (draws + n_unresolved) / R,
                (losses + n_unresolved) / R,
            ]
        else:
            label_probs = list(lower)
            label_probs_censored = list(lower)
            upper = list(lower)
        target_bounds = {"lower": lower, "upper": upper}

    return {
        "variant": variant_name,
        "policy": policy_id,
        "rules_version": rules_version,
        "R": R,
        "root_turn": "w" if root_turn == chess.WHITE else "b",
        "label_probs": label_probs,
        "label_probs_censored": label_probs_censored,
        "target_bounds": target_bounds,
        "counts": [wins, draws, losses],
        "n_unresolved": n_unresolved,
        "horizon_hits": horizon_hits,
        "horizon_rate": n_unresolved / R,
        "decisive_rate": (wins + losses) / R,
        "mean_plies": total_plies / R,
        "total_plies": total_plies,
        "budget_used": int(max_budget_used),
        "extensions_used": int(total_extensions),
        "unresolved_as_draw": as_draw,
    }


def rollout_continuations_v2(
    variant_name: str,
    state_dict: Mapping[str, Any],
    policy_id: str = "terminal_aware_v1",
    R: int = 32,
    initial_budget: int = 2560,
    extend_step_plies: int = 2560,
    max_total_plies: int = 7680,
    max_extensions: int = 2,
    seed: int = 0,
    rules_version: str = RULES_VERSION,
) -> dict[str, Any]:
    """Long-study rollout wrapper with 2,560-ply budget, same-stream extensions, and no draw censoring."""
    return rollout_continuations(
        variant_name=variant_name,
        state_dict=state_dict,
        policy_id=policy_id,
        R=R,
        max_plies=initial_budget,
        seed=seed,
        unresolved_as_draw=False,
        allow_unresolved=True,
        extend_step_plies=extend_step_plies,
        max_total_plies=max_total_plies,
        max_extensions=max_extensions,
        rules_version=rules_version,
    )
