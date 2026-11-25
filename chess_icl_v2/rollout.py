from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import random
import multiprocessing
import time
from .common import seed_for
from .policy import choose
from .rules import white_result
from .state import State


def continuation(state: State, policy: str, seed: int, max_plies: int):
    if max_plies < 0:
        raise ValueError("max_plies must be nonnegative")
    b = state.board()
    root_white = b.turn
    rng = random.Random(seed)
    moves = []
    start = time.perf_counter()
    while white_result(b) is None and len(moves) < max_plies:
        move = choose(b, rng, policy)
        moves.append(move.uci())
        b.push(move)
    result = white_result(b)
    relative = result if root_white or result in (None, "D") else {"W": "L", "L": "W"}[result]
    return {"seed": seed, "policy_id": policy, "max_plies": max_plies,
            "moves": moves, "white_result": result, "root_result": relative,
            "status": "terminal" if result is not None else "unresolved",
            "termination": b.outcome().termination.name if result is not None else "ply_budget",
            "seconds": time.perf_counter() - start,
            "final_fen": b.fen(en_passant="fen")}


def _job(args):
    return continuation(*args)


def estimate(state, policy, replicates, max_plies, master_seed, workers=1):
    if replicates < 1 or workers < 1:
        raise ValueError("Need positive replicate and worker counts")
    jobs = [(state, policy, seed_for(master_seed, "label", state.key, r), max_plies)
            for r in range(replicates)]
    if workers == 1:
        runs = list(map(_job, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            runs = list(pool.map(_job, jobs))
    counts = [sum(r["root_result"] == c for r in runs) for c in ("W", "D", "L")]
    unresolved = replicates - sum(counts)
    target = [n / replicates for n in counts] if unresolved == 0 else None
    # Marginal identification intervals, not three independently attainable upper bounds.
    bounds = [[n / replicates, (n + unresolved) / replicates] for n in counts]
    return {"policy_id": policy, "replicates": replicates, "counts_wdl": counts,
            "unresolved": unresolved, "target_wdl": target, "bounds_wdl": bounds,
            "mc_standard_error": ([((p * (1 - p)) / replicates) ** .5 for p in target]
                                  if target is not None else None), "runs": runs}
