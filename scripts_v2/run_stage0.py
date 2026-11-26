#!/usr/bin/env python3
"""Stage 0 CPU Diagnostics across all 14 variants and 3 rollout policies.

Benchmarks `uniform_legal_v1`, `terminal_aware_v1`, and `search_d1_v1` across
`standard` + all 13 single variants:
1. Plays 60 full games per (policy, variant) up to max_plies=320.
2. Samples 80 unique midgame roots (ply >= 8, balanced White/Black to move) per
   (policy, variant) and runs R=16 continuations per root (= 17,920 continuations
   per policy).
3. Computes ms/ply, mean_plies, horizon_rate, draw_rate, decisive_rate, std(p_win),
   irreducible entropy, and MC noise floor.
4. Evaluates a shared cohort of 80 standard-legal midgame positions (valid across
   all 14 variants) to measure the Rule-Sensitivity / Context Gain Ceiling:
   ceiling_gain = NLL(K=0 mixture) - mean_v NLL(p_v).
5. Saves results to `runs_v2/stage0/stage0_metrics.json` and
   `runs_v2/stage0/sample_games.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import chess
import numpy as np

from chess_icl_v2.rollouts import POLICY_IDS, play_full_game, rollout_continuations
from chess_icl_v2.variants import SINGLE_VARIANTS, get_board

STAGE0_VARIANTS: list[str] = ["standard"] + list(SINGLE_VARIANTS)
EPS = 1e-12


def _entropy(probs: list[float]) -> float:
    return -sum(p * math.log(max(p, EPS)) for p in probs if p > 0.0)


def _cross_entropy(target: list[float], pred: list[float]) -> float:
    return -sum(t * math.log(max(q, EPS)) for t, q in zip(target, pred) if t > 0.0)


def _play_game_worker(task: tuple[str, str, int, int]) -> dict[str, Any]:
    variant_name, policy_id, max_plies, seed = task
    t0 = time.perf_counter()
    game = play_full_game(
        variant_name=variant_name,
        policy_id=policy_id,
        max_plies=max_plies,
        seed=seed,
        collect_san=True,
        collect_states=True,
        min_sample_ply=8,
    )
    game["elapsed_s"] = time.perf_counter() - t0
    return game


def _rollout_root_worker(
    task: tuple[str, dict[str, Any], str, int, int, int]
) -> dict[str, Any]:
    variant_name, state_dict, policy_id, r_count, max_plies, seed = task
    t0 = time.perf_counter()
    res = rollout_continuations(
        variant_name=variant_name,
        state_dict=state_dict,
        policy_id=policy_id,
        R=r_count,
        max_plies=max_plies,
        seed=seed,
    )
    res["elapsed_s"] = time.perf_counter() - t0
    return res


def _sample_roots_from_games(
    games: list[dict[str, Any]],
    variant_name: str,
    policy_id: str,
    num_roots: int,
    max_plies: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample `num_roots` unique ply>=8 states, balanced White/Black where possible."""
    rng = random.Random(seed)
    white_pool: dict[str, dict[str, Any]] = {}
    black_pool: dict[str, dict[str, Any]] = {}

    def _ingest(game_list: list[dict[str, Any]]) -> None:
        for g in game_list:
            for _ply, sd in g.get("states_at_plies", []):
                # Deduplicate by variant+fen+blocked
                key = f"{sd['fen']}|{sd.get('blocked', [])}"
                if sd["turn"]:
                    white_pool.setdefault(key, sd)
                else:
                    black_pool.setdefault(key, sd)

    _ingest(games)

    # Backfill extra games if a short variant (e.g. antichess/knight_army) had < num_roots
    extra_idx = 0
    while len(white_pool) + len(black_pool) < num_roots and extra_idx < 120:
        g = play_full_game(
            variant_name=variant_name,
            policy_id=policy_id,
            max_plies=max_plies,
            seed=seed + 50000 + extra_idx,
            collect_san=False,
            collect_states=True,
            min_sample_ply=8,
        )
        _ingest([g])
        extra_idx += 1

    w_list = list(white_pool.values())
    b_list = list(black_pool.values())
    rng.shuffle(w_list)
    rng.shuffle(b_list)

    half = num_roots // 2
    selected_w = w_list[:half]
    selected_b = b_list[: num_roots - len(selected_w)]
    if len(selected_w) + len(selected_b) < num_roots:
        rem = w_list[len(selected_w) :] + b_list[len(selected_b) :]
        rng.shuffle(rem)
        selected = selected_w + selected_b + rem[: num_roots - len(selected_w) - len(selected_b)]
    else:
        selected = selected_w + selected_b
    return selected[:num_roots]


def _build_shared_cohort(num_roots: int = 80, seed: int = 2026) -> list[dict[str, Any]]:
    """Sample `num_roots` standard-legal midgame FENs valid across ALL 14 variants."""
    rng = random.Random(seed)
    blocked_0 = set(get_board("blocked_squares", seed=0).blocked)
    check_variants = [v for v in STAGE0_VARIANTS if v != "antichess"]

    pool: dict[str, dict[str, Any]] = {}
    game_idx = 0
    while len(pool) < num_roots * 2 and game_idx < 120:
        g = play_full_game(
            variant_name="standard",
            policy_id="terminal_aware_v1",
            max_plies=160,
            seed=900000 + game_idx,
            collect_san=False,
            collect_states=True,
            min_sample_ply=10,
        )
        game_idx += 1
        for ply, sd in g.get("states_at_plies", []):
            if ply > 80:
                continue
            fen = sd["fen"]
            if fen in pool:
                continue
            b_std = get_board("standard", fen=fen)
            # Exclude positions with pieces sitting on blocked_squares(seed=0)
            if any(sq in blocked_0 for sq in b_std.piece_map().keys()):
                continue
            # Ensure non-mover is not already in check under ANY variant's attack rules
            valid_all = True
            for v in check_variants:
                bv = get_board(v, fen=fen, seed=0)
                if bv.was_into_check() or bv.is_game_over():
                    valid_all = False
                    break
            if valid_all:
                pool[fen] = sd

    whites = [sd for sd in pool.values() if sd["turn"]]
    blacks = [sd for sd in pool.values() if not sd["turn"]]
    rng.shuffle(whites)
    rng.shuffle(blacks)
    half = num_roots // 2
    sel = whites[:half] + blacks[: num_roots - half]
    if len(sel) < num_roots:
        rem = [sd for sd in pool.values() if sd not in sel]
        rng.shuffle(rem)
        sel.extend(rem[: num_roots - len(sel)])
    return sel[:num_roots]


def _summarize_variant_policy(
    games: list[dict[str, Any]],
    root_results: list[dict[str, Any]],
    r_count: int,
) -> dict[str, Any]:
    total_cont_plies = sum(r["total_plies"] for r in root_results)
    total_cont_time_s = sum(r["elapsed_s"] for r in root_results)
    total_game_plies = sum(g["move_count"] for g in games)
    total_game_time_s = sum(g["elapsed_s"] for g in games)

    probs = np.array([r["label_probs"] for r in root_results], dtype=np.float64)
    p_win = probs[:, 0]
    p_draw = probs[:, 1]
    p_loss = probs[:, 2]

    entropies = [_entropy(r["label_probs"]) for r in root_results]
    brier_noise = [
        sum(p * (1.0 - p) for p in r["label_probs"]) / max(1, r_count - 1)
        for r in root_results
    ]
    nll_excess = [
        (sum(1 for p in r["label_probs"] if p > 0.0) - 1) / (2.0 * r_count)
        for r in root_results
    ]

    return {
        "num_games": len(games),
        "num_roots": len(root_results),
        "R": r_count,
        "game_mean_plies": float(np.mean([g["move_count"] for g in games])),
        "game_horizon_rate": float(np.mean([1.0 if g["hit_horizon"] else 0.0 for g in games])),
        "game_decisive_rate": float(np.mean([0.0 if g["result"] == "draw" else 1.0 for g in games])),
        "mean_plies": float(np.mean([r["mean_plies"] for r in root_results])),
        "horizon_rate": float(np.mean([r["horizon_rate"] for r in root_results])),
        "draw_rate": float(np.mean(p_draw)),
        "decisive_rate": float(np.mean(p_win + p_loss)),
        "mean_p_win": float(np.mean(p_win)),
        "mean_p_loss": float(np.mean(p_loss)),
        "std_p_win": float(np.std(p_win)),
        "irreducible_entropy_nats": float(np.mean(entropies)),
        "mc_brier_noise_floor": float(np.mean(brier_noise)),
        "mc_nll_excess_floor": float(np.mean(nll_excess)),
        "ms_per_ply": float((total_cont_time_s * 1000.0) / max(1, total_cont_plies)),
        "game_ms_per_ply": float((total_game_time_s * 1000.0) / max(1, total_game_plies)),
        "total_continuation_plies": int(total_cont_plies),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 0 diagnostics for chess_icl_v2.")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--roots", type=int, default=80)
    parser.add_argument("--R", type=int, default=16)
    parser.add_argument("--max-plies", type=int, default=320)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-dir", type=Path, default=Path("runs_v2/stage0"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    print(
        f"Stage 0 Diagnostics: {len(STAGE0_VARIANTS)} variants x {len(POLICY_IDS)} policies | "
        f"games={args.games}, roots={args.roots}, R={args.R}, max_plies={args.max_plies}, workers={args.workers}",
        flush=True,
    )

    # 1. Build shared 80-root cohort first
    shared_cohort = _build_shared_cohort(num_roots=args.roots, seed=args.seed)
    print(f"Shared cohort built: {len(shared_cohort)} standard-legal midgame roots valid across all 14 variants.", flush=True)

    metrics_by_policy: dict[str, dict[str, Any]] = {}
    sample_games_records: list[dict[str, Any]] = []

    with mp.Pool(processes=args.workers) as pool:
        for p_idx, policy_id in enumerate(POLICY_IDS):
            tp0 = time.perf_counter()
            print(f"\n--- Policy [{p_idx+1}/{len(POLICY_IDS)}]: {policy_id} ---", flush=True)

            # A. Play full games in parallel
            game_tasks = [
                (v, policy_id, args.max_plies, args.seed * 1000 + v_idx * 100 + g_idx)
                for v_idx, v in enumerate(STAGE0_VARIANTS)
                for g_idx in range(args.games)
            ]
            all_games = pool.map(_play_game_worker, game_tasks)

            games_by_variant: dict[str, list[dict[str, Any]]] = {v: [] for v in STAGE0_VARIANTS}
            for g in all_games:
                games_by_variant[g["variant"]].append(g)

            # Save game records (without heavy states_at_plies) for token measurement
            for v in STAGE0_VARIANTS:
                for g in games_by_variant[v]:
                    rec = {k: val for k, val in g.items() if k != "states_at_plies"}
                    sample_games_records.append(rec)

            # B. Sample 80 roots per variant & launch R=16 continuation tasks
            cont_tasks: list[tuple[str, dict[str, Any], str, int, int, int]] = []
            for v_idx, v in enumerate(STAGE0_VARIANTS):
                roots = _sample_roots_from_games(
                    games_by_variant[v],
                    variant_name=v,
                    policy_id=policy_id,
                    num_roots=args.roots,
                    max_plies=args.max_plies,
                    seed=args.seed + v_idx * 1000 + p_idx * 100000,
                )
                for r_idx, sd in enumerate(roots):
                    cont_tasks.append(
                        (v, sd, policy_id, args.R, args.max_plies, args.seed * 100 + v_idx * 1000 + r_idx)
                    )

            cont_results = pool.map(_rollout_root_worker, cont_tasks, chunksize=4)
            cont_by_variant: dict[str, list[dict[str, Any]]] = {v: [] for v in STAGE0_VARIANTS}
            for r in cont_results:
                cont_by_variant[r["variant"]].append(r)

            per_variant_stats: dict[str, Any] = {}
            for v in STAGE0_VARIANTS:
                per_variant_stats[v] = _summarize_variant_policy(
                    games_by_variant[v], cont_by_variant[v], args.R
                )

            # C. Shared-cohort Rule-Sensitivity / Context Gain Ceiling
            shared_tasks = [
                (
                    v,
                    {**sd, "variant": v},
                    policy_id,
                    args.R,
                    args.max_plies,
                    args.seed * 777 + v_idx * 1000 + q_idx,
                )
                for v_idx, v in enumerate(STAGE0_VARIANTS)
                for q_idx, sd in enumerate(shared_cohort)
            ]
            shared_results = pool.map(_rollout_root_worker, shared_tasks, chunksize=4)

            # Index shared_results[v][q]
            shared_probs: dict[str, list[list[float]]] = {v: [] for v in STAGE0_VARIANTS}
            for idx, r in enumerate(shared_results):
                shared_probs[r["variant"]].append(r["label_probs"])

            # K=0 mixture across 14 variants per shared root q
            mixture_probs: list[list[float]] = []
            for q_idx in range(len(shared_cohort)):
                avg = [
                    sum(shared_probs[v][q_idx][c] for v in STAGE0_VARIANTS) / len(STAGE0_VARIANTS)
                    for c in range(3)
                ]
                mixture_probs.append(avg)

            per_var_ceiling: dict[str, float] = {}
            nll_k0_list: list[float] = []
            nll_oracle_list: list[float] = []
            for v in STAGE0_VARIANTS:
                v_k0 = [
                    _cross_entropy(shared_probs[v][q_idx], mixture_probs[q_idx])
                    for q_idx in range(len(shared_cohort))
                ]
                v_or = [_entropy(shared_probs[v][q_idx]) for q_idx in range(len(shared_cohort))]
                per_var_ceiling[v] = float(np.mean(v_k0) - np.mean(v_or))
                nll_k0_list.extend(v_k0)
                nll_oracle_list.extend(v_or)

            macro_summary = {
                "mean_ms_per_ply": float(np.mean([s["ms_per_ply"] for s in per_variant_stats.values()])),
                "mean_plies": float(np.mean([s["mean_plies"] for s in per_variant_stats.values()])),
                "mean_horizon_rate": float(np.mean([s["horizon_rate"] for s in per_variant_stats.values()])),
                "max_horizon_rate": float(np.max([s["horizon_rate"] for s in per_variant_stats.values()])),
                "mean_draw_rate": float(np.mean([s["draw_rate"] for s in per_variant_stats.values()])),
                "mean_decisive_rate": float(np.mean([s["decisive_rate"] for s in per_variant_stats.values()])),
                "mean_std_p_win": float(np.mean([s["std_p_win"] for s in per_variant_stats.values()])),
                "mean_irreducible_entropy_nats": float(
                    np.mean([s["irreducible_entropy_nats"] for s in per_variant_stats.values()])
                ),
                "mean_mc_brier_noise_floor": float(
                    np.mean([s["mc_brier_noise_floor"] for s in per_variant_stats.values()])
                ),
                "mean_mc_nll_excess_floor": float(
                    np.mean([s["mc_nll_excess_floor"] for s in per_variant_stats.values()])
                ),
                "shared_cohort_nll_k0_mixture": float(np.mean(nll_k0_list)),
                "shared_cohort_nll_oracle": float(np.mean(nll_oracle_list)),
                "shared_cohort_ceiling_gain_nats": float(np.mean(nll_k0_list) - np.mean(nll_oracle_list)),
                "shared_cohort_per_variant_ceiling_nats": per_var_ceiling,
                "wall_seconds": float(time.perf_counter() - tp0),
            }

            metrics_by_policy[policy_id] = {
                "macro": macro_summary,
                "per_variant": per_variant_stats,
            }

            print(
                f"Done {policy_id} in {macro_summary['wall_seconds']:.1f}s | "
                f"ms/ply={macro_summary['mean_ms_per_ply']:.3f} | "
                f"horizon={macro_summary['mean_horizon_rate']:.3f} (max={macro_summary['max_horizon_rate']:.3f}) | "
                f"decisive={macro_summary['mean_decisive_rate']:.3f} | "
                f"std(pW)={macro_summary['mean_std_p_win']:.3f} | "
                f"ceiling_gain={macro_summary['shared_cohort_ceiling_gain_nats']:.4f} nats",
                flush=True,
            )

    # Save JSONL sample games & metrics JSON
    games_path = args.out_dir / "sample_games.jsonl"
    with games_path.open("w", encoding="utf-8") as f:
        for rec in sample_games_records:
            f.write(json.dumps(rec) + "\n")

    payload = {
        "config": {
            "variants": STAGE0_VARIANTS,
            "policies": list(POLICY_IDS),
            "games_per_variant": args.games,
            "roots_per_variant": args.roots,
            "R": args.R,
            "max_plies": args.max_plies,
            "workers": args.workers,
            "seed": args.seed,
            "shared_cohort_size": len(shared_cohort),
            "total_wall_seconds": float(time.perf_counter() - t_start),
        },
        "policies": metrics_by_policy,
    }
    metrics_path = args.out_dir / "stage0_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {metrics_path} and {games_path} ({len(sample_games_records)} games).", flush=True)


if __name__ == "__main__":
    main()
