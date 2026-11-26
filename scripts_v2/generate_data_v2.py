"""Stage 1 parallel dataset generator for chess_icl_v2.

Generates full-game demonstrations and empirical rollout-labeled query roots
with strict source-game disjointness, canonical & Hamming near-duplicate
filtering, side-to-move balance, manifest tracking, and leakage auditing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Any, Optional, Sequence

import chess
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_icl_v2.data.splits import (  # noqa: E402
    audit_dataset_integrity,
    deduplicate_and_filter_roots,
    split_games_into_pools,
)
from chess_icl_v2.rollouts import play_full_game, rollout_continuations  # noqa: E402
from chess_icl_v2.variants import ALL_VARIANT_NAMES, VariantBoard  # noqa: E402

DEFAULT_VARIANTS: tuple[str, ...] = tuple(ALL_VARIANT_NAMES)


def _stable_int_seed(*parts: Any) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Multiprocessing Worker Functions (top-level for pickling)
# ---------------------------------------------------------------------------


def _worker_play_game(task: tuple[str, str, int, int, bool, int, str]) -> dict[str, Any]:
    variant_name, policy_id, max_plies, seed, collect_states, min_sample_ply, gid = task
    game = play_full_game(
        variant_name=variant_name,
        policy_id=policy_id,
        max_plies=max_plies,
        seed=seed,
        collect_san=True,
        collect_states=collect_states,
        min_sample_ply=min_sample_ply,
    )
    game["source_game_id"] = gid
    game["game_id"] = gid
    return game


def _worker_rollout_root(
    task: tuple[str, str, int, dict[str, Any], str, int, int, int],
) -> dict[str, Any]:
    variant_name, split_name, root_idx, root_item, policy_id, r_count, horizon, seed = task
    sd = root_item["state_dict"]
    rollout_res = rollout_continuations(
        variant_name=variant_name,
        state_dict=sd,
        policy_id=policy_id,
        R=r_count,
        max_plies=horizon,
        seed=seed,
    )

    board = VariantBoard.from_state_dict(sd)
    mat_diff_w = int(board.evaluate_material(chess.WHITE))
    eq_mat = abs(mat_diff_w) <= 2

    label_probs = rollout_res["label_probs"]
    label_names = ("win", "draw", "loss")
    argmax_lbl = label_names[int(np.argmax(label_probs))]

    return {
        "variant": variant_name,
        "split": split_name,
        "root_id": f"{variant_name}_{split_name}_{root_idx}",
        "source_game_id": root_item["source_game_id"],
        "game_id": root_item["source_game_id"],
        "ply": int(root_item["ply"]),
        "fen": sd["fen"],
        "board_fen_with_barriers": sd.get("board_fen_with_barriers", board.board_fen_with_barriers()),
        "blocked_squares": list(sd.get("blocked", [])),
        "state_dict": sd,
        "canonical_state_key": root_item["canonical_state_key"],
        "side_to_move": rollout_res["root_turn"],
        "material_diff_white": mat_diff_w,
        "equal_material": eq_mat,
        "y_true": label_probs,
        "label_probs": label_probs,
        "label_probs_censored": rollout_res["label_probs_censored"],
        "label": argmax_lbl,
        "R": int(r_count),
        "policy": policy_id,
        "horizon_rate": float(rollout_res["horizon_rate"]),
        "decisive_rate": float(rollout_res["decisive_rate"]),
        "mean_plies": float(rollout_res["mean_plies"]),
    }


# ---------------------------------------------------------------------------
# Root Sampling & Formatting Helpers
# ---------------------------------------------------------------------------


def _sample_balanced_candidate_roots(
    source_games: Sequence[dict[str, Any]],
    variant_name: str,
    max_candidates: int,
    seed: int,
    min_ply: int = 8,
) -> list[dict[str, Any]]:
    """Sample candidate query roots balanced across White/Black and middlegame/endgame plies."""
    rng = np.random.default_rng(seed)
    white_buckets: dict[str, list[dict[str, Any]]] = {"early": [], "mid": [], "late": []}
    black_buckets: dict[str, list[dict[str, Any]]] = {"early": [], "mid": [], "late": []}

    for game in source_games:
        gid = str(game["source_game_id"])
        for ply, sd in game.get("states_at_plies", []):
            if ply < min_ply:
                continue
            board = VariantBoard.from_state_dict(sd)
            if board.is_game_over(claim_draw=True):
                continue
            ckey = board.canonical_state_key()
            item = {
                "variant": variant_name,
                "source_game_id": gid,
                "ply": int(ply),
                "fen": sd["fen"],
                "blocked": list(sd.get("blocked", [])),
                "turn": bool(sd["turn"]),
                "state_dict": sd,
                "canonical_state_key": ckey,
            }
            phase = "early" if ply < 24 else ("mid" if ply < 50 else "late")
            if bool(sd["turn"]):
                white_buckets[phase].append(item)
            else:
                black_buckets[phase].append(item)

    def _interleave_buckets(buckets: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        for phase in buckets:
            lst = buckets[phase]
            if len(lst) > 1:
                perm = rng.permutation(len(lst))
                buckets[phase] = [lst[int(i)] for i in perm]
        # Cycle across mid, late, early to prioritize middlegame/endgame diversity
        order = ("mid", "late", "early")
        idxs = {p: 0 for p in order}
        merged: list[dict[str, Any]] = []
        while True:
            progressed = False
            for p in order:
                if idxs[p] < len(buckets[p]):
                    merged.append(buckets[p][idxs[p]])
                    idxs[p] += 1
                    progressed = True
            if not progressed:
                break
        return merged

    w_list = _interleave_buckets(white_buckets)
    b_list = _interleave_buckets(black_buckets)

    # Alternate White and Black so candidate stream stays balanced
    combined: list[dict[str, Any]] = []
    wi, bi = 0, 0
    while len(combined) < max_candidates and (wi < len(w_list) or bi < len(b_list)):
        if wi < len(w_list):
            combined.append(w_list[wi])
            wi += 1
        if len(combined) < max_candidates and bi < len(b_list):
            combined.append(b_list[bi])
            bi += 1
    return combined


def _select_balanced_subset(
    roots: Sequence[dict[str, Any]],
    target_n: int,
) -> list[dict[str, Any]]:
    """Trim deduplicated roots to at most `target_n` while preserving White/Black balance."""
    if len(roots) <= target_n:
        return list(roots)
    whites = [r for r in roots if bool(r["turn"])]
    blacks = [r for r in roots if not bool(r["turn"])]
    selected: list[dict[str, Any]] = []
    wi, bi = 0, 0
    while len(selected) < target_n and (wi < len(whites) or bi < len(blacks)):
        if wi < len(whites):
            selected.append(whites[wi])
            wi += 1
        if len(selected) < target_n and bi < len(blacks):
            selected.append(blacks[bi])
            bi += 1
    return selected


def _strip_game_for_demo_jsonl(game: dict[str, Any]) -> dict[str, Any]:
    """Drop per-ply state snapshots when writing compact games_demo.jsonl."""
    return {k: v for k, v in game.items() if k != "states_at_plies"}


def _compute_split_fractions(
    demo_games: int,
    source_games: int,
    train_roots: int,
    val_roots: int,
    test_roots: int,
) -> tuple[float, float, float, float]:
    total = demo_games + source_games
    if total <= 0:
        raise ValueError("demo_games + source_games must be > 0")
    demo_frac = demo_games / total
    source_frac = source_games / total
    r_tot = train_roots + val_roots + test_roots
    if r_tot <= 0:
        tr_w, va_w, te_w = 0.5, 0.25, 0.25
    else:
        tr_w = train_roots / r_tot
        va_w = val_roots / r_tot
        te_w = test_roots / r_tot
    train_frac = source_frac * tr_w
    val_frac = source_frac * va_w
    test_frac = max(0.0, 1.0 - demo_frac - train_frac - val_frac)
    return demo_frac, train_frac, val_frac, test_frac


# ---------------------------------------------------------------------------
# Main Stage 1 Dataset Pipeline
# ---------------------------------------------------------------------------


def generate_dataset(
    output_dir: Path,
    variants: Sequence[str],
    policy_id: str = "terminal_aware_v1",
    horizon: int = 320,
    demo_games: int = 200,
    source_games: int = 300,
    train_roots: int = 400,
    val_roots: int = 100,
    test_roots: int = 200,
    r_train: int = 16,
    r_eval: int = 32,
    max_similarity: float = 0.90,
    min_ply: int = 8,
    workers: int = 4,
    seed: int = 42,
    alt_demo_policy: Optional[str] = None,
) -> dict[str, Any]:
    """Generate full Stage 1 dataset with parallel game play, dedup, rollouts, and audit."""
    t_start = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variant_list = list(variants)

    demo_frac, train_frac, val_frac, test_frac = _compute_split_fractions(
        demo_games=demo_games,
        source_games=source_games,
        train_roots=train_roots,
        val_roots=val_roots,
        test_roots=test_roots,
    )

    total_games_per_var = demo_games + source_games
    game_tasks: list[tuple[str, str, int, int, bool, int, str]] = []
    for variant in variant_list:
        for i in range(total_games_per_var):
            g_seed = _stable_int_seed(seed, "game", variant, policy_id, i)
            gid = f"{variant}_g_{i}"
            game_tasks.append(
                (variant, policy_id, horizon, g_seed, True, min_ply, gid)
            )

    alt_tasks: list[tuple[str, str, int, int, bool, int, str]] = []
    if alt_demo_policy:
        for variant in variant_list:
            for i in range(demo_games):
                g_seed = _stable_int_seed(seed, "alt_demo", variant, alt_demo_policy, i)
                gid = f"{variant}_alt_demo_{i}"
                alt_tasks.append(
                    (variant, alt_demo_policy, horizon, g_seed, False, min_ply, gid)
                )

    with mp.Pool(processes=max(1, int(workers))) as pool:
        played_games = pool.map(_worker_play_game, game_tasks)
        played_alt_demos = pool.map(_worker_play_game, alt_tasks) if alt_tasks else []

        games_by_var: dict[str, list[dict[str, Any]]] = {v: [] for v in variant_list}
        for g in played_games:
            games_by_var[g["variant"]].append(g)

        alt_by_var: dict[str, list[dict[str, Any]]] = {v: [] for v in variant_list}
        for g in played_alt_demos:
            alt_by_var[g["variant"]].append(_strip_game_for_demo_jsonl(g))

        # Split pools, sample balanced roots, deduplicate & filter near-dups
        demos_by_var: dict[str, list[dict[str, Any]]] = {}
        dedup_stats_by_var: dict[str, Any] = {}
        rollout_tasks: list[tuple[str, str, int, dict[str, Any], str, int, int, int]] = []

        target_counts = {"train": train_roots, "val": val_roots, "test": test_roots}

        for variant in variant_list:
            pools = split_games_into_pools(
                games_by_var[variant],
                demo_frac=demo_frac,
                train_frac=train_frac,
                val_frac=val_frac,
                test_frac=test_frac,
                seed=_stable_int_seed(seed, "pool_split", variant),
            )
            demo_recs = [_strip_game_for_demo_jsonl(g) for g in pools["demo_games"]]
            demos_by_var[variant] = demo_recs

            # Oversample candidate roots 3x before dedup so final balanced targets can be met
            cand_by_split = {
                "train": _sample_balanced_candidate_roots(
                    pools["train_source_games"],
                    variant,
                    max_candidates=max(train_roots * 3, train_roots + 16),
                    seed=_stable_int_seed(seed, "cand", variant, "train"),
                    min_ply=min_ply,
                ),
                "val": _sample_balanced_candidate_roots(
                    pools["val_source_games"],
                    variant,
                    max_candidates=max(val_roots * 3, val_roots + 16),
                    seed=_stable_int_seed(seed, "cand", variant, "val"),
                    min_ply=min_ply,
                ),
                "test": _sample_balanced_candidate_roots(
                    pools["test_source_games"],
                    variant,
                    max_candidates=max(test_roots * 3, test_roots + 16),
                    seed=_stable_int_seed(seed, "cand", variant, "test"),
                    min_ply=min_ply,
                ),
            }

            dedup_res = deduplicate_and_filter_roots(
                cand_by_split,
                demo_games=demo_recs,
                max_similarity=max_similarity,
                min_ply=min_ply,
            )
            dedup_stats_by_var[variant] = dedup_res["dedup_stats"]

            for sp in ("train", "val", "test"):
                chosen_roots = _select_balanced_subset(
                    dedup_res["roots_by_split"][sp], target_n=target_counts[sp]
                )
                r_count = r_train if sp == "train" else r_eval
                for r_idx, root_item in enumerate(chosen_roots):
                    r_seed = _stable_int_seed(seed, "rollout", variant, sp, r_idx)
                    rollout_tasks.append(
                        (
                            variant,
                            sp,
                            r_idx,
                            root_item,
                            policy_id,
                            r_count,
                            horizon,
                            r_seed,
                        )
                    )

        # Parallel continuation rollouts across worker pool
        rolled_roots = pool.map(_worker_rollout_root, rollout_tasks)

    # Group labeled roots and write per-variant JSONLs
    positions_by_var_split: dict[str, dict[str, list[dict[str, Any]]]] = {
        v: {"train": [], "val": [], "test": []} for v in variant_list
    }
    for rec in rolled_roots:
        positions_by_var_split[rec["variant"]][rec["split"]].append(rec)

    manifest_variants: dict[str, Any] = {}
    for variant in variant_list:
        v_dir = output_dir / variant
        v_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(v_dir / "games_demo.jsonl", demos_by_var[variant])
        if alt_demo_policy:
            _write_jsonl(v_dir / "games_demo_alt_policy.jsonl", alt_by_var[variant])

        sp_summaries: dict[str, Any] = {}
        for sp in ("train", "val", "test"):
            recs = positions_by_var_split[variant][sp]
            _write_jsonl(v_dir / f"positions_{sp}.jsonl", recs)
            if recs:
                probs_mat = np.vstack([r["label_probs"] for r in recs])
                w_frac = float(np.mean([r["side_to_move"] == "w" for r in recs]))
                eq_frac = float(np.mean([r["equal_material"] for r in recs]))
                h_rate = float(np.mean([r["horizon_rate"] for r in recs]))
                mean_probs = [float(x) for x in np.mean(probs_mat, axis=0)]
            else:
                w_frac, eq_frac, h_rate, mean_probs = 0.0, 0.0, 0.0, [0.0, 0.0, 0.0]

            sp_summaries[sp] = {
                "n_positions": len(recs),
                "white_frac": w_frac,
                "black_frac": 1.0 - w_frac if recs else 0.0,
                "equal_material_frac": eq_frac,
                "mean_outcome_probs": mean_probs,
                "mean_horizon_rate": h_rate,
            }

        demo_h_rate = (
            float(np.mean([g["hit_horizon"] for g in demos_by_var[variant]]))
            if demos_by_var[variant]
            else 0.0
        )
        manifest_variants[variant] = {
            "n_demos": len(demos_by_var[variant]),
            "demo_horizon_rate": demo_h_rate,
            "dedup_stats": dedup_stats_by_var[variant],
            "splits": sp_summaries,
        }

    # Run integrity & leakage audit
    audit_report = audit_dataset_integrity(output_dir, variant_list)
    audit_path = output_dir / "dataset_audit.json"
    audit_path.write_text(json.dumps(audit_report, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - t_start
    manifest = {
        "policy_id": policy_id,
        "alt_demo_policy": alt_demo_policy,
        "horizon": int(horizon),
        "R_train": int(r_train),
        "R_eval": int(r_eval),
        "demo_games_target": int(demo_games),
        "source_games_target": int(source_games),
        "train_roots_target": int(train_roots),
        "val_roots_target": int(val_roots),
        "test_roots_target": int(test_roots),
        "max_similarity": float(max_similarity),
        "min_ply": int(min_ply),
        "workers": int(workers),
        "seed": int(seed),
        "elapsed_seconds": float(elapsed),
        "audit_passed": bool(audit_report["passed"]),
        "variants": manifest_variants,
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not audit_report["passed"]:
        raise RuntimeError(
            f"Dataset integrity audit failed with {audit_report['n_errors']} errors: "
            f"{audit_report['errors'][:5]}"
        )
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stage 1 dataset for chess_icl_v2.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dataset root directory.")
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS), help="Variant names to generate.")
    parser.add_argument("--policy-id", type=str, default="terminal_aware_v1", help="Main rollout policy.")
    parser.add_argument("--horizon", type=int, default=320, help="Max plies per game/rollout.")
    parser.add_argument("--demo-games", type=int, default=200, help="Demo games per variant.")
    parser.add_argument("--source-games", type=int, default=300, help="Root-source games per variant.")
    parser.add_argument("--train-roots", type=int, default=400, help="Train query roots per variant.")
    parser.add_argument("--val-roots", type=int, default=100, help="Val query roots per variant.")
    parser.add_argument("--test-roots", type=int, default=200, help="Test query roots per variant.")
    parser.add_argument("--R-train", type=int, default=16, dest="r_train", help="Rollouts per train root.")
    parser.add_argument("--R-eval", type=int, default=32, dest="r_eval", help="Rollouts per val/test root.")
    parser.add_argument("--max-similarity", type=float, default=0.90, help="Hamming near-dup threshold.")
    parser.add_argument("--min-ply", type=int, default=8, help="Minimum ply for root sampling.")
    parser.add_argument("--workers", type=int, default=4, help="Number of CPU worker processes.")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed.")
    parser.add_argument("--alt-demo-policy", type=str, default=None, help="Optional second demo policy for Exp 3.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    manifest = generate_dataset(
        output_dir=args.output_dir,
        variants=args.variants,
        policy_id=args.policy_id,
        horizon=args.horizon,
        demo_games=args.demo_games,
        source_games=args.source_games,
        train_roots=args.train_roots,
        val_roots=args.val_roots,
        test_roots=args.test_roots,
        r_train=args.r_train,
        r_eval=args.r_eval,
        max_similarity=args.max_similarity,
        min_ply=args.min_ply,
        workers=args.workers,
        seed=args.seed,
        alt_demo_policy=args.alt_demo_policy,
    )
    print(
        f"[generate_data_v2] Generated {len(manifest['variants'])} variants in "
        f"{manifest['elapsed_seconds']:.2f}s (audit_passed={manifest['audit_passed']})"
    )


if __name__ == "__main__":
    main()
