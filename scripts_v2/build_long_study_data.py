"""Long-study dataset production, censoring/budget, radius-2 audit, and immutable snapshot pipeline.

Supports four execution modes:
  - ``initial``: Generates the frozen 2026112501 dataset across standard + 13 SINGLE_VARIANTS
    with independent train/val/test demo pools, at most 2 adjacent roots per query family,
    2,560-ply administrative budget with same-stream extensions, strict unresolved censoring
    (never silently converted to draws), and full radius-2 + pilot separation auditing.
  - ``train_wave``: Generates additional training-only waves with independent seeds without
    modifying val/test positions or val/test demo pools.
  - ``freeze_snapshot``: Runs full audits, builds the outcome-independent early validation
    panel manifest (`val_panel_manifest.json`), copies/hardlinks files into an immutable
    `snapshots/<snapshot_id>/` directory with SHA-256 checksums and line counts.
  - ``r256_subset``: Re-estimates R=256 continuation targets on an outcome-independent
    sha256(family_id)-selected subset of retained test families.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping, Optional, Sequence

import chess
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_icl_v2.data.splits import (  # noqa: E402
    DEFAULT_PILOT_PATHS,
    RadiusTwoIndex,
    _add_games_to_index,
    audit_dataset_integrity,
    board_similarity_signature,
    build_pilot_index,
    compute_canonical_state_key,
    filter_roots_radius_two,
    radius_two_audit,
    sample_candidate_pairs_by_family,
)
from chess_icl_v2.rollouts.generator import (  # noqa: E402
    RULES_VERSION,
    play_full_game,
    rollout_continuations,
)
from chess_icl_v2.variants import SINGLE_VARIANTS, VariantBoard  # noqa: E402

DEFAULT_LONG_STUDY_VARIANTS: tuple[str, ...] = ("standard", *SINGLE_VARIANTS)
DEFAULT_MASTER_SEED = 2026112501


def _stable_int_seed(*parts: Any) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_lines(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                records.append(json.loads(s))
    return records


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _strip_game_for_demo_jsonl(game: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in game.items() if k != "states_at_plies"}


# ---------------------------------------------------------------------------
# Multiprocessing Workers (top-level for spawn pickling)
# ---------------------------------------------------------------------------


def _worker_play_long_game(
    task: tuple[str, str, int, int, int, int, int, bool, int, str, str],
) -> dict[str, Any]:
    (
        variant_name,
        policy_id,
        initial_budget,
        extend_step_plies,
        max_total_plies,
        max_extensions,
        seed,
        collect_states,
        min_sample_ply,
        family_id,
        rules_version,
    ) = task
    game = play_full_game(
        variant_name=variant_name,
        policy_id=policy_id,
        max_plies=initial_budget,
        seed=seed,
        collect_san=True,
        collect_states=collect_states,
        min_sample_ply=min_sample_ply,
        extend_until_terminal=True,
        extend_step_plies=extend_step_plies,
        max_total_plies=max_total_plies,
        max_extensions=max_extensions,
        unresolved_as_draw=False,
        rules_version=rules_version,
    )
    game["family_id"] = family_id
    game["source_game_id"] = family_id
    game["game_id"] = family_id
    return game


def _worker_rollout_long_root(
    task: tuple[
        str, str, dict[str, Any], str, int, int, int, int, int, int, str, Optional[str]
    ],
) -> dict[str, Any]:
    (
        variant_name,
        split_name,
        root_item,
        policy_id,
        r_count,
        initial_budget,
        extend_step_plies,
        max_total_plies,
        max_extensions,
        seed,
        rules_version,
        wave_id,
    ) = task
    sd = root_item["state_dict"]
    rollout_res = rollout_continuations(
        variant_name=variant_name,
        state_dict=sd,
        policy_id=policy_id,
        R=r_count,
        max_plies=initial_budget,
        seed=seed,
        unresolved_as_draw=False,
        allow_unresolved=True,
        extend_step_plies=extend_step_plies,
        max_total_plies=max_total_plies,
        max_extensions=max_extensions,
        rules_version=rules_version,
    )

    board = VariantBoard.from_state_dict(sd)
    mat_diff_w = int(board.evaluate_material(chess.WHITE))
    eq_mat = abs(mat_diff_w) <= 2

    family_id = str(root_item["family_id"])
    ply = int(root_item["ply"])
    label_probs = rollout_res["label_probs"]
    if label_probs is not None:
        label_names = ("win", "draw", "loss")
        argmax_lbl = label_names[int(np.argmax(label_probs))]
    else:
        argmax_lbl = "unresolved"

    rec: dict[str, Any] = {
        "variant": variant_name,
        "split": split_name,
        "rules_version": rules_version,
        "root_id": f"{variant_name}_{split_name}_{family_id}_ply{ply}",
        "family_id": family_id,
        "source_game_id": family_id,
        "game_id": family_id,
        "ply": ply,
        "pair_plies": list(root_item.get("pair_plies", [ply])),
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
        "target_bounds": rollout_res["target_bounds"],
        "counts": rollout_res["counts"],
        "n_unresolved": int(rollout_res["n_unresolved"]),
        "label": argmax_lbl,
        "R": int(r_count),
        "policy": policy_id,
        "initial_budget": int(initial_budget),
        "budget_used": int(rollout_res["budget_used"]),
        "extensions_used": int(rollout_res["extensions_used"]),
        "horizon_rate": float(rollout_res["horizon_rate"]),
        "decisive_rate": float(rollout_res["decisive_rate"]),
        "mean_plies": float(rollout_res["mean_plies"]),
        "total_plies": int(rollout_res["total_plies"]),
    }
    if wave_id is not None:
        rec["wave_id"] = wave_id
    return rec


def _run_pool_map(pool: Optional[Any], fn: Any, tasks: Sequence[Any]) -> list[Any]:
    if not tasks:
        return []
    if pool is None:
        return [fn(t) for t in tasks]
    return pool.map(fn, tasks)


def _generate_terminal_demos_for_pool(
    pool: Optional[Any],
    variant_name: str,
    pool_name: str,
    target_count: int,
    policy_id: str,
    initial_budget: int,
    extend_step_plies: int,
    max_total_plies: int,
    max_extensions: int,
    seed: int,
    rules_version: str,
    family_prefix: str,
) -> tuple[list[dict[str, Any]], int, int]:
    """Generate `target_count` terminal demonstration games (never admitting non-terminal)."""
    admitted: list[dict[str, Any]] = []
    total_played = 0
    total_plies = 0
    attempt_offset = 0
    max_attempts = max(target_count * 4, target_count + 16)

    while len(admitted) < target_count and attempt_offset < max_attempts:
        needed = target_count - len(admitted)
        batch_size = needed if attempt_offset == 0 else min(needed * 2, max_attempts - attempt_offset)
        tasks = []
        for k in range(batch_size):
            slot = attempt_offset + k
            fid = f"{family_prefix}_{slot}"
            g_seed = _stable_int_seed(seed, "demo_game", variant_name, pool_name, family_prefix, slot)
            tasks.append(
                (
                    variant_name,
                    policy_id,
                    initial_budget,
                    extend_step_plies,
                    max_total_plies,
                    max_extensions,
                    g_seed,
                    False,
                    8,
                    fid,
                    rules_version,
                )
            )
        attempt_offset += batch_size
        games = _run_pool_map(pool, _worker_play_long_game, tasks)
        total_played += len(games)
        for g in games:
            total_plies += int(g.get("move_count", 0))
            if bool(g.get("is_terminal")):
                admitted.append(_strip_game_for_demo_jsonl(g))
                if len(admitted) >= target_count:
                    break

    return admitted, total_played, total_plies


def _split_summary_stats(
    resolved_recs: Sequence[Mapping[str, Any]],
    all_recs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    n_all = len(all_recs)
    n_res = len(resolved_recs)
    n_cens = n_all - n_res
    if resolved_recs:
        probs_mat = np.vstack([r["label_probs"] for r in resolved_recs])
        w_frac = float(np.mean([r["side_to_move"] == "w" for r in resolved_recs]))
        eq_frac = float(np.mean([r["equal_material"] for r in resolved_recs]))
        mean_probs = [float(x) for x in np.mean(probs_mat, axis=0)]
    else:
        w_frac, eq_frac, mean_probs = 0.0, 0.0, [0.0, 0.0, 0.0]

    families_retained = len({str(r["family_id"]) for r in resolved_recs})
    return {
        "n_attempted": n_all,
        "n_positions": n_res,
        "n_censored": n_cens,
        "censoring_rate": float(n_cens / n_all) if n_all > 0 else 0.0,
        "n_families_retained": families_retained,
        "white_frac": w_frac,
        "black_frac": (1.0 - w_frac) if resolved_recs else 0.0,
        "equal_material_frac": eq_frac,
        "mean_outcome_probs": mean_probs,
    }


def _resolve_pilot_paths(pilot_paths: Optional[Sequence[Path]]) -> list[Path]:
    if pilot_paths is not None:
        return [Path(p) for p in pilot_paths]
    return [REPO_ROOT / p if not p.is_absolute() else p for p in DEFAULT_PILOT_PATHS]


# ---------------------------------------------------------------------------
# Mode: initial
# ---------------------------------------------------------------------------


def run_initial_generation(
    output_dir: Path,
    variants: Sequence[str] = DEFAULT_LONG_STUDY_VARIANTS,
    policy_id: str = "terminal_aware_v1",
    r_train: int = 32,
    r_eval: int = 32,
    initial_budget: int = 2560,
    extend_step_plies: int = 2560,
    max_total_plies: int = 10240,
    max_extensions: int = 3,
    demo_train_families: int = 128,
    demo_val_families: int = 64,
    demo_test_families: int = 64,
    train_families: int = 256,
    val_families: int = 64,
    test_families: int = 64,
    min_ply: int = 8,
    workers: int = 4,
    seed: int = DEFAULT_MASTER_SEED,
    val_first: bool = False,
    pilot_paths: Optional[Sequence[Path]] = None,
    check_pilot: bool = True,
    rules_version: str = RULES_VERSION,
    rollout_initial_budget: Optional[int] = None,
    rollout_max_total_plies: Optional[int] = None,
    rollout_max_extensions: Optional[int] = None,
) -> dict[str, Any]:
    """Generate the initial long-study dataset across variants with full audits."""
    t_start = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variant_list = list(variants)

    resolved_pilot_paths = _resolve_pilot_paths(pilot_paths)
    pilot_index: Optional[RadiusTwoIndex] = None
    pilot_stats: Optional[dict[str, Any]] = None
    if check_pilot:
        pilot_index, pilot_stats = build_pilot_index(pilot_paths=resolved_pilot_paths)

    split_order = ("val", "test", "train") if val_first else ("test", "val", "train")
    demo_quotas = {
        "train": int(demo_train_families),
        "val": int(demo_val_families),
        "test": int(demo_test_families),
    }
    source_quotas = {
        "train": int(train_families),
        "val": int(val_families),
        "test": int(test_families),
    }

    total_games_played = 0
    total_game_plies = 0
    total_continuations = 0
    total_rollout_plies = 0

    ctx = mp.get_context("spawn")
    pool_cm = ctx.Pool(processes=int(workers)) if int(workers) > 1 else None

    manifest_variants: dict[str, Any] = {}
    try:
        for variant in variant_list:
            v_dir = output_dir / variant
            v_dir.mkdir(parents=True, exist_ok=True)

            # 1. Generate independent terminal demo pools per split
            demos_by_split: dict[str, list[dict[str, Any]]] = {}
            for sp in split_order:
                demos_sp, n_played, n_plies = _generate_terminal_demos_for_pool(
                    pool=pool_cm,
                    variant_name=variant,
                    pool_name=sp,
                    target_count=demo_quotas[sp],
                    policy_id=policy_id,
                    initial_budget=initial_budget,
                    extend_step_plies=extend_step_plies,
                    max_total_plies=max_total_plies,
                    max_extensions=max_extensions,
                    seed=seed,
                    rules_version=rules_version,
                    family_prefix=f"{variant}_demo_{sp}",
                )
                demos_by_split[sp] = demos_sp
                total_games_played += n_played
                total_game_plies += n_plies
                _write_jsonl(v_dir / f"games_demo_{sp}.jsonl", demos_sp)

            # 2. Generate query-source families per split
            source_tasks: list[Any] = []
            for sp in split_order:
                for i in range(source_quotas[sp]):
                    fid = f"{variant}_src_{sp}_{i}"
                    g_seed = _stable_int_seed(seed, "source_game", variant, sp, i)
                    source_tasks.append(
                        (
                            variant,
                            policy_id,
                            initial_budget,
                            extend_step_plies,
                            max_total_plies,
                            max_extensions,
                            g_seed,
                            True,
                            min_ply,
                            fid,
                            rules_version,
                        )
                    )

            played_sources = _run_pool_map(pool_cm, _worker_play_long_game, source_tasks)
            total_games_played += len(played_sources)
            total_game_plies += sum(int(g.get("move_count", 0)) for g in played_sources)

            sources_by_split: dict[str, list[dict[str, Any]]] = {sp: [] for sp in ("train", "val", "test")}
            for g in played_sources:
                fid = str(g["family_id"])
                for sp in ("train", "val", "test"):
                    if f"_src_{sp}_" in fid:
                        sources_by_split[sp].append(g)
                        break

            # 3. Sample candidate adjacent root pairs per family and filter with radius-2 + pilot separation
            cand_pairs_by_split = {
                sp: sample_candidate_pairs_by_family(
                    sources_by_split[sp],
                    variant_name=variant,
                    seed=_stable_int_seed(seed, "cand_pairs", variant, sp),
                    min_ply=min_ply,
                )
                for sp in ("test", "val", "train")
            }

            filtered = filter_roots_radius_two(
                candidate_pairs_by_split=cand_pairs_by_split,
                demo_pools=demos_by_split,
                pilot_index=pilot_index,
                pilot_paths=resolved_pilot_paths,
                max_hamming=2,
                min_ply=min_ply,
                check_pilot=check_pilot,
            )
            roots_by_split = filtered["roots_by_split"]

            # 4. Rollout continuations (val-first ordering if requested)
            r_init_b = initial_budget if rollout_initial_budget is None else int(rollout_initial_budget)
            r_max_tot = max_total_plies if rollout_max_total_plies is None else int(rollout_max_total_plies)
            r_max_ext = max_extensions if rollout_max_extensions is None else int(rollout_max_extensions)
            rollout_tasks: list[Any] = []
            for sp in split_order:
                r_count = r_train if sp == "train" else r_eval
                for root_item in roots_by_split[sp]:
                    r_seed = _stable_int_seed(
                        seed, "rollout", variant, sp, root_item["family_id"], root_item["ply"]
                    )
                    rollout_tasks.append(
                        (
                            variant,
                            sp,
                            root_item,
                            policy_id,
                            r_count,
                            r_init_b,
                            extend_step_plies,
                            r_max_tot,
                            r_max_ext,
                            r_seed,
                            rules_version,
                            None,
                        )
                    )

            rolled = _run_pool_map(pool_cm, _worker_rollout_long_root, rollout_tasks)
            by_split_all: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
            for rec in rolled:
                by_split_all[rec["split"]].append(rec)
                total_continuations += int(rec["R"])
                total_rollout_plies += int(rec["total_plies"])

            sp_summaries: dict[str, Any] = {}
            for sp in ("train", "val", "test"):
                all_recs = by_split_all[sp]
                resolved_recs = [
                    r for r in all_recs if int(r["n_unresolved"]) == 0 and r["label_probs"] is not None
                ]
                _write_jsonl(v_dir / f"positions_{sp}.jsonl", resolved_recs)
                if len(resolved_recs) < len(all_recs):
                    _write_jsonl(v_dir / f"positions_{sp}_all_with_censoring.jsonl", all_recs)
                sp_summaries[sp] = _split_summary_stats(resolved_recs, all_recs)

            if val_first:
                vs = sp_summaries["val"]
                print(
                    f"[val-first] {variant}: val_retained={vs['n_positions']} "
                    f"({vs['n_families_retained']} families), censored={vs['n_censored']}"
                )

            manifest_variants[variant] = {
                "demo_counts": {sp: len(demos_by_split[sp]) for sp in ("train", "val", "test")},
                "filter_stats": filtered["filter_stats"],
                "splits": sp_summaries,
            }
    finally:
        if pool_cm is not None:
            pool_cm.close()
            pool_cm.join()

    # 5. Full integrity, family quota, prompt leakage, and radius-2 + pilot audit
    audit_report = audit_dataset_integrity(
        data_dir=output_dir,
        variants=variant_list,
        independent_demos=True,
        run_radius_two_audit=True,
        pilot_paths=resolved_pilot_paths,
        pilot_index=pilot_index,
        check_pilot=check_pilot,
        max_roots_per_family=2,
        require_terminal_demos=True,
    )
    (output_dir / "dataset_audit.json").write_text(json.dumps(audit_report, indent=2), encoding="utf-8")

    elapsed = max(1e-9, time.perf_counter() - t_start)
    total_plies = total_game_plies + total_rollout_plies
    throughput = {
        "games_generated": total_games_played,
        "games_per_sec": total_games_played / elapsed,
        "continuations_run": total_continuations,
        "continuations_per_sec": total_continuations / elapsed,
        "total_plies": total_plies,
        "plies_per_sec": total_plies / elapsed,
        "elapsed_sec": elapsed,
    }
    manifest = {
        "mode": "initial",
        "rules_version": rules_version,
        "policy_id": policy_id,
        "master_seed": int(seed),
        "R_train": int(r_train),
        "R_eval": int(r_eval),
        "initial_budget": int(initial_budget),
        "extend_step_plies": int(extend_step_plies),
        "max_total_plies": int(max_total_plies),
        "max_extensions": int(max_extensions),
        "demo_quotas": demo_quotas,
        "source_quotas": source_quotas,
        "min_ply": int(min_ply),
        "workers": int(workers),
        "val_first": bool(val_first),
        "pilot_stats": pilot_stats,
        "throughput": throughput,
        "audit_passed": bool(audit_report["passed"]),
        "variants": manifest_variants,
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"[build_long_study_data:initial] variants={len(variant_list)} "
        f"games={total_games_played} ({throughput['games_per_sec']:.2f} games/s) "
        f"conts={total_continuations} ({throughput['continuations_per_sec']:.2f} cont/s) "
        f"elapsed={elapsed:.2f}s audit_passed={audit_report['passed']}"
    )

    if not audit_report["passed"]:
        raise RuntimeError(
            f"Long-study integrity audit failed ({audit_report['n_errors']} errors): "
            f"{audit_report['errors'][:5]}"
        )
    return manifest


# ---------------------------------------------------------------------------
# Mode: train_wave (training-only wave; val/test untouched)
# ---------------------------------------------------------------------------


def run_train_wave(
    output_dir: Path,
    wave_id: int,
    variants: Optional[Sequence[str]] = None,
    policy_id: str = "terminal_aware_v1",
    r_train: int = 32,
    initial_budget: int = 2560,
    extend_step_plies: int = 2560,
    max_total_plies: int = 10240,
    max_extensions: int = 3,
    train_families: int = 128,
    train_demo_games: int = 0,
    min_ply: int = 8,
    workers: int = 4,
    seed: int = DEFAULT_MASTER_SEED,
    pilot_paths: Optional[Sequence[Path]] = None,
    check_pilot: bool = True,
    rules_version: str = RULES_VERSION,
    rollout_initial_budget: Optional[int] = None,
    rollout_max_total_plies: Optional[int] = None,
    rollout_max_extensions: Optional[int] = None,
) -> dict[str, Any]:
    """Generate an additional training-only wave without touching val or test splits/demos."""
    t_start = time.perf_counter()
    output_dir = Path(output_dir)
    if variants is None:
        variant_list = sorted(
            p.name for p in output_dir.iterdir()
            if p.is_dir() and (p / "positions_train.jsonl").exists()
        )
    else:
        variant_list = list(variants)

    resolved_pilot_paths = _resolve_pilot_paths(pilot_paths)
    pilot_index: Optional[RadiusTwoIndex] = None
    if check_pilot:
        pilot_index, _ = build_pilot_index(pilot_paths=resolved_pilot_paths)

    total_games_played = 0
    total_game_plies = 0
    total_continuations = 0
    total_rollout_plies = 0

    ctx = mp.get_context("spawn")
    pool_cm = ctx.Pool(processes=int(workers)) if int(workers) > 1 else None
    wave_tag = f"w{wave_id}"
    wave_variants: dict[str, Any] = {}

    try:
        for variant in variant_list:
            v_dir = output_dir / variant
            demo_train = _load_jsonl(v_dir / "games_demo_train.jsonl")
            demo_val = _load_jsonl(v_dir / "games_demo_val.jsonl")
            demo_test = _load_jsonl(v_dir / "games_demo_test.jsonl")

            existing_train = _load_jsonl(v_dir / "positions_train.jsonl")
            existing_val = _load_jsonl(v_dir / "positions_val.jsonl")
            existing_test = _load_jsonl(v_dir / "positions_test.jsonl")

            # Optional training-only demo expansion
            added_demos: list[dict[str, Any]] = []
            if train_demo_games > 0:
                added_demos, n_dp, n_dpl = _generate_terminal_demos_for_pool(
                    pool=pool_cm,
                    variant_name=variant,
                    pool_name=f"train_{wave_tag}",
                    target_count=int(train_demo_games),
                    policy_id=policy_id,
                    initial_budget=initial_budget,
                    extend_step_plies=extend_step_plies,
                    max_total_plies=max_total_plies,
                    max_extensions=max_extensions,
                    seed=_stable_int_seed(seed, "train_wave_demo", wave_id),
                    rules_version=rules_version,
                    family_prefix=f"{variant}_demo_train_{wave_tag}",
                )
                total_games_played += n_dp
                total_game_plies += n_dpl
                # Filter new training demos against all existing query roots (train/val/test)
                # so newly added demo boards cannot fall within Hamming <= 2 of prior roots.
                existing_roots = existing_train + existing_val + existing_test
                if existing_roots and added_demos:
                    existing_sig_pairs = [
                        (
                            *board_similarity_signature(r),
                            compute_canonical_state_key(r),
                        )
                        for r in existing_roots
                    ]
                    clean_added: list[dict[str, Any]] = []
                    for g in added_demos:
                        g_idx = RadiusTwoIndex(deduplicate_exact_boards=True)
                        _add_games_to_index(g_idx, [g], source_tag=str(g.get("game_id", "")))
                        if not any(
                            g_idx.find_collision(arr, t_bit, canonical_key=ckey, max_hamming=2) is not None
                            for arr, t_bit, ckey in existing_sig_pairs
                        ):
                            clean_added.append(g)
                    added_demos = clean_added
                demo_train.extend(added_demos)
                _write_jsonl(v_dir / "games_demo_train.jsonl", demo_train)

            # Generate new training source families
            src_tasks = []
            for i in range(int(train_families)):
                fid = f"{variant}_src_train_{wave_tag}_{i}"
                g_seed = _stable_int_seed(seed, "train_wave_src", wave_id, variant, i)
                src_tasks.append(
                    (
                        variant,
                        policy_id,
                        initial_budget,
                        extend_step_plies,
                        max_total_plies,
                        max_extensions,
                        g_seed,
                        True,
                        min_ply,
                        fid,
                        rules_version,
                    )
                )
            played_src = _run_pool_map(pool_cm, _worker_play_long_game, src_tasks)
            total_games_played += len(played_src)
            total_game_plies += sum(int(g.get("move_count", 0)) for g in played_src)

            cand_pairs = sample_candidate_pairs_by_family(
                played_src,
                variant_name=variant,
                seed=_stable_int_seed(seed, "train_wave_cand", wave_id, variant),
                min_ply=min_ply,
            )

            filtered = filter_roots_radius_two(
                candidate_pairs_by_split={"train": cand_pairs},
                demo_pools={"train": demo_train, "val": demo_val, "test": demo_test},
                pilot_index=pilot_index,
                pilot_paths=resolved_pilot_paths,
                existing_roots_by_split={
                    "test": existing_test,
                    "val": existing_val,
                    "train": existing_train,
                },
                max_hamming=2,
                min_ply=min_ply,
                check_pilot=check_pilot,
            )

            new_roots = filtered["roots_by_split"]["train"]
            rollout_tasks = []
            for root_item in new_roots:
                r_seed = _stable_int_seed(
                    seed, "train_wave_rollout", wave_id, variant, root_item["family_id"], root_item["ply"]
                )
                rollout_tasks.append(
                    (
                        variant,
                        "train",
                        root_item,
                        policy_id,
                        r_train,
                        initial_budget,
                        extend_step_plies,
                        max_total_plies,
                        max_extensions,
                        r_seed,
                        rules_version,
                        wave_tag,
                    )
                )

            rolled = _run_pool_map(pool_cm, _worker_rollout_long_root, rollout_tasks)
            for rec in rolled:
                total_continuations += int(rec["R"])
                total_rollout_plies += int(rec["total_plies"])

            resolved_new = [
                r for r in rolled if int(r["n_unresolved"]) == 0 and r["label_probs"] is not None
            ]
            _write_jsonl(v_dir / f"positions_train_wave_{wave_id}.jsonl", resolved_new)
            if len(resolved_new) < len(rolled):
                _write_jsonl(v_dir / f"positions_train_wave_{wave_id}_all_with_censoring.jsonl", rolled)

            merged_train = existing_train + resolved_new
            _write_jsonl(v_dir / "positions_train.jsonl", merged_train)

            if (v_dir / "positions_train_all_with_censoring.jsonl").exists() or len(resolved_new) < len(rolled):
                prev_all = _load_jsonl(v_dir / "positions_train_all_with_censoring.jsonl") or existing_train
                _write_jsonl(v_dir / "positions_train_all_with_censoring.jsonl", prev_all + rolled)

            wave_variants[variant] = {
                "added_train_demos": len(added_demos),
                "added_train_positions": len(resolved_new),
                "added_train_censored": len(rolled) - len(resolved_new),
                "total_train_positions": len(merged_train),
                "filter_stats": filtered["filter_stats"],
            }
    finally:
        if pool_cm is not None:
            pool_cm.close()
            pool_cm.join()

    audit_report = audit_dataset_integrity(
        data_dir=output_dir,
        variants=variant_list,
        independent_demos=True,
        run_radius_two_audit=True,
        pilot_paths=resolved_pilot_paths,
        pilot_index=pilot_index,
        check_pilot=check_pilot,
        max_roots_per_family=2,
        require_terminal_demos=True,
    )
    elapsed = max(1e-9, time.perf_counter() - t_start)
    wave_manifest = {
        "mode": "train_wave",
        "wave_id": int(wave_id),
        "rules_version": rules_version,
        "audit_passed": bool(audit_report["passed"]),
        "throughput": {
            "games_generated": total_games_played,
            "games_per_sec": total_games_played / elapsed,
            "continuations_run": total_continuations,
            "continuations_per_sec": total_continuations / elapsed,
            "total_plies": total_game_plies + total_rollout_plies,
            "elapsed_sec": elapsed,
        },
        "variants": wave_variants,
    }
    (output_dir / f"train_wave_{wave_id}_manifest.json").write_text(
        json.dumps(wave_manifest, indent=2), encoding="utf-8"
    )
    print(
        f"[build_long_study_data:train_wave] wave={wave_id} variants={len(variant_list)} "
        f"conts={total_continuations} ({total_continuations / elapsed:.2f} cont/s) "
        f"elapsed={elapsed:.2f}s audit_passed={audit_report['passed']}"
    )
    if not audit_report["passed"]:
        raise RuntimeError(
            f"Train-wave audit failed ({audit_report['n_errors']} errors): {audit_report['errors'][:5]}"
        )
    return wave_manifest


# ---------------------------------------------------------------------------
# Mode: freeze_snapshot & verify_snapshot
# ---------------------------------------------------------------------------


def build_val_panel_manifest(
    data_dir: Path,
    variants: Sequence[str],
    panel_families: int = 16,
) -> dict[str, Any]:
    """Select the fixed early validation panel using the first `panel_families` sorted by sha256(family_id)."""
    per_variant: dict[str, Any] = {}
    for variant in variants:
        val_recs = _load_jsonl(Path(data_dir) / variant / "positions_val.jsonl")
        by_fid: dict[str, list[dict[str, Any]]] = {}
        for idx, rec in enumerate(val_recs):
            fid = str(rec.get("family_id") or rec.get("source_game_id") or f"{variant}_val_{idx}")
            by_fid.setdefault(fid, []).append(rec)

        hashed_fids = sorted(
            ((_sha256_text(fid), fid) for fid in by_fid),
            key=lambda pair: (pair[0], pair[1]),
        )
        chosen = hashed_fids[: max(0, int(panel_families))]

        selected_families: list[dict[str, Any]] = []
        selected_root_ids: list[str] = []
        for h_hex, fid in chosen:
            siblings = by_fid[fid]
            sib_ids = [str(s["root_id"]) for s in siblings]
            selected_root_ids.extend(sib_ids)
            selected_families.append(
                {
                    "family_id": fid,
                    "family_sha256": h_hex,
                    "root_ids": sib_ids,
                    "plies": [int(s["ply"]) for s in siblings if "ply" in s],
                }
            )

        per_variant[variant] = {
            "n_retained_val_families": len(by_fid),
            "n_panel_families": len(selected_families),
            "n_panel_roots": len(selected_root_ids),
            "selected_families": selected_families,
            "selected_root_ids": selected_root_ids,
        }

    return {
        "selection_rule": "first_k_retained_val_families_sorted_by_sha256(family_id)_with_all_siblings",
        "panel_families_requested": int(panel_families),
        "root_ids_by_variant": {
            variant: list(info["selected_root_ids"]) for variant, info in per_variant.items()
        },
        "variants": per_variant,
    }


def _copy_or_hardlink(src: Path, dst: Path, prefer_hardlink: bool = True) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if prefer_hardlink:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def freeze_snapshot(
    source_dir: Path,
    snapshot_dir: Path,
    snapshot_id: str,
    variants: Optional[Sequence[str]] = None,
    panel_families: int = 16,
    pilot_paths: Optional[Sequence[Path]] = None,
    check_pilot: bool = True,
    prefer_hardlink: bool = False,
    rules_version: str = RULES_VERSION,
) -> dict[str, Any]:
    """Audit `source_dir`, select early val panel, copy/hardlink to immutable `snapshot_dir`, and checksum."""
    source_dir = Path(source_dir)
    snapshot_dir = Path(snapshot_dir)
    if snapshot_dir.exists() and any(snapshot_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty snapshot directory: {snapshot_dir}")

    if variants is None:
        variant_list = sorted(
            p.name for p in source_dir.iterdir()
            if p.is_dir() and (p / "positions_val.jsonl").exists()
        )
    else:
        variant_list = list(variants)

    resolved_pilot_paths = _resolve_pilot_paths(pilot_paths)
    audit_report = audit_dataset_integrity(
        data_dir=source_dir,
        variants=variant_list,
        independent_demos=True,
        run_radius_two_audit=True,
        pilot_paths=resolved_pilot_paths,
        max_roots_per_family=2,
        require_terminal_demos=True,
    ) if check_pilot else audit_dataset_integrity(
        data_dir=source_dir,
        variants=variant_list,
        independent_demos=True,
        run_radius_two_audit=True,
        pilot_paths=[],
        max_roots_per_family=2,
        require_terminal_demos=True,
    )
    if not audit_report["passed"]:
        raise RuntimeError(
            f"Cannot freeze snapshot {snapshot_id!r}: audit failed with {audit_report['n_errors']} errors: "
            f"{audit_report['errors'][:5]}"
        )

    snapshot_dir.mkdir(parents=True, exist_ok=True)

    val_panel = build_val_panel_manifest(source_dir, variant_list, panel_families=panel_families)
    val_panel_path = snapshot_dir / "val_panel_manifest.json"
    val_panel_path.write_text(json.dumps(val_panel, indent=2), encoding="utf-8")

    audit_out_path = snapshot_dir / "dataset_audit.json"
    audit_out_path.write_text(json.dumps(audit_report, indent=2), encoding="utf-8")

    # Copy/hardlink dataset manifest if present
    src_manifest = source_dir / "dataset_manifest.json"
    if src_manifest.exists():
        _copy_or_hardlink(src_manifest, snapshot_dir / "dataset_manifest.json", prefer_hardlink=prefer_hardlink)

    # Copy/hardlink all variant JSONLs
    for variant in variant_list:
        src_v = source_dir / variant
        dst_v = snapshot_dir / variant
        dst_v.mkdir(parents=True, exist_ok=True)
        for src_file in sorted(src_v.glob("*.jsonl")):
            _copy_or_hardlink(src_file, dst_v / src_file.name, prefer_hardlink=prefer_hardlink)

    # Compute SHA-256 checksums & line counts for every file in snapshot_dir, and mark read-only
    file_entries: dict[str, dict[str, Any]] = {}
    for fpath in sorted(snapshot_dir.rglob("*")):
        if not fpath.is_file():
            continue
        rel = fpath.relative_to(snapshot_dir).as_posix()
        if rel == "snapshot_manifest.json":
            continue
        file_entries[rel] = {
            "sha256": _sha256_file(fpath),
            "line_count": _count_lines(fpath),
            "size_bytes": int(fpath.stat().st_size),
        }
        os.chmod(fpath, 0o444)

    snapshot_manifest = {
        "snapshot_id": snapshot_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rules_version": rules_version,
        "source_dir": str(source_dir),
        "variants": variant_list,
        "exposed_variants": variant_list,
        "audit_passed": True,
        "val_panel_manifest": "val_panel_manifest.json",
        "n_files": len(file_entries),
        "files": file_entries,
    }
    manifest_path = snapshot_dir / "snapshot_manifest.json"
    manifest_path.write_text(json.dumps(snapshot_manifest, indent=2), encoding="utf-8")
    os.chmod(manifest_path, 0o444)

    print(
        f"[build_long_study_data:freeze_snapshot] Froze snapshot={snapshot_id!r} "
        f"at {snapshot_dir} ({len(file_entries)} checksummed read-only files)"
    )
    return snapshot_manifest


def verify_snapshot(snapshot_dir: Path) -> dict[str, Any]:
    """Verify every file listed in `snapshot_manifest.json` matches its SHA-256 and line count."""
    snapshot_dir = Path(snapshot_dir)
    manifest_path = snapshot_dir / "snapshot_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing snapshot_manifest.json in {snapshot_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    mismatches: list[str] = []
    files = manifest.get("files", {})
    for rel, expected in files.items():
        fpath = snapshot_dir / rel
        if not fpath.exists():
            mismatches.append(f"Missing file: {rel}")
            continue
        actual_sha = _sha256_file(fpath)
        actual_lines = _count_lines(fpath)
        if actual_sha != expected["sha256"]:
            mismatches.append(f"SHA-256 mismatch for {rel}: {actual_sha} != {expected['sha256']}")
        if actual_lines != int(expected["line_count"]):
            mismatches.append(f"Line count mismatch for {rel}: {actual_lines} != {expected['line_count']}")

    return {
        "passed": len(mismatches) == 0,
        "n_checked": len(files),
        "mismatches": mismatches,
    }


# ---------------------------------------------------------------------------
# Mode: r256_subset (outcome-independent test family-hash subset)
# ---------------------------------------------------------------------------


def run_r256_subset(
    source_dir: Path,
    out_dir: Path,
    variants: Optional[Sequence[str]] = None,
    subset_families: int = 16,
    r_eval: int = 256,
    policy_id: str = "terminal_aware_v1",
    initial_budget: int = 2560,
    extend_step_plies: int = 2560,
    max_total_plies: int = 10240,
    max_extensions: int = 3,
    workers: int = 4,
    seed: int = DEFAULT_MASTER_SEED,
    rules_version: str = RULES_VERSION,
) -> dict[str, Any]:
    """Re-estimate R=256 targets on a preselected sha256(family_id) subset of retained test families."""
    t_start = time.perf_counter()
    source_dir = Path(source_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if variants is None:
        variant_list = sorted(
            p.name for p in source_dir.iterdir()
            if p.is_dir() and (p / "positions_test.jsonl").exists()
        )
    else:
        variant_list = list(variants)

    ctx = mp.get_context("spawn")
    pool_cm = ctx.Pool(processes=int(workers)) if int(workers) > 1 else None
    per_variant: dict[str, Any] = {}
    total_continuations = 0
    total_rollout_plies = 0

    try:
        for variant in variant_list:
            test_recs = _load_jsonl(source_dir / variant / "positions_test.jsonl")
            by_fid: dict[str, list[dict[str, Any]]] = {}
            for idx, rec in enumerate(test_recs):
                fid = str(rec.get("family_id") or rec.get("source_game_id") or f"{variant}_test_{idx}")
                by_fid.setdefault(fid, []).append(rec)

            hashed = sorted(((_sha256_text(fid), fid) for fid in by_fid), key=lambda p: (p[0], p[1]))
            chosen = hashed[: max(0, int(subset_families))]

            tasks = []
            for _, fid in chosen:
                for rec in by_fid[fid]:
                    root_item = {
                        "family_id": fid,
                        "ply": int(rec["ply"]),
                        "pair_plies": rec.get("pair_plies", [rec["ply"]]),
                        "state_dict": rec["state_dict"],
                        "canonical_state_key": rec["canonical_state_key"],
                    }
                    r_seed = _stable_int_seed(seed, "r256_subset", variant, fid, rec["ply"])
                    tasks.append(
                        (
                            variant,
                            "test",
                            root_item,
                            policy_id,
                            int(r_eval),
                            initial_budget,
                            extend_step_plies,
                            max_total_plies,
                            max_extensions,
                            r_seed,
                            rules_version,
                            None,
                        )
                    )

            rolled = _run_pool_map(pool_cm, _worker_rollout_long_root, tasks)
            for r in rolled:
                total_continuations += int(r["R"])
                total_rollout_plies += int(r["total_plies"])

            resolved = [r for r in rolled if int(r["n_unresolved"]) == 0 and r["label_probs"] is not None]
            v_out = out_dir / variant
            v_out.mkdir(parents=True, exist_ok=True)
            _write_jsonl(v_out / "positions_test_r256_subset.jsonl", resolved)
            if len(resolved) < len(rolled):
                _write_jsonl(v_out / "positions_test_r256_subset_all_with_censoring.jsonl", rolled)

            per_variant[variant] = {
                "selected_families": [fid for _, fid in chosen],
                "n_attempted": len(rolled),
                "n_resolved": len(resolved),
                "n_censored": len(rolled) - len(resolved),
            }
    finally:
        if pool_cm is not None:
            pool_cm.close()
            pool_cm.join()

    elapsed = max(1e-9, time.perf_counter() - t_start)
    manifest = {
        "mode": "r256_subset",
        "selection_rule": "first_k_retained_test_families_sorted_by_sha256(family_id)_with_all_siblings",
        "R": int(r_eval),
        "subset_families_requested": int(subset_families),
        "rules_version": rules_version,
        "throughput": {
            "continuations_run": total_continuations,
            "continuations_per_sec": total_continuations / elapsed,
            "total_plies": total_rollout_plies,
            "elapsed_sec": elapsed,
        },
        "variants": per_variant,
    }
    (out_dir / "r256_subset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Long-study data production and snapshot tool.")
    parser.add_argument(
        "--mode",
        choices=("initial", "train_wave", "freeze_snapshot", "r256_subset"),
        default="initial",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Dataset directory.")
    parser.add_argument("--source-dir", type=Path, default=None, help="Source dir for snapshot/r256.")
    parser.add_argument("--snapshot-dir", type=Path, default=None, help="Destination snapshot dir.")
    parser.add_argument("--snapshot-id", type=str, default=str(DEFAULT_MASTER_SEED))
    parser.add_argument("--variants", nargs="+", default=None, help="Variant subset (default: 14 settings).")
    parser.add_argument("--policy-id", type=str, default="terminal_aware_v1")
    parser.add_argument("--seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument("--R-train", type=int, default=32, dest="r_train")
    parser.add_argument("--R-eval", type=int, default=32, dest="r_eval")
    parser.add_argument("--initial-budget", type=int, default=2560)
    parser.add_argument("--extend-step-plies", type=int, default=2560)
    parser.add_argument("--max-total-plies", type=int, default=10240)
    parser.add_argument("--max-extensions", type=int, default=3)
    parser.add_argument("--demo-train-families", type=int, default=128)
    parser.add_argument("--demo-val-families", type=int, default=64)
    parser.add_argument("--demo-test-families", type=int, default=64)
    parser.add_argument("--train-families", type=int, default=256)
    parser.add_argument("--val-families", type=int, default=64)
    parser.add_argument("--test-families", type=int, default=64)
    parser.add_argument("--min-ply", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--val-first", action="store_true")
    parser.add_argument("--wave-id", type=int, default=1)
    parser.add_argument("--train-demo-games", type=int, default=0)
    parser.add_argument("--panel-families", type=int, default=16)
    parser.add_argument("--subset-families", type=int, default=16)
    parser.add_argument("--r256-eval", type=int, default=256)
    parser.add_argument("--pilot-paths", nargs="*", type=Path, default=None)
    parser.add_argument("--no-pilot-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    variants = list(args.variants) if args.variants is not None else list(DEFAULT_LONG_STUDY_VARIANTS)

    if args.mode == "initial":
        run_initial_generation(
            output_dir=args.output_dir,
            variants=variants,
            policy_id=args.policy_id,
            r_train=args.r_train,
            r_eval=args.r_eval,
            initial_budget=args.initial_budget,
            extend_step_plies=args.extend_step_plies,
            max_total_plies=args.max_total_plies,
            max_extensions=args.max_extensions,
            demo_train_families=args.demo_train_families,
            demo_val_families=args.demo_val_families,
            demo_test_families=args.demo_test_families,
            train_families=args.train_families,
            val_families=args.val_families,
            test_families=args.test_families,
            min_ply=args.min_ply,
            workers=args.workers,
            seed=args.seed,
            val_first=args.val_first,
            pilot_paths=args.pilot_paths,
            check_pilot=not args.no_pilot_check,
        )
    elif args.mode == "train_wave":
        run_train_wave(
            output_dir=args.output_dir,
            wave_id=args.wave_id,
            variants=args.variants,
            policy_id=args.policy_id,
            r_train=args.r_train,
            initial_budget=args.initial_budget,
            extend_step_plies=args.extend_step_plies,
            max_total_plies=args.max_total_plies,
            max_extensions=args.max_extensions,
            train_families=args.train_families,
            train_demo_games=args.train_demo_games,
            min_ply=args.min_ply,
            workers=args.workers,
            seed=args.seed,
            pilot_paths=args.pilot_paths,
            check_pilot=not args.no_pilot_check,
        )
    elif args.mode == "freeze_snapshot":
        src = args.source_dir or args.output_dir
        snap_dir = args.snapshot_dir or (args.output_dir / "snapshots" / args.snapshot_id)
        freeze_snapshot(
            source_dir=src,
            snapshot_dir=snap_dir,
            snapshot_id=args.snapshot_id,
            variants=args.variants,
            panel_families=args.panel_families,
            pilot_paths=args.pilot_paths,
            check_pilot=not args.no_pilot_check,
        )
    elif args.mode == "r256_subset":
        src = args.source_dir or args.output_dir
        run_r256_subset(
            source_dir=src,
            out_dir=args.output_dir,
            variants=args.variants,
            subset_families=args.subset_families,
            r_eval=args.r256_eval,
            policy_id=args.policy_id,
            initial_budget=args.initial_budget,
            extend_step_plies=args.extend_step_plies,
            max_total_plies=args.max_total_plies,
            max_extensions=args.max_extensions,
            workers=args.workers,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
