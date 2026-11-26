"""Run all non-neural baselines on chess_icl_v2 JSONL data across K and 5 demo draws.

Reads `<data_dir>/<variant>/{games_demo,positions_train,positions_val,positions_test}.jsonl`.
Fits learned baselines (TrainingPrior, LengthHistogram, QueryMaterial) strictly on
`positions_train.jsonl`, evaluates all 6 non-neural baselines on `val` or `test`,
and saves `baselines_<split>.json` + `baselines_<split>_summary.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Optional, Sequence

import chess
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_icl_v2.eval import (  # noqa: E402
    BoardKNNBaseline,
    DemoMajorityBaseline,
    LengthHistogramBaseline,
    QueryMaterialBaseline,
    TrainingPriorBaseline,
    UniformBaseline,
    bootstrap_by_source_game,
    compute_argmax_accuracy,
    compute_brier,
    compute_l1_tv,
    compute_mc_noise_floor,
    compute_nll,
    compute_side_calibration,
    filter_equal_material,
    macro_average_variants,
)

DEFAULT_K_VALUES: tuple[int, ...] = (0, 1, 2, 4, 8, 16)
CONTEXT_INVARIANT_BASELINES: frozenset[str] = frozenset(
    {"uniform", "training_prior", "query_material"}
)


def _stable_hash(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSON lines from `path`."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                records.append(json.loads(s))
    return records


def _inject_blocked_squares_into_fen(
    fen: str,
    blocked_squares: Optional[Sequence[Any]],
) -> str:
    """Embed `blocked_squares` (0..63 or algebraic names) as 'X' in FEN placement."""
    if not blocked_squares:
        return fen
    parts = fen.strip().split()
    placement = parts[0]
    if "X" in placement:
        return fen

    ranks_str = placement.split("/")
    if len(ranks_str) != 8:
        return fen

    grid: list[str] = []
    for rank_s in ranks_str:
        for ch in rank_s:
            if ch.isdigit():
                grid.extend(["."] * int(ch))
            else:
                grid.append(ch)
    if len(grid) != 64:
        return fen

    files = "abcdefgh"
    for sq in blocked_squares:
        if isinstance(sq, int):
            sq_idx = sq
        elif isinstance(sq, str) and len(sq) == 2:
            sq_idx = (int(sq[1]) - 1) * 8 + files.index(sq[0].lower())
        else:
            continue
        if 0 <= sq_idx < 64:
            rank_from_top = 7 - (sq_idx // 8)
            file_idx = sq_idx % 8
            grid[rank_from_top * 8 + file_idx] = "X"

    new_ranks: list[str] = []
    for r in range(8):
        row = grid[r * 8 : (r + 1) * 8]
        compacted: list[str] = []
        empty = 0
        for ch in row:
            if ch == ".":
                empty += 1
            else:
                if empty > 0:
                    compacted.append(str(empty))
                    empty = 0
                compacted.append(ch)
        if empty > 0:
            compacted.append(str(empty))
        new_ranks.append("".join(compacted))

    parts[0] = "/".join(new_ranks)
    return " ".join(parts)


def _normalize_target(pos: dict[str, Any]) -> np.ndarray:
    """Extract soft [p_win, p_draw, p_loss] target from a position record."""
    for key in ("y_true", "target", "target_probs", "outcome_probs", "probs"):
        if key in pos and pos[key] is not None:
            arr = np.asarray(pos[key], dtype=np.float64)
            if arr.shape == (3,):
                s = float(np.sum(arr))
                return arr / s if s > 0 else np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    if "win_prob" in pos and "draw_prob" in pos and "loss_prob" in pos:
        arr = np.array([pos["win_prob"], pos["draw_prob"], pos["loss_prob"]], dtype=np.float64)
        s = float(np.sum(arr))
        return arr / s if s > 0 else np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    label = str(pos.get("label", "draw")).strip().lower()
    mapping = {
        "win": [1.0, 0.0, 0.0],
        "w": [1.0, 0.0, 0.0],
        "draw": [0.0, 1.0, 0.0],
        "d": [0.0, 1.0, 0.0],
        "loss": [0.0, 0.0, 1.0],
        "l": [0.0, 0.0, 1.0],
    }
    return np.array(mapping.get(label, [0.0, 1.0, 0.0]), dtype=np.float64)


def _normalize_position(
    pos: dict[str, Any],
    variant: str,
    idx: int,
    default_r: int = 20,
) -> dict[str, Any]:
    """Normalize a JSONL position record into an episode template dict."""
    raw_fen = (
        pos.get("query_fen")
        or pos.get("fen")
        or pos.get("state")
        or chess.STARTING_FEN
    )
    blocked = pos.get("blocked_squares") if pos.get("blocked_squares") is not None else pos.get("blocked")
    fen = _inject_blocked_squares_into_fen(str(raw_fen), blocked)
    parts = fen.split()
    side = pos.get("side_to_move") or (parts[1].lower() if len(parts) >= 2 else "w")
    if side not in ("w", "b"):
        side = "w" if str(side).lower().startswith("w") else "b"

    y_true = _normalize_target(pos)
    root_id = str(pos.get("root_id") or pos.get("id") or f"{variant}_root_{idx}")
    source_game_id = str(
        pos.get("source_game_id")
        or pos.get("source_game")
        or pos.get("game_id")
        or root_id
    )
    r_val = int(pos.get("R") or pos.get("n_rollouts") or pos.get("num_rollouts") or default_r)

    return {
        "variant": variant,
        "query_fen": fen,
        "side_to_move": side,
        "y_true": y_true.tolist(),
        "root_id": root_id,
        "source_game_id": source_game_id,
        "R": r_val,
        "demos": [],
    }


def _normalize_demo(demo: dict[str, Any]) -> dict[str, Any]:
    """Normalize a JSONL demo record into canonical demo dict."""
    raw_init = (
        demo.get("initial_fen")
        or demo.get("initial_state")
        or demo.get("start_fen")
        or chess.STARTING_FEN
    )
    blocked = demo.get("blocked_squares") if demo.get("blocked_squares") is not None else demo.get("blocked")
    init_fen = _inject_blocked_squares_into_fen(str(raw_init), blocked)
    moves = demo.get("moves") or demo.get("uci_moves") or demo.get("san_moves") or []
    if isinstance(moves, str):
        moves = [m for m in moves.strip().split() if m]
    outcome = demo.get("outcome") or demo.get("result") or demo.get("outcome_white") or "draw"
    return {
        "initial_fen": init_fen,
        "moves": list(moves),
        "outcome": outcome,
    }


def sample_demos(
    demo_pool: Sequence[dict[str, Any]],
    k: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Deterministically sample `min(k, len(demo_pool))` demos without replacement."""
    if k <= 0 or not demo_pool:
        return []
    rng = np.random.default_rng(seed)
    k_eff = min(int(k), len(demo_pool))
    chosen = rng.choice(len(demo_pool), size=k_eff, replace=False)
    return [demo_pool[int(i)] for i in chosen]


def discover_variants(
    data_dir: Path,
    split_filename: str,
    requested: Optional[Sequence[str]] = None,
) -> list[str]:
    """Discover variant subdirectories containing `split_filename` and `games_demo.jsonl`."""
    if requested:
        variants = list(requested)
    else:
        variants = sorted(
            p.name
            for p in data_dir.iterdir()
            if p.is_dir()
            and (p / split_filename).exists()
            and (p / "games_demo.jsonl").exists()
        )
    return variants


def build_train_episodes(
    data_dir: Path,
    k_values: Sequence[int],
    seed: int,
    max_per_variant: Optional[int] = None,
    default_r: int = 20,
) -> list[dict[str, Any]]:
    """Build training episodes strictly from `positions_train.jsonl` + `games_demo.jsonl`."""
    train_variants = discover_variants(data_dir, "positions_train.jsonl")
    if not train_variants:
        raise FileNotFoundError(
            f"No variants with positions_train.jsonl + games_demo.jsonl found under {data_dir}"
        )

    rng = np.random.default_rng(seed + 99991)
    train_episodes: list[dict[str, Any]] = []
    for variant in train_variants:
        v_dir = data_dir / variant
        demos = [_normalize_demo(d) for d in load_jsonl(v_dir / "games_demo.jsonl")]
        raw_pos = load_jsonl(v_dir / "positions_train.jsonl")
        if max_per_variant is not None:
            raw_pos = raw_pos[:max_per_variant]

        for idx, pos in enumerate(raw_pos):
            ep = _normalize_position(pos, variant=variant, idx=idx, default_r=default_r)
            valid_ks = [k for k in k_values if k <= len(demos)] or [0]
            k_pick = int(valid_ks[int(rng.integers(0, len(valid_ks)))])
            draw_seed = int(rng.integers(0, 2**31 - 1))
            ep["demos"] = sample_demos(demos, k_pick, seed=draw_seed)
            train_episodes.append(ep)

    return train_episodes


def evaluate_baselines(
    data_dir: Path,
    output_dir: Path,
    split: str = "val",
    variants: Optional[Sequence[str]] = None,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    n_draws: int = 5,
    seed: int = 42,
    n_boot: int = 1000,
    default_r: int = 20,
    knn_top_k: int = 10,
    knn_metric: str = "cosine",
    knn_state_mode: str = "trajectories",
    max_train_per_variant: Optional[int] = None,
    max_eval_per_variant: Optional[int] = None,
) -> dict[str, Any]:
    """Fit learned baselines strictly on train split and evaluate all baselines on `split`."""
    if split not in {"val", "test"}:
        raise ValueError(f"split must be 'val' or 'test', got {split!r}")

    split_filename = f"positions_{split}.jsonl"
    eval_variants = discover_variants(data_dir, split_filename, requested=variants)
    if not eval_variants:
        raise FileNotFoundError(
            f"No variants with {split_filename} + games_demo.jsonl found under {data_dir}"
        )

    # 1. Fit parametric baselines strictly on training split
    train_episodes = build_train_episodes(
        data_dir=data_dir,
        k_values=k_values,
        seed=seed,
        max_per_variant=max_train_per_variant,
        default_r=default_r,
    )
    prior_model = TrainingPriorBaseline().fit(train_episodes)
    length_model = LengthHistogramBaseline().fit(train_episodes)
    material_model = QueryMaterialBaseline().fit(train_episodes)

    models: dict[str, Any] = {
        "uniform": UniformBaseline(),
        "training_prior": prior_model,
        "query_material": material_model,
        "demo_majority": DemoMajorityBaseline(alpha=1.0, prior=prior_model.prior_),
        "length_histogram": length_model,
        "board_knn": BoardKNNBaseline(
            top_k=knn_top_k,
            metric=knn_metric,  # type: ignore[arg-type]
            state_mode=knn_state_mode,  # type: ignore[arg-type]
            prior=prior_model.prior_,
        ),
    }

    # 2. Load eval split data per variant
    eval_data: dict[str, dict[str, Any]] = {}
    for variant in eval_variants:
        v_dir = data_dir / variant
        demos = [_normalize_demo(d) for d in load_jsonl(v_dir / "games_demo.jsonl")]
        raw_pos = load_jsonl(v_dir / split_filename)
        if max_eval_per_variant is not None:
            raw_pos = raw_pos[:max_eval_per_variant]
        positions = [
            _normalize_position(pos, variant=variant, idx=idx, default_r=default_r)
            for idx, pos in enumerate(raw_pos)
        ]
        y_true = np.vstack([ep["y_true"] for ep in positions])
        sides = [ep["side_to_move"] for ep in positions]
        gids = [ep["source_game_id"] for ep in positions]
        eq_slice, eq_idx = filter_equal_material(positions, threshold=2.0, return_indices=True)
        r_eff = int(round(float(np.mean([ep["R"] for ep in positions]))))
        mc_floor = compute_mc_noise_floor(y_true, R=max(r_eff, 1))
        eval_data[variant] = {
            "demos": demos,
            "positions": positions,
            "y_true": y_true,
            "sides": sides,
            "source_game_ids": gids,
            "equal_material_indices": eq_idx,
            "n_equal_material": len(eq_slice),
            "mc_noise_floor": mc_floor,
        }

    # 3. Evaluate across models, variants, K, and demo draws
    sorted_ks = sorted({int(k) for k in k_values})
    results_by_model: dict[str, dict[str, Any]] = {}

    for model_name, predictor in models.items():
        per_k_results: dict[str, Any] = {}
        # Store K=0 per-root NLLs per variant for Context Gain computation
        k0_nll_by_variant: dict[str, np.ndarray] = {}

        # Precompute K-invariant predictions once per variant if applicable
        invariant_preds_by_variant: dict[str, np.ndarray] = {}
        if model_name in CONTEXT_INVARIANT_BASELINES:
            for variant in eval_variants:
                invariant_preds_by_variant[variant] = predictor.predict_proba(
                    eval_data[variant]["positions"]
                )

        for k in sorted_ks:
            per_variant_res: dict[str, Any] = {}
            macro_nll_inputs: dict[str, np.ndarray] = {}
            macro_gain_inputs: dict[str, np.ndarray] = {}
            macro_eq_nll_inputs: dict[str, np.ndarray] = {}
            gids_by_variant: dict[str, list[str]] = {}
            eq_gids_by_variant: dict[str, list[str]] = {}

            for variant in eval_variants:
                v_info = eval_data[variant]
                y_true = v_info["y_true"]
                sides = v_info["sides"]
                gids = v_info["source_game_ids"]
                eq_idx = v_info["equal_material_indices"]
                gids_by_variant[variant] = gids

                n_roots = len(y_true)
                n_eff_draws = 1 if (k == 0 or model_name in CONTEXT_INVARIANT_BASELINES) else n_draws
                nll_draws = np.empty((n_draws, n_roots), dtype=np.float64)
                probs_draws = np.empty((n_draws, n_roots, 3), dtype=np.float64)

                for d_idx in range(n_draws):
                    if d_idx >= n_eff_draws:
                        nll_draws[d_idx] = nll_draws[0]
                        probs_draws[d_idx] = probs_draws[0]
                        continue

                    if model_name in CONTEXT_INVARIANT_BASELINES:
                        probs = invariant_preds_by_variant[variant]
                    else:
                        draw_seed = (
                            seed + d_idx * 10007 + k * 101 + _stable_hash(variant) % 10000
                        )
                        sampled = sample_demos(v_info["demos"], k=k, seed=draw_seed)
                        episodes = []
                        for pos in v_info["positions"]:
                            ep_c = dict(pos)
                            ep_c["demos"] = sampled
                            episodes.append(ep_c)
                        probs = predictor.predict_proba(episodes)

                    _, per_sample_nll = compute_nll(probs, y_true)
                    nll_draws[d_idx] = per_sample_nll
                    probs_draws[d_idx] = probs

                mean_probs = np.mean(probs_draws, axis=0)
                per_root_nll = np.mean(nll_draws, axis=0)
                if k == 0:
                    k0_nll_by_variant[variant] = per_root_nll

                nll_boot = bootstrap_by_source_game(
                    per_root_nll, gids, n_boot=n_boot, seed=seed
                )
                draw_means = np.mean(nll_draws, axis=1)
                draw_std = float(np.std(draw_means, ddof=1)) if n_draws > 1 else 0.0

                brier_val = compute_brier(mean_probs, y_true)
                tv_l1 = compute_l1_tv(mean_probs, y_true)
                acc_res = compute_argmax_accuracy(mean_probs, y_true)
                side_cal = compute_side_calibration(mean_probs, y_true, sides)

                # Equal-material slice
                if eq_idx:
                    eq_nll_mean, eq_nll_per_root = compute_nll(
                        mean_probs[eq_idx], y_true[eq_idx]
                    )
                    eq_gids = [gids[i] for i in eq_idx]
                    eq_boot = bootstrap_by_source_game(
                        eq_nll_per_root, eq_gids, n_boot=n_boot, seed=seed
                    )
                    eq_brier = compute_brier(mean_probs[eq_idx], y_true[eq_idx])
                    eq_tv_l1 = compute_l1_tv(mean_probs[eq_idx], y_true[eq_idx])
                    eq_acc = compute_argmax_accuracy(mean_probs[eq_idx], y_true[eq_idx])["accuracy"]
                    macro_eq_nll_inputs[variant] = eq_nll_per_root
                    eq_gids_by_variant[variant] = eq_gids
                    eq_summary: dict[str, Any] = {
                        "n_samples": len(eq_idx),
                        "nll": float(eq_nll_mean),
                        "nll_boot_std": eq_boot["std"],
                        "nll_ci": eq_boot["ci"],
                        "brier": float(eq_brier),
                        "tv": float(eq_tv_l1["tv"]),
                        "accuracy": float(eq_acc),
                    }
                else:
                    eq_summary = {"n_samples": 0, "nll": None, "brier": None, "tv": None, "accuracy": None}

                # Context gain vs K=0 on identical roots (if K=0 was evaluated)
                if 0 in sorted_ks and variant in k0_nll_by_variant:
                    per_root_gain = k0_nll_by_variant[variant] - per_root_nll
                    gain_boot = bootstrap_by_source_game(
                        per_root_gain, gids, n_boot=n_boot, seed=seed
                    )
                    macro_gain_inputs[variant] = per_root_gain
                else:
                    gain_boot = {"mean": 0.0, "std": 0.0, "ci": [0.0, 0.0]}

                macro_nll_inputs[variant] = nll_draws  # (n_draws, n_roots)

                per_variant_res[variant] = {
                    "n_roots": n_roots,
                    "nll": nll_boot["mean"],
                    "nll_boot_std": nll_boot["std"],
                    "nll_ci": nll_boot["ci"],
                    "draw_std": draw_std,
                    "brier": brier_val,
                    "tv": tv_l1["tv"],
                    "l1": tv_l1["l1"],
                    "accuracy": acc_res["accuracy"],
                    "pred_counts": acc_res["pred_counts"],
                    "true_counts": acc_res["true_counts"],
                    "side_calibration": side_cal,
                    "equal_material": eq_summary,
                    "context_gain": {
                        "mean": gain_boot["mean"],
                        "std": gain_boot["std"],
                        "ci": gain_boot["ci"],
                    },
                    "mc_noise_floor": v_info["mc_noise_floor"],
                }

            macro_nll = macro_average_variants(
                macro_nll_inputs,
                source_game_ids_by_variant=gids_by_variant,
                n_boot=n_boot,
                seed=seed,
            )
            macro_gain = (
                macro_average_variants(
                    macro_gain_inputs,
                    source_game_ids_by_variant=gids_by_variant,
                    n_boot=n_boot,
                    seed=seed,
                )
                if macro_gain_inputs
                else {"macro_mean": 0.0, "macro_std": 0.0, "ci": [0.0, 0.0]}
            )
            macro_eq_nll = (
                macro_average_variants(
                    macro_eq_nll_inputs,
                    source_game_ids_by_variant=eq_gids_by_variant,
                    n_boot=n_boot,
                    seed=seed,
                )
                if macro_eq_nll_inputs
                else {"macro_mean": None, "macro_std": None, "ci": None}
            )

            macro_brier = float(np.mean([per_variant_res[v]["brier"] for v in eval_variants]))
            macro_tv = float(np.mean([per_variant_res[v]["tv"] for v in eval_variants]))
            macro_acc = float(np.mean([per_variant_res[v]["accuracy"] for v in eval_variants]))
            white_nlls = [
                per_variant_res[v]["side_calibration"]["white"]["nll"]
                for v in eval_variants
                if per_variant_res[v]["side_calibration"]["white"]["nll"] is not None
            ]
            black_nlls = [
                per_variant_res[v]["side_calibration"]["black"]["nll"]
                for v in eval_variants
                if per_variant_res[v]["side_calibration"]["black"]["nll"] is not None
            ]
            white_eces = [
                per_variant_res[v]["side_calibration"]["white"]["ece"]
                for v in eval_variants
                if per_variant_res[v]["side_calibration"]["white"]["ece"] is not None
            ]
            black_eces = [
                per_variant_res[v]["side_calibration"]["black"]["ece"]
                for v in eval_variants
                if per_variant_res[v]["side_calibration"]["black"]["ece"] is not None
            ]

            per_k_results[str(k)] = {
                "macro": {
                    "nll": macro_nll["macro_mean"],
                    "nll_boot_std": macro_nll["macro_std"],
                    "nll_ci": macro_nll["ci"],
                    "draw_std": macro_nll["macro_draw_std"],
                    "brier": macro_brier,
                    "tv": macro_tv,
                    "accuracy": macro_acc,
                    "white_nll": float(np.mean(white_nlls)) if white_nlls else None,
                    "black_nll": float(np.mean(black_nlls)) if black_nlls else None,
                    "white_ece": float(np.mean(white_eces)) if white_eces else None,
                    "black_ece": float(np.mean(black_eces)) if black_eces else None,
                    "equal_material_nll": macro_eq_nll["macro_mean"],
                    "equal_material_ci": macro_eq_nll["ci"],
                    "context_gain": macro_gain["macro_mean"],
                    "context_gain_std": macro_gain["macro_std"],
                    "context_gain_ci": macro_gain["ci"],
                },
                "per_variant": per_variant_res,
            }

        results_by_model[model_name] = {"by_k": per_k_results}

    macro_mc_entropy = float(
        np.mean([eval_data[v]["mc_noise_floor"]["entropy_floor"] for v in eval_variants])
    )
    macro_mc_debiased = float(
        np.mean(
            [eval_data[v]["mc_noise_floor"]["debiased_entropy_floor"] for v in eval_variants]
        )
    )

    payload: dict[str, Any] = {
        "split": split,
        "variants": eval_variants,
        "k_values": sorted_ks,
        "n_draws": int(n_draws),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "n_train_episodes": len(train_episodes),
        "mc_noise_floor_macro": {
            "entropy_floor": macro_mc_entropy,
            "debiased_entropy_floor": macro_mc_debiased,
        },
        "models": results_by_model,
    }

    summary = {
        "split": split,
        "variants": eval_variants,
        "k_values": sorted_ks,
        "n_draws": int(n_draws),
        "mc_noise_floor_macro": payload["mc_noise_floor_macro"],
        "models": {
            m: {k_str: k_data["macro"] for k_str, k_data in m_data["by_k"].items()}
            for m, m_data in results_by_model.items()
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / f"baselines_{split}.json"
    summary_path = output_dir / f"baselines_{split}_summary.json"
    full_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run chess_icl_v2 non-neural baselines.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Root directory with <variant>/ JSONLs.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for output JSON files.")
    parser.add_argument("--split", choices=("val", "test"), default="val", help="Evaluation split.")
    parser.add_argument("--variants", nargs="*", default=None, help="Optional subset of variants to evaluate.")
    parser.add_argument("--k-values", nargs="+", type=int, default=list(DEFAULT_K_VALUES), help="Context lengths K.")
    parser.add_argument("--n-draws", type=int, default=5, help="Demo draw seeds per K.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    parser.add_argument("--n-boot", type=int, default=1000, help="Cluster bootstrap replicates.")
    parser.add_argument("--R", type=int, default=20, dest="default_r", help="Fallback rollout count R.")
    parser.add_argument("--knn-top-k", type=int, default=10, help="Top-k neighbors for BoardKNNBaseline.")
    parser.add_argument("--knn-metric", choices=("cosine", "hamming"), default="cosine")
    parser.add_argument("--knn-state-mode", choices=("trajectories", "final"), default="trajectories")
    parser.add_argument("--max-train-per-variant", type=int, default=None)
    parser.add_argument("--max-eval-per-variant", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    payload = evaluate_baselines(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split=args.split,
        variants=args.variants,
        k_values=args.k_values,
        n_draws=args.n_draws,
        seed=args.seed,
        n_boot=args.n_boot,
        default_r=args.default_r,
        knn_top_k=args.knn_top_k,
        knn_metric=args.knn_metric,
        knn_state_mode=args.knn_state_mode,
        max_train_per_variant=args.max_train_per_variant,
        max_eval_per_variant=args.max_eval_per_variant,
    )
    print(
        f"[run_baselines] Saved {args.split} results for {len(payload['variants'])} variants "
        f"across K={payload['k_values']} to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
