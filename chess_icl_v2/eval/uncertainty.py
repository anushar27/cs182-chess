"""Hierarchical uncertainty quantification for chess_icl_v2.

Supports:
  1. Cluster bootstrap over source_game_id (accounting for correlated roots from the same game).
  2. Aggregation across model_seeds and demo_draws (e.g., 5 draws per K).
  3. Unweighted macro-averaging across variants with per-variant and aggregate CIs.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Hashable, Mapping, Optional, Sequence

import numpy as np


def _cluster_bootstrap_distribution(
    per_root_metrics: Sequence[float] | np.ndarray,
    source_game_ids: Sequence[Hashable],
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, np.ndarray, int]:
    """Run vectorized cluster bootstrap over source_game_ids.

    Returns (observed_mean, boot_means_array, n_clusters).
    """
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    x = np.asarray(per_root_metrics, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"per_root_metrics must be 1D, got shape {x.shape}")
    if len(x) == 0:
        raise ValueError("per_root_metrics cannot be empty")
    if len(source_game_ids) != len(x):
        raise ValueError(
            f"Length mismatch: per_root_metrics ({len(x)}) vs source_game_ids ({len(source_game_ids)})"
        )

    cluster_sums: dict[Hashable, float] = defaultdict(float)
    cluster_counts: dict[Hashable, int] = defaultdict(int)
    ordered_keys: list[Hashable] = []
    for val, gid in zip(x, source_game_ids):
        if gid not in cluster_counts:
            ordered_keys.append(gid)
        cluster_sums[gid] += float(val)
        cluster_counts[gid] += 1

    g = len(ordered_keys)
    s_arr = np.array([cluster_sums[k] for k in ordered_keys], dtype=np.float64)
    m_arr = np.array([cluster_counts[k] for k in ordered_keys], dtype=np.float64)
    obs_mean = float(np.mean(x))

    if g == 1:
        boot_means = np.full((n_boot,), obs_mean, dtype=np.float64)
        return obs_mean, boot_means, g

    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, g, size=(n_boot, g))
    boot_means = np.sum(s_arr[boot_idx], axis=1) / np.sum(m_arr[boot_idx], axis=1)
    return obs_mean, boot_means, g


def bootstrap_by_source_game(
    per_root_metrics: Sequence[float] | np.ndarray,
    source_game_ids: Sequence[Hashable],
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Cluster-bootstrap per-root metrics grouped by `source_game_ids`.

    Args:
        per_root_metrics: 1D array of metric values per query root.
        source_game_ids: Sequence of game identifiers (same length as `per_root_metrics`).
        n_boot: Number of bootstrap replicates.
        seed: Random seed for reproducible resampling.
        alpha: Significance level (default 0.05 for 95% CI).

    Returns:
        Dict containing mean, boot_mean, std, ci_low, ci_high, ci ([lo, hi]),
        n_roots, n_clusters, and n_boot.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    obs_mean, boot_means, g = _cluster_bootstrap_distribution(
        per_root_metrics, source_game_ids, n_boot=n_boot, seed=seed
    )
    ci_low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    ci_high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    boot_std = float(np.std(boot_means, ddof=1)) if n_boot > 1 else 0.0

    return {
        "mean": obs_mean,
        "boot_mean": float(np.mean(boot_means)),
        "std": boot_std,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci": [ci_low, ci_high],
        "n_roots": int(len(per_root_metrics)),
        "n_clusters": int(g),
        "n_boot": int(n_boot),
    }


def _to_3d_seed_draw_root(values: Any) -> np.ndarray:
    """Normalize 1D (roots), 2D (draws, roots), or 3D (seeds, draws, roots) array to 3D."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return arr[None, None, :]
    if arr.ndim == 2:
        return arr[None, :, :]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Expected 1D, 2D, or 3D array, got shape {arr.shape}")


def aggregate_seed_draw_hierarchy(
    values: Any,
    source_game_ids: Optional[Sequence[Hashable]] = None,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Aggregate metric values of shape (n_seeds, n_draws, n_roots) with full uncertainty.

    Computes:
      - Mean across seeds, draws, and roots.
      - Per-seed means and across-seed standard deviation/SE.
      - Within-seed across-draw standard deviation/SE.
      - Cluster bootstrap over `source_game_ids` on seed-and-draw-averaged root values.
      - Combined SE = sqrt(game_boot_std^2 + seed_se^2).
    """
    arr = _to_3d_seed_draw_root(values)
    n_seeds, n_draws, n_roots = arr.shape
    if n_seeds == 0 or n_draws == 0 or n_roots == 0:
        raise ValueError(f"Cannot aggregate empty dimension in shape {arr.shape}")

    if source_game_ids is None:
        gids: Sequence[Hashable] = list(range(n_roots))
    else:
        gids = source_game_ids

    per_seed_draw_means = np.mean(arr, axis=2)  # (n_seeds, n_draws)
    per_seed_means = np.mean(per_seed_draw_means, axis=1)  # (n_seeds,)
    per_root_mean = np.mean(arr, axis=(0, 1))  # (n_roots,)

    seed_std = float(np.std(per_seed_means, ddof=1)) if n_seeds > 1 else 0.0
    seed_se = float(seed_std / np.sqrt(n_seeds)) if n_seeds > 1 else 0.0

    if n_draws > 1:
        draw_stds = np.std(per_seed_draw_means, axis=1, ddof=1)
        draw_std = float(np.mean(draw_stds))
        draw_se = float(draw_std / np.sqrt(n_draws))
    else:
        draw_std = 0.0
        draw_se = 0.0

    boot_res = bootstrap_by_source_game(
        per_root_mean, gids, n_boot=n_boot, seed=seed, alpha=alpha
    )
    combined_se = float(np.sqrt(boot_res["std"] ** 2 + seed_se ** 2))

    return {
        "mean": float(np.mean(per_root_mean)),
        "seed_means": [float(v) for v in per_seed_means],
        "seed_std": seed_std,
        "seed_se": seed_se,
        "draw_std": draw_std,
        "draw_se": draw_se,
        "game_bootstrap": boot_res,
        "combined_se": combined_se,
        "ci": boot_res["ci"],
        "n_seeds": int(n_seeds),
        "n_draws": int(n_draws),
        "n_roots": int(n_roots),
    }


def macro_average_variants(
    per_variant_metrics: Mapping[str, Any],
    source_game_ids_by_variant: Optional[Mapping[str, Sequence[Hashable]]] = None,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compute unweighted macro-average across variants with per-variant and aggregate CIs.

    Uses stratified cluster bootstrap (resampling source games within each variant)
    to form the macro-average bootstrap distribution and 95% confidence interval.
    """
    if not per_variant_metrics:
        raise ValueError("per_variant_metrics cannot be empty")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    variant_names = sorted(per_variant_metrics.keys())
    per_variant_summary: dict[str, dict[str, Any]] = {}
    variant_boot_distributions: list[np.ndarray] = []
    variant_arrays: dict[str, np.ndarray] = {}

    for idx, v in enumerate(variant_names):
        arr = _to_3d_seed_draw_root(per_variant_metrics[v])
        variant_arrays[v] = arr
        gids = (
            source_game_ids_by_variant[v]
            if source_game_ids_by_variant is not None and v in source_game_ids_by_variant
            else list(range(arr.shape[2]))
        )
        v_seed = seed + idx + 1
        summary = aggregate_seed_draw_hierarchy(
            arr, source_game_ids=gids, n_boot=n_boot, seed=v_seed, alpha=alpha
        )
        per_variant_summary[v] = summary

        per_root_mean = np.mean(arr, axis=(0, 1))
        _, v_boot_means, _ = _cluster_bootstrap_distribution(
            per_root_mean, gids, n_boot=n_boot, seed=v_seed
        )
        variant_boot_distributions.append(v_boot_means)

    macro_boot = np.mean(np.vstack(variant_boot_distributions), axis=0)
    var_means = np.array([per_variant_summary[v]["mean"] for v in variant_names], dtype=np.float64)
    macro_mean = float(np.mean(var_means))
    across_variant_std = float(np.std(var_means, ddof=1)) if len(var_means) > 1 else 0.0

    # Seed and draw variance of the macro-average when variants share seed/draw counts
    seed_counts = {variant_arrays[v].shape[0] for v in variant_names}
    draw_counts = {variant_arrays[v].shape[1] for v in variant_names}
    if len(seed_counts) == 1:
        n_seeds = next(iter(seed_counts))
        macro_per_seed = np.mean(
            np.vstack(
                [np.mean(variant_arrays[v], axis=(1, 2)) for v in variant_names]
            ),
            axis=0,
        )
        macro_seed_std = float(np.std(macro_per_seed, ddof=1)) if n_seeds > 1 else 0.0
        macro_seed_se = float(macro_seed_std / np.sqrt(n_seeds)) if n_seeds > 1 else 0.0
    else:
        macro_seed_std = float(
            np.mean([per_variant_summary[v]["seed_std"] for v in variant_names])
        )
        macro_seed_se = float(
            np.sqrt(np.sum([per_variant_summary[v]["seed_se"] ** 2 for v in variant_names]))
            / len(variant_names)
        )

    if len(seed_counts) == 1 and len(draw_counts) == 1:
        n_draws = next(iter(draw_counts))
        macro_per_seed_draw = np.mean(
            np.stack([np.mean(variant_arrays[v], axis=2) for v in variant_names], axis=0),
            axis=0,
        )  # (n_seeds, n_draws)
        macro_draw_std = (
            float(np.mean(np.std(macro_per_seed_draw, axis=1, ddof=1)))
            if n_draws > 1
            else 0.0
        )
    else:
        macro_draw_std = float(
            np.mean([per_variant_summary[v]["draw_std"] for v in variant_names])
        )

    macro_boot_std = float(np.std(macro_boot, ddof=1)) if n_boot > 1 else 0.0
    ci_low = float(np.percentile(macro_boot, 100.0 * (alpha / 2.0)))
    ci_high = float(np.percentile(macro_boot, 100.0 * (1.0 - alpha / 2.0)))
    combined_se = float(np.sqrt(macro_boot_std ** 2 + macro_seed_se ** 2))

    return {
        "macro_mean": macro_mean,
        "macro_boot_mean": float(np.mean(macro_boot)),
        "macro_std": macro_boot_std,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci": [ci_low, ci_high],
        "across_variant_std": across_variant_std,
        "macro_seed_std": macro_seed_std,
        "macro_seed_se": macro_seed_se,
        "macro_draw_std": macro_draw_std,
        "combined_se": combined_se,
        "n_variants": len(variant_names),
        "per_variant": per_variant_summary,
    }


def aggregate_evaluation_records(
    records: Sequence[Mapping[str, Any]],
    value_key: str = "nll",
    variant_key: str = "variant",
    seed_key: str = "model_seed",
    draw_key: str = "demo_draw",
    root_key: str = "root_id",
    game_key: str = "source_game_id",
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Aggregate flat evaluation records across model_seeds, demo_draws, roots, and variants."""
    if not records:
        raise ValueError("records cannot be empty")

    by_variant: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for rec in records:
        by_variant[str(rec[variant_key])].append(rec)

    per_variant_arrays: dict[str, np.ndarray] = {}
    gids_by_variant: dict[str, list[Hashable]] = {}

    for v, v_recs in sorted(by_variant.items()):
        seeds = sorted({rec.get(seed_key, 0) for rec in v_recs})
        draws = sorted({rec.get(draw_key, 0) for rec in v_recs})
        roots = sorted({rec[root_key] for rec in v_recs}, key=lambda x: str(x))

        seed_to_i = {s: i for i, s in enumerate(seeds)}
        draw_to_j = {d: j for j, d in enumerate(draws)}
        root_to_k = {r: k for k, r in enumerate(roots)}

        tensor = np.full((len(seeds), len(draws), len(roots)), np.nan, dtype=np.float64)
        root_to_gid: dict[Any, Hashable] = {}
        for rec in v_recs:
            si = seed_to_i[rec.get(seed_key, 0)]
            dj = draw_to_j[rec.get(draw_key, 0)]
            rk = root_to_k[rec[root_key]]
            tensor[si, dj, rk] = float(rec[value_key])
            root_to_gid[rec[root_key]] = rec.get(game_key, rec[root_key])

        if np.any(np.isnan(tensor)):
            raise ValueError(
                f"Incomplete (model_seed, demo_draw, root_id) grid for variant {v!r}"
            )
        per_variant_arrays[v] = tensor
        gids_by_variant[v] = [root_to_gid[r] for r in roots]

    return macro_average_variants(
        per_variant_arrays,
        source_game_ids_by_variant=gids_by_variant,
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
    )
