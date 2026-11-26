"""Proper scoring metrics, calibration, context gain, and MC noise floor for chess_icl_v2.

Class index convention throughout chess_icl_v2:
    0 -> win  (from query side-to-move perspective)
    1 -> draw
    2 -> loss (from query side-to-move perspective)
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

CLASS_NAMES: tuple[str, str, str] = ("win", "draw", "loss")


def _as_prob_matrix(arr: Any, name: str) -> np.ndarray:
    """Validate and convert input into a 2D float64 array of shape (N, 3)."""
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 2 or out.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {out.shape}")
    if out.size > 0 and not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values")
    return out


def _normalize_side(side: Any) -> str:
    """Normalize a side-to-move indicator to 'white' or 'black'."""
    if isinstance(side, (bool, np.bool_)):
        return "white" if bool(side) else "black"
    if isinstance(side, str):
        s = side.strip().lower()
        if s in {"w", "white"}:
            return "white"
        if s in {"b", "black"}:
            return "black"
    raise ValueError(
        f"Unrecognized side-to-move value {side!r}; expected 'w'/'white'/True or 'b'/'black'/False"
    )


def compute_nll(
    p_pred: Any,
    y_true: Any,
    eps: float = 1e-7,
) -> tuple[float, np.ndarray]:
    """Compute soft-target cross-entropy (NLL): -sum_c y_c * log(clip(p_c, eps, 1)).

    Args:
        p_pred: Predicted probabilities of shape (N, 3).
        y_true: Soft probability targets of shape (N, 3).
        eps: Lower clipping bound for predicted probabilities.

    Returns:
        Tuple of (mean_nll, per_sample_nll) where per_sample_nll has shape (N,).
    """
    p = _as_prob_matrix(p_pred, "p_pred")
    y = _as_prob_matrix(y_true, "y_true")
    if p.shape != y.shape:
        raise ValueError(f"Shape mismatch: p_pred {p.shape} vs y_true {y.shape}")
    if len(p) == 0:
        return 0.0, np.empty((0,), dtype=np.float64)

    p_clipped = np.clip(p, eps, 1.0)
    per_sample = -np.sum(y * np.log(p_clipped), axis=-1)
    return float(np.mean(per_sample)), per_sample


def compute_brier_per_sample(p_pred: Any, y_true: Any) -> np.ndarray:
    """Compute per-sample multiclass Brier score: sum_c (p_c - y_c)^2."""
    p = _as_prob_matrix(p_pred, "p_pred")
    y = _as_prob_matrix(y_true, "y_true")
    if p.shape != y.shape:
        raise ValueError(f"Shape mismatch: p_pred {p.shape} vs y_true {y.shape}")
    if len(p) == 0:
        return np.empty((0,), dtype=np.float64)
    return np.sum((p - y) ** 2, axis=-1)


def compute_brier(p_pred: Any, y_true: Any) -> float:
    """Compute mean multiclass Brier score: mean(sum((p_pred - y_true)^2, axis=-1))."""
    per_sample = compute_brier_per_sample(p_pred, y_true)
    if len(per_sample) == 0:
        return 0.0
    return float(np.mean(per_sample))


def compute_l1_tv(p_pred: Any, y_true: Any) -> dict[str, float]:
    """Compute mean Total Variation distance and L1 error.

    TV = 0.5 * sum(|p_pred - y_true|, axis=-1), L1 = sum(|p_pred - y_true|, axis=-1).
    """
    p = _as_prob_matrix(p_pred, "p_pred")
    y = _as_prob_matrix(y_true, "y_true")
    if p.shape != y.shape:
        raise ValueError(f"Shape mismatch: p_pred {p.shape} vs y_true {y.shape}")
    if len(p) == 0:
        return {"tv": 0.0, "l1": 0.0}

    l1_per_sample = np.sum(np.abs(p - y), axis=-1)
    tv_per_sample = 0.5 * l1_per_sample
    return {
        "tv": float(np.mean(tv_per_sample)),
        "l1": float(np.mean(l1_per_sample)),
    }


def compute_argmax_accuracy(p_pred: Any, y_true: Any) -> dict[str, Any]:
    """Compute argmax classification accuracy plus predicted and true class counts."""
    p = _as_prob_matrix(p_pred, "p_pred")
    y = _as_prob_matrix(y_true, "y_true")
    if p.shape != y.shape:
        raise ValueError(f"Shape mismatch: p_pred {p.shape} vs y_true {y.shape}")

    n = len(p)
    if n == 0:
        empty_counts = {name: 0 for name in CLASS_NAMES}
        return {
            "accuracy": 0.0,
            "n_samples": 0,
            "pred_counts": dict(empty_counts),
            "true_counts": dict(empty_counts),
            "per_class_accuracy": {name: None for name in CLASS_NAMES},
        }

    pred_cls = np.argmax(p, axis=-1)
    true_cls = np.argmax(y, axis=-1)

    pred_counts = {
        name: int(np.sum(pred_cls == idx)) for idx, name in enumerate(CLASS_NAMES)
    }
    true_counts = {
        name: int(np.sum(true_cls == idx)) for idx, name in enumerate(CLASS_NAMES)
    }
    per_class_acc: dict[str, Optional[float]] = {}
    for idx, name in enumerate(CLASS_NAMES):
        mask = true_cls == idx
        if np.any(mask):
            per_class_acc[name] = float(np.mean(pred_cls[mask] == idx))
        else:
            per_class_acc[name] = None

    return {
        "accuracy": float(np.mean(pred_cls == true_cls)),
        "n_samples": int(n),
        "pred_counts": pred_counts,
        "true_counts": true_counts,
        "per_class_accuracy": per_class_acc,
    }


def _compute_ece_1d(confidences: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error for confidence vs expected correctness."""
    n = len(confidences)
    if n == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences > lo) & (confidences <= hi)
        count = int(np.sum(mask))
        if count > 0:
            avg_conf = float(np.mean(confidences[mask]))
            avg_acc = float(np.mean(outcomes[mask]))
            ece += (count / n) * abs(avg_acc - avg_conf)
    return float(ece)


def compute_ece(
    p_pred: Any,
    y_true: Any,
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute top-label confidence ECE and classwise ECE for soft probability targets.

    For soft targets y_true, the expected accuracy on sample i with predicted class c*
    is y_true[i, c*]. Classwise ECE averages 1D calibration error across W, D, L.
    """
    p = _as_prob_matrix(p_pred, "p_pred")
    y = _as_prob_matrix(y_true, "y_true")
    if p.shape != y.shape:
        raise ValueError(f"Shape mismatch: p_pred {p.shape} vs y_true {y.shape}")
    if len(p) == 0:
        return {"ece": 0.0, "classwise_ece": 0.0}

    pred_cls = np.argmax(p, axis=-1)
    top_conf = p[np.arange(len(p)), pred_cls]
    top_target = y[np.arange(len(p)), pred_cls]
    top_ece = _compute_ece_1d(top_conf, top_target, n_bins=n_bins)

    cls_eces = [
        _compute_ece_1d(p[:, c], y[:, c], n_bins=n_bins) for c in range(3)
    ]
    return {
        "ece": float(top_ece),
        "classwise_ece": float(np.mean(cls_eces)),
    }


def _side_summary(
    p: np.ndarray,
    y: np.ndarray,
    n_bins: int,
    eps: float,
) -> dict[str, Any]:
    """Compute calibration and scoring metrics on a slice of samples."""
    n = len(p)
    if n == 0:
        return {
            "n_samples": 0,
            "nll": None,
            "brier": None,
            "ece": None,
            "classwise_ece": None,
            "mean_pred": None,
            "mean_true": None,
        }
    nll_mean, _ = compute_nll(p, y, eps=eps)
    brier_mean = compute_brier(p, y)
    ece_dict = compute_ece(p, y, n_bins=n_bins)
    return {
        "n_samples": int(n),
        "nll": float(nll_mean),
        "brier": float(brier_mean),
        "ece": float(ece_dict["ece"]),
        "classwise_ece": float(ece_dict["classwise_ece"]),
        "mean_pred": [float(v) for v in np.mean(p, axis=0)],
        "mean_true": [float(v) for v in np.mean(y, axis=0)],
    }


def compute_side_calibration(
    p_pred: Any,
    y_true: Any,
    sides: Sequence[Any],
    n_bins: int = 10,
    eps: float = 1e-7,
) -> dict[str, Any]:
    """Compute separate NLL, Brier, ECE, and mean predicted W/D/L for White vs Black to move."""
    p = _as_prob_matrix(p_pred, "p_pred")
    y = _as_prob_matrix(y_true, "y_true")
    if p.shape != y.shape:
        raise ValueError(f"Shape mismatch: p_pred {p.shape} vs y_true {y.shape}")
    if len(sides) != len(p):
        raise ValueError(
            f"Length mismatch: sides ({len(sides)}) vs p_pred ({len(p)})"
        )

    norm_sides = np.array([_normalize_side(s) for s in sides], dtype=object)
    white_mask = norm_sides == "white"
    black_mask = norm_sides == "black"

    white_res = _side_summary(p[white_mask], y[white_mask], n_bins=n_bins, eps=eps)
    black_res = _side_summary(p[black_mask], y[black_mask], n_bins=n_bins, eps=eps)
    overall_res = _side_summary(p, y, n_bins=n_bins, eps=eps)

    nll_gap = None
    if white_res["nll"] is not None and black_res["nll"] is not None:
        nll_gap = abs(white_res["nll"] - black_res["nll"])

    return {
        "white": white_res,
        "black": black_res,
        "overall": overall_res,
        "side_nll_gap": nll_gap,
    }


def compute_context_gain(
    nll_k0_per_root: Union[Sequence[float], np.ndarray, Mapping[str, Sequence[float]]],
    nll_k_per_root: Union[Sequence[float], np.ndarray, Mapping[str, Sequence[float]]],
    variants: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Compute Context Gain: Delta NLL = NLL(K=0) - NLL(K) on identical roots.

    Positive values indicate that K demonstrations reduced NLL relative to K=0.

    Supports either:
      - 1D arrays of per-root NLLs (optionally paired with `variants` labels of same length), or
      - Mappings `{variant_name: per_root_nll_sequence}` with identical variant keys and lengths.
    """
    if isinstance(nll_k0_per_root, Mapping) or isinstance(nll_k_per_root, Mapping):
        if not (isinstance(nll_k0_per_root, Mapping) and isinstance(nll_k_per_root, Mapping)):
            raise TypeError("Both inputs must be mappings if either is a mapping")
        if set(nll_k0_per_root.keys()) != set(nll_k_per_root.keys()):
            raise ValueError("Variant keys mismatch between nll_k0_per_root and nll_k_per_root")
        k0_parts: list[np.ndarray] = []
        k_parts: list[np.ndarray] = []
        var_labels: list[str] = []
        for v in sorted(nll_k0_per_root.keys()):
            a0 = np.asarray(nll_k0_per_root[v], dtype=np.float64)
            ak = np.asarray(nll_k_per_root[v], dtype=np.float64)
            if a0.shape != ak.shape or a0.ndim != 1:
                raise ValueError(f"Variant {v!r} arrays must be 1D with equal shape")
            k0_parts.append(a0)
            k_parts.append(ak)
            var_labels.extend([v] * len(a0))
        k0 = np.concatenate(k0_parts) if k0_parts else np.empty((0,), dtype=np.float64)
        kk = np.concatenate(k_parts) if k_parts else np.empty((0,), dtype=np.float64)
        variants_arr = np.array(var_labels, dtype=object) if var_labels else None
    else:
        k0 = np.asarray(nll_k0_per_root, dtype=np.float64)
        kk = np.asarray(nll_k_per_root, dtype=np.float64)
        if k0.ndim != 1 or kk.ndim != 1 or k0.shape != kk.shape:
            raise ValueError(
                f"nll_k0_per_root {k0.shape} and nll_k_per_root {kk.shape} must be 1D with identical shape"
            )
        if variants is not None:
            if len(variants) != len(k0):
                raise ValueError(
                    f"variants length ({len(variants)}) must match per-root arrays ({len(k0)})"
                )
            variants_arr = np.asarray(list(variants), dtype=object)
        else:
            variants_arr = None

    if len(k0) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "se": 0.0,
            "macro_mean": 0.0,
            "macro_std": 0.0,
            "per_variant": {},
            "per_root_gain": np.empty((0,), dtype=np.float64),
            "n_roots": 0,
        }

    gains = k0 - kk
    n = len(gains)
    mean_gain = float(np.mean(gains))
    std_gain = float(np.std(gains, ddof=1)) if n > 1 else 0.0
    se_gain = float(std_gain / np.sqrt(n)) if n > 1 else 0.0

    per_variant: dict[str, dict[str, Any]] = {}
    if variants_arr is not None:
        unique_variants = sorted({str(v) for v in variants_arr})
        for v in unique_variants:
            mask = variants_arr == v
            vg = gains[mask]
            nv = len(vg)
            v_mean = float(np.mean(vg))
            v_std = float(np.std(vg, ddof=1)) if nv > 1 else 0.0
            per_variant[v] = {
                "mean": v_mean,
                "std": v_std,
                "se": float(v_std / np.sqrt(nv)) if nv > 1 else 0.0,
                "n_roots": int(nv),
                "nll_k0": float(np.mean(k0[mask])),
                "nll_k": float(np.mean(kk[mask])),
            }
        var_means = np.array([per_variant[v]["mean"] for v in unique_variants], dtype=np.float64)
        macro_mean = float(np.mean(var_means))
        macro_std = float(np.std(var_means, ddof=1)) if len(var_means) > 1 else 0.0
    else:
        macro_mean = mean_gain
        macro_std = 0.0

    return {
        "mean": mean_gain,
        "std": std_gain,
        "se": se_gain,
        "macro_mean": macro_mean,
        "macro_std": macro_std,
        "per_variant": per_variant,
        "per_root_gain": gains,
        "n_roots": int(n),
    }


def compute_mc_noise_floor(y_true: Any, R: int) -> dict[str, float]:
    """Compute theoretical irreducible entropy and finite-sample Monte Carlo variance from R rollouts.

    Let y_true[i] be the empirical W/D/L distribution from R i.i.d. rollouts.
    Returns:
      - entropy_floor: mean Shannon entropy H(y_true) in nats.
      - debiased_entropy_floor: H(y_true) + mean((k_eff - 1) / (2 * R)), Miller-Madow
        correction for finite-sample bias of plug-in entropy.
      - mc_nll_bias: mean((k_eff - 1) / (2 * R)), expected KL(p || y_hat) bias.
      - mc_nll_variance_per_root: mean_i [(sum_c y_{i,c} (log y_{i,c})^2 - H(y_i)^2) / R],
        expected rollout variance of NLL per root.
      - mc_brier_floor: mean_i (1 - sum_c y_{i,c}^2), Bayes single-rollout Brier floor.
      - mc_brier_variance: mean_i ((1 - sum_c y_{i,c}^2) / R), expected L2 variance of
        the R-rollout empirical target around the true distribution.
      - R: rollout count.
    """
    if R < 1:
        raise ValueError(f"R must be >= 1, got {R}")
    y = _as_prob_matrix(y_true, "y_true")
    if len(y) == 0:
        return {
            "entropy_floor": 0.0,
            "debiased_entropy_floor": 0.0,
            "mc_nll_bias": 0.0,
            "mc_nll_variance_per_root": 0.0,
            "mc_brier_floor": 0.0,
            "mc_brier_variance": 0.0,
            "R": int(R),
        }

    # Exact 0 * log(0) = 0 handling
    safe_y = np.where(y > 0.0, y, 1.0)
    log_y = np.where(y > 0.0, np.log(safe_y), 0.0)
    entropies = -np.sum(y * log_y, axis=-1)
    entropy_floor = float(np.mean(entropies))

    # Effective support size k_eff per root (number of classes with y > 0)
    k_eff = np.sum(y > 0.0, axis=-1)
    mc_nll_bias = float(np.mean(np.maximum(k_eff - 1, 0) / (2.0 * R)))
    debiased_entropy_floor = entropy_floor + mc_nll_bias

    # Variance of -sum_c y_hat_c log(p_c) when y_hat ~ Multinomial(R, p)/R at p = y_true
    e_log2 = np.sum(y * (log_y ** 2), axis=-1)
    nll_var_per_root = np.maximum(e_log2 - entropies ** 2, 0.0) / float(R)
    mc_nll_variance = float(np.mean(nll_var_per_root))

    # Brier floor & multinomial target variance: sum_c p_c (1 - p_c) / R = (1 - sum_c p_c^2) / R
    gini = np.maximum(1.0 - np.sum(y ** 2, axis=-1), 0.0)
    mc_brier_floor = float(np.mean(gini))
    mc_brier_variance = float(np.mean(gini / float(R)))

    return {
        "entropy_floor": entropy_floor,
        "debiased_entropy_floor": debiased_entropy_floor,
        "mc_nll_bias": mc_nll_bias,
        "mc_nll_variance_per_root": mc_nll_variance,
        "mc_brier_floor": mc_brier_floor,
        "mc_brier_variance": mc_brier_variance,
        "R": int(R),
    }
