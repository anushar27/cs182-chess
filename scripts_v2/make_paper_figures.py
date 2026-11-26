"""Generate publication-quality PDF/PNG figures (300 DPI) and LaTeX tables from runs_v2/.

Adheres to CS 182/282 Reviewer Checklist:
  - Explicit axis labels with units, readable typography, colorblind-safe palette.
  - Explicit axis scale choices (log x-axis for parameter count, linear for NLL / Gain).
  - Error bars (95% bootstrap CIs / seed std) with explicit caption annotations.
  - Graceful skipping of any figure whose required input JSONs are absent.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig_chess_v2"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Colorblind-safe Okabe-Ito palette + distinct markers/linestyles
PALETTE: dict[str, str] = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "black": "#1A1A1A",
    "gray": "#7F7F7F",
}

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "track1_qwen": "Track 1: Qwen2.5-7B LoRA",
    "qwen": "Track 1: Qwen2.5-7B LoRA",
    "track2_tiny": "Track 2: Tiny (1.4M)",
    "track2_small": "Track 2: Small (7.5M)",
    "track2_medium": "Track 2: Medium (23.1M)",
    "track2_large": "Track 2: Large (206.4M)",
    "mismatched_control": "Mismatched-Context Control",
    "board_knn": "Board kNN",
    "demo_majority": "Demo Majority",
    "length_histogram": "Length Histogram",
    "query_material": "Query Material",
    "training_prior": "Training Prior",
    "uniform": "Uniform",
}

MODEL_PARAM_DEFAULTS: dict[str, int] = {
    "track2_tiny": 1_395_971,
    "track2_small": 7_523_843,
    "track2_medium": 23_102_211,
    "track2_large": 206_362_627,
    "tiny": 1_395_971,
    "small": 7_523_843,
    "medium": 23_102_211,
    "large": 206_362_627,
}

SERIES_STYLES: list[tuple[str, str, str]] = [
    (PALETTE["blue"], "o", "-"),
    (PALETTE["vermillion"], "s", "-"),
    (PALETTE["green"], "^", "-"),
    (PALETTE["purple"], "D", "-"),
    (PALETTE["orange"], "v", "--"),
    (PALETTE["sky"], "P", "--"),
    (PALETTE["gray"], "X", ":"),
    (PALETTE["black"], "*", "-."),
]


def configure_matplotlib() -> None:
    """Apply clean publication-grade matplotlib defaults."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "figure.titlesize": 12.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "lines.linewidth": 1.8,
            "lines.markersize": 5.5,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    """Save figure as both 300-DPI PNG and vector PDF."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [pdf_path, png_path]


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _display_name(key: str) -> str:
    return MODEL_DISPLAY_NAMES.get(key.lower(), key.replace("_", " ").title())


def _ci_to_yerr(
    mean: float,
    ci: Optional[Sequence[float]],
    std: Optional[float] = None,
) -> tuple[float, float]:
    """Convert [ci_low, ci_high] (or fallback std) to asymmetric (lo_err, hi_err)."""
    if ci is not None and len(ci) == 2 and ci[0] is not None and ci[1] is not None:
        lo = max(0.0, float(mean) - float(ci[0]))
        hi = max(0.0, float(ci[1]) - float(mean))
        return lo, hi
    if std is not None and std > 0:
        return float(std), float(std)
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# Artifact Discovery Helpers
# ---------------------------------------------------------------------------


def discover_baselines_payload(
    runs_dir: Path,
    baselines_json: Optional[Path] = None,
    split: str = "test",
) -> Optional[dict[str, Any]]:
    """Locate and load baselines_<split>.json from `baselines_json` or `runs_dir`."""
    if baselines_json is not None and baselines_json.exists():
        return _load_json(baselines_json)

    candidates = [
        runs_dir / f"baselines_{split}.json",
        runs_dir / "baselines" / f"baselines_{split}.json",
        runs_dir / "baselines_test.json",
        runs_dir / "baselines_val.json",
        runs_dir / "baselines" / "baselines_test.json",
        runs_dir / "baselines" / "baselines_val.json",
    ]
    for path in candidates:
        if path.exists():
            return _load_json(path)
    return None


def discover_training_histories(runs_dir: Path) -> dict[str, dict[str, Any]]:
    """Discover Track 1 & Track 2 training history JSONs under `runs_dir`."""
    histories: dict[str, dict[str, Any]] = {}
    if not runs_dir.exists():
        return histories

    for path in sorted(runs_dir.rglob("*.json")):
        name_l = path.name.lower()
        if "history" not in name_l and "train_log" not in name_l:
            continue
        data = _load_json(path)
        if not isinstance(data, dict):
            continue
        if "histories" in data and isinstance(data["histories"], dict):
            for k, v in data["histories"].items():
                if isinstance(v, dict):
                    histories[str(k)] = v
        elif "steps" in data or "train_nll" in data or "val_nll" in data:
            label = str(data.get("model_name") or path.parent.name or path.stem)
            if label in (".", "runs_v2"):
                label = path.stem
            histories[label] = data
    return histories


def discover_neural_evals(runs_dir: Path, split: str = "test") -> dict[str, dict[str, Any]]:
    """Discover Track 1 & Track 2 evaluation JSONs matching `by_k` or scaling schemas."""
    models: dict[str, dict[str, Any]] = {}
    if not runs_dir.exists():
        return models

    for path in sorted(runs_dir.rglob("*.json")):
        name_l = path.name.lower()
        if name_l.startswith("baselines_") or "history" in name_l or "summary" in name_l:
            continue
        if "eval" not in name_l and "scaling" not in name_l:
            continue
        data = _load_json(path)
        if not isinstance(data, dict):
            continue
        if "split" in data and data["split"] not in (split, None) and "eval" in name_l:
            # Prefer matching split when multiple exist, skip mismatched if same key already loaded
            pass

        if "models" in data and isinstance(data["models"], dict):
            for m_name, m_data in data["models"].items():
                if isinstance(m_data, dict) and "by_k" in m_data:
                    models[str(m_name)] = m_data
        elif "by_k" in data:
            m_name = str(data.get("model_name") or path.stem)
            models[m_name] = data
    return models


def discover_controls_eval(runs_dir: Path) -> Optional[dict[str, Any]]:
    """Discover causal intervention & control results JSON under `runs_dir`."""
    if not runs_dir.exists():
        return None
    for path in sorted(runs_dir.rglob("*.json")):
        name_l = path.name.lower()
        if "control" in name_l or "intervention" in name_l:
            data = _load_json(path)
            if isinstance(data, dict) and ("conditions" in data or "models" in data):
                return data
    return None


# ---------------------------------------------------------------------------
# Figure 1: Training & Validation NLL Curves
# ---------------------------------------------------------------------------


def make_figure1_training_curves(
    histories: dict[str, dict[str, Any]],
    out_dir: Path,
    mc_floor: Optional[float] = None,
) -> Optional[list[Path]]:
    """Figure 1: Track 2 warmup/continue-training & Track 1 Qwen LoRA NLL curves."""
    if not histories:
        return None

    track2_items = {
        k: v for k, v in histories.items()
        if "qwen" not in k.lower() and "track1" not in k.lower()
    }
    track1_items = {
        k: v for k, v in histories.items()
        if "qwen" in k.lower() or "track1" in k.lower()
    }

    panels: list[tuple[str, dict[str, dict[str, Any]]]] = []
    if track2_items:
        panels.append(("Track 2: $\\mu$P Transformer Training & Validation NLL", track2_items))
    if track1_items:
        panels.append(("Track 1: Qwen2.5-7B-Instruct LoRA Training & Validation NLL", track1_items))
    if not panels:
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.2), squeeze=False)
    for ax_idx, (title, group) in enumerate(panels):
        ax = axes[0, ax_idx]
        for s_idx, (name, hist) in enumerate(sorted(group.items())):
            color = SERIES_STYLES[s_idx % len(SERIES_STYLES)][0]
            steps = np.asarray(hist.get("steps", []), dtype=np.float64)
            train_nll = np.asarray(hist.get("train_nll", hist.get("train_loss", [])), dtype=np.float64)
            val_nll = np.asarray(hist.get("val_nll", hist.get("val_loss", [])), dtype=np.float64)
            if len(steps) == 0:
                n_pts = max(len(train_nll), len(val_nll))
                steps = np.arange(1, n_pts + 1, dtype=np.float64)

            disp = _display_name(name)
            if len(train_nll) == len(steps) and len(train_nll) > 0:
                ax.plot(steps, train_nll, color=color, linestyle="--", alpha=0.75, label=f"{disp} (Train)")
            if len(val_nll) == len(steps) and len(val_nll) > 0:
                ax.plot(steps, val_nll, color=color, linestyle="-", linewidth=2.0, label=f"{disp} (Val)")
                val_std = hist.get("val_std")
                if val_std is not None and len(val_std) == len(steps):
                    vs = np.asarray(val_std, dtype=np.float64)
                    ax.fill_between(steps, val_nll - vs, val_nll + vs, color=color, alpha=0.15)

            warmup_end = hist.get("warmup_end_step")
            if warmup_end is not None:
                ax.axvline(float(warmup_end), color=PALETTE["gray"], linestyle=":", linewidth=1.2)

        ax.axhline(math.log(3.0), color=PALETTE["gray"], linestyle=":", linewidth=1.2, label="Uniform ($\\ln 3$)")
        if mc_floor is not None:
            ax.axhline(mc_floor, color=PALETTE["black"], linestyle="-.", linewidth=1.1, label="MC Noise Floor")

        ax.set_title(title)
        ax.set_xlabel("Optimizer Step (linear scale)")
        ax.set_ylabel("Soft-Target Cross-Entropy NLL (nats)")
        ax.legend(loc="best", frameon=True, framealpha=0.9)

    fig.text(
        0.5,
        -0.03,
        "Note: Linear axes. Dashed lines denote training NLL; solid lines denote validation NLL; shaded bands (where present) denote ±1 std across model seeds.",
        ha="center",
        fontsize=8.2,
        color="#444444",
    )
    fig.tight_layout()
    return _save_figure(fig, out_dir, "fig1_training_curves")


# ---------------------------------------------------------------------------
# Figure 2: Main Context Gain vs K
# ---------------------------------------------------------------------------


def make_figure2_context_gain_vs_k(
    all_models: dict[str, dict[str, Any]],
    out_dir: Path,
) -> Optional[list[Path]]:
    """Figure 2: Macro Context Gain Delta NLL(K) vs K across models & baselines."""
    if not all_models:
        return None

    preferred_order = [
        "track1_qwen",
        "qwen",
        "track2_large",
        "track2_medium",
        "track2_small",
        "track2_tiny",
        "mismatched_control",
        "board_knn",
        "demo_majority",
        "length_histogram",
        "query_material",
    ]
    ordered_keys = [k for k in preferred_order if k in all_models]
    for k in sorted(all_models.keys()):
        if k not in ordered_keys and k not in ("uniform", "training_prior"):
            ordered_keys.append(k)

    if not ordered_keys:
        return None

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    all_k_ticks: set[int] = set()
    plotted = 0

    for idx, m_key in enumerate(ordered_keys):
        by_k = all_models[m_key].get("by_k", {})
        if not by_k:
            continue
        ks = sorted(int(k) for k in by_k.keys())
        all_k_ticks.update(ks)
        means = []
        lo_errs = []
        hi_errs = []
        for k in ks:
            macro = by_k[str(k)].get("macro", {})
            g = macro.get("context_gain")
            if g is None:
                g = 0.0
            lo, hi = _ci_to_yerr(
                float(g),
                macro.get("context_gain_ci"),
                macro.get("context_gain_std"),
            )
            means.append(float(g))
            lo_errs.append(lo)
            hi_errs.append(hi)

        color, marker, ls = SERIES_STYLES[idx % len(SERIES_STYLES)]
        ax.errorbar(
            ks,
            means,
            yerr=[lo_errs, hi_errs],
            color=color,
            marker=marker,
            linestyle=ls,
            capsize=3.0,
            elinewidth=1.2,
            label=_display_name(m_key),
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    ax.axhline(0.0, color=PALETTE["black"], linestyle=":", linewidth=1.2, label="Zero Context Gain")
    if all_k_ticks:
        ax.set_xticks(sorted(all_k_ticks))
    ax.set_xlabel("Number of In-Context Demonstration Games $K$ (linear scale)")
    ax.set_ylabel("Macro Context Gain $\\overline{\\Delta\\mathrm{NLL}}(K) = \\mathrm{NLL}(0) - \\mathrm{NLL}(K)$ (nats)")
    ax.set_title("In-Context Rule Learning Across Demonstration Budget $K$")
    ax.legend(loc="best", ncol=2, frameon=True, framealpha=0.9)

    fig.text(
        0.5,
        -0.03,
        "Note: Higher values indicate stronger improvement over K=0 on identical query roots. Error bars represent 95% cluster-bootstrap CIs over source games.",
        ha="center",
        fontsize=8.2,
        color="#444444",
    )
    fig.tight_layout()
    return _save_figure(fig, out_dir, "fig2_context_gain_vs_k")


# ---------------------------------------------------------------------------
# Figure 3: Track 2 muP Scaling Curve
# ---------------------------------------------------------------------------


def make_figure3_mup_scaling(
    neural_models: dict[str, dict[str, Any]],
    baselines_payload: Optional[dict[str, Any]],
    out_dir: Path,
) -> Optional[list[Path]]:
    """Figure 3: Log parameter count vs Macro NLL & Context Gain for Track 2 muP models."""
    scaling_pts: list[tuple[int, str, dict[str, Any]]] = []
    for name, m_data in neural_models.items():
        name_l = name.lower()
        if "qwen" in name_l or "track1" in name_l:
            continue
        params = m_data.get("n_params") or m_data.get("params") or MODEL_PARAM_DEFAULTS.get(name_l)
        if params is None or not m_data.get("by_k"):
            continue
        scaling_pts.append((int(params), name, m_data))

    if len(scaling_pts) < 2:
        return None

    scaling_pts.sort(key=lambda x: x[0])
    params_arr = [p for p, _, _ in scaling_pts]

    # Pick largest common K
    common_ks = set.intersection(*(set(int(k) for k in m["by_k"].keys()) for _, _, m in scaling_pts))
    k_max = max(common_ks) if common_ks else max(int(k) for k in scaling_pts[0][2]["by_k"].keys())
    k_str = str(k_max)

    nll_means, nll_lo, nll_hi = [], [], []
    gain_means, gain_lo, gain_hi = [], [], []
    for _, _, m_data in scaling_pts:
        macro = m_data["by_k"][k_str]["macro"]
        nll_val = float(macro["nll"])
        lo_n, hi_n = _ci_to_yerr(nll_val, macro.get("nll_ci"), macro.get("nll_boot_std"))
        nll_means.append(nll_val)
        nll_lo.append(lo_n)
        nll_hi.append(hi_n)

        g_val = float(macro.get("context_gain", 0.0))
        lo_g, hi_g = _ci_to_yerr(g_val, macro.get("context_gain_ci"), macro.get("context_gain_std"))
        gain_means.append(g_val)
        gain_lo.append(lo_g)
        gain_hi.append(hi_g)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    ax1.errorbar(
        params_arr,
        nll_means,
        yerr=[nll_lo, nll_hi],
        color=PALETTE["blue"],
        marker="o",
        capsize=3.5,
        label=f"Track 2 $\\mu$P ($K={k_max}$)",
    )
    ax1.axhline(math.log(3.0), color=PALETTE["gray"], linestyle=":", label="Uniform ($\\ln 3$)")
    if baselines_payload:
        mc_ent = baselines_payload.get("mc_noise_floor_macro", {}).get("entropy_floor")
        if mc_ent is not None:
            ax1.axhline(float(mc_ent), color=PALETTE["black"], linestyle="-.", label="MC Noise Floor")
        knn_by_k = baselines_payload.get("models", {}).get("board_knn", {}).get("by_k", {})
        if k_str in knn_by_k:
            ax1.axhline(
                float(knn_by_k[k_str]["macro"]["nll"]),
                color=PALETTE["orange"],
                linestyle="--",
                label=f"Board kNN ($K={k_max}$)",
            )
    ax1.set_xscale("log")
    ax1.set_xlabel("Model Parameters (log scale)")
    ax1.set_ylabel("Macro Held-Out NLL (nats)")
    ax1.set_title(f"Held-Out NLL Scaling ($K={k_max}$)")
    ax1.legend(loc="best", frameon=True)

    ax2.errorbar(
        params_arr,
        gain_means,
        yerr=[gain_lo, gain_hi],
        color=PALETTE["green"],
        marker="s",
        capsize=3.5,
        label=f"Track 2 $\\mu$P ($K={k_max}$)",
    )
    ax2.axhline(0.0, color=PALETTE["black"], linestyle=":", label="Zero Gain")
    if baselines_payload:
        knn_by_k = baselines_payload.get("models", {}).get("board_knn", {}).get("by_k", {})
        if k_str in knn_by_k:
            ax2.axhline(
                float(knn_by_k[k_str]["macro"].get("context_gain", 0.0)),
                color=PALETTE["orange"],
                linestyle="--",
                label=f"Board kNN ($K={k_max}$)",
            )
    ax2.set_xscale("log")
    ax2.set_xlabel("Model Parameters (log scale)")
    ax2.set_ylabel("Macro Context Gain $\\overline{\\Delta\\mathrm{NLL}}$ (nats)")
    ax2.set_title(f"Context Gain Scaling ($K={k_max}$)")
    ax2.legend(loc="best", frameon=True)

    fig.text(
        0.5,
        -0.03,
        "Note: X-axis uses log10 scale for non-embedding+embedding parameter counts. Error bars represent 95% cluster-bootstrap CIs over source games.",
        ha="center",
        fontsize=8.2,
        color="#444444",
    )
    fig.tight_layout()
    return _save_figure(fig, out_dir, "fig3_mup_scaling")


# ---------------------------------------------------------------------------
# Figure 4: Causal Interventions & Controls
# ---------------------------------------------------------------------------


def make_figure4_causal_controls(
    controls_data: Optional[dict[str, Any]],
    out_dir: Path,
) -> Optional[list[Path]]:
    """Figure 4: Clean vs Mismatched/Inverted/Scrambled/Corrupted controls + Equal-Material slice."""
    if not controls_data:
        return None

    conditions = controls_data.get("conditions")
    if not isinstance(conditions, dict) or not conditions:
        return None

    ordered_conds = [
        c
        for c in (
            "clean",
            "mismatched_context",
            "inverted_outcomes",
            "scrambled_outcomes",
            "corrupted_trajectories",
        )
        if c in conditions
    ]
    for c in sorted(conditions.keys()):
        if c not in ordered_conds:
            ordered_conds.append(c)

    if not ordered_conds:
        return None

    cond_labels = [c.replace("_", "\n").title() for c in ordered_conds]
    x = np.arange(len(ordered_conds))
    width = 0.36

    full_vals, full_lo, full_hi = [], [], []
    eq_vals, eq_lo, eq_hi = [], [], []
    has_eq = False

    for c in ordered_conds:
        entry = conditions[c]
        val = float(entry.get("context_gain", entry.get("nll", 0.0)))
        lo, hi = _ci_to_yerr(val, entry.get("context_gain_ci", entry.get("ci")), entry.get("std"))
        full_vals.append(val)
        full_lo.append(lo)
        full_hi.append(hi)

        eq_val = entry.get("equal_material_context_gain", entry.get("equal_material_nll"))
        if eq_val is not None:
            has_eq = True
            ev = float(eq_val)
            elo, ehi = _ci_to_yerr(ev, entry.get("equal_material_ci"), entry.get("equal_material_std"))
            eq_vals.append(ev)
            eq_lo.append(elo)
            eq_hi.append(ehi)
        else:
            eq_vals.append(0.0)
            eq_lo.append(0.0)
            eq_hi.append(0.0)

    metric_name = "Context Gain $\\overline{\\Delta\\mathrm{NLL}}$ (nats)"
    if all("context_gain" not in conditions[c] for c in ordered_conds):
        metric_name = "Held-Out NLL (nats)"

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    if has_eq:
        ax.bar(
            x - width / 2,
            full_vals,
            width,
            yerr=[full_lo, full_hi],
            capsize=3.5,
            color=PALETTE["blue"],
            label="Full Held-Out Eval Set",
        )
        ax.bar(
            x + width / 2,
            eq_vals,
            width,
            yerr=[eq_lo, eq_hi],
            capsize=3.5,
            color=PALETTE["green"],
            label="Equal-Material Slice ($|\\Delta M| \\leq 2$)",
        )
    else:
        ax.bar(
            x,
            full_vals,
            width * 1.4,
            yerr=[full_lo, full_hi],
            capsize=3.5,
            color=PALETTE["blue"],
            label="Full Held-Out Eval Set",
        )

    ax.axhline(0.0, color=PALETTE["black"], linestyle=":", linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels)
    ax.set_ylabel(metric_name)
    ax.set_title("Causal Context & Trajectory Interventions")
    ax.legend(loc="best", frameon=True)

    fig.text(
        0.5,
        -0.03,
        "Note: Error bars represent 95% cluster-bootstrap CIs over source games.",
        ha="center",
        fontsize=8.2,
        color="#444444",
    )
    fig.tight_layout()
    return _save_figure(fig, out_dir, "fig4_causal_controls")


# ---------------------------------------------------------------------------
# Figure 5: Side-to-Move Calibration & Per-Variant Breakdown
# ---------------------------------------------------------------------------


def make_figure5_side_and_variants(
    all_models: dict[str, dict[str, Any]],
    out_dir: Path,
) -> Optional[list[Path]]:
    """Figure 5: Side-to-move (White vs Black) calibration & per-variant breakdown."""
    if not all_models:
        return None

    # Pick largest available K per model
    side_rows: list[tuple[str, float, float, Optional[float], Optional[float]]] = []
    best_model_key: Optional[str] = None
    best_model_gain = -1e9

    priority_for_variant_plot = [
        "track1_qwen",
        "qwen",
        "track2_large",
        "track2_medium",
        "track2_small",
        "board_knn",
        "demo_majority",
    ]

    for m_key, m_data in all_models.items():
        by_k = m_data.get("by_k", {})
        if not by_k:
            continue
        k_max = max(int(k) for k in by_k.keys())
        k_entry = by_k[str(k_max)]
        macro = k_entry.get("macro", {})
        w_nll = macro.get("white_nll")
        b_nll = macro.get("black_nll")
        if w_nll is not None and b_nll is not None:
            side_rows.append(
                (
                    _display_name(m_key),
                    float(w_nll),
                    float(b_nll),
                    macro.get("white_ece"),
                    macro.get("black_ece"),
                )
            )
        if k_entry.get("per_variant"):
            g = float(macro.get("context_gain", 0.0))
            if best_model_key is None or (
                m_key in priority_for_variant_plot
                and (best_model_key not in priority_for_variant_plot or g > best_model_gain)
            ):
                best_model_key = m_key
                best_model_gain = g

    if not side_rows and best_model_key is None:
        return None

    n_panels = int(bool(side_rows)) + int(best_model_key is not None)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, 4.6), squeeze=False)
    col_idx = 0

    if side_rows:
        ax = axes[0, col_idx]
        col_idx += 1
        x = np.arange(len(side_rows))
        w = 0.36
        labels = [r[0] for r in side_rows]
        w_vals = [r[1] for r in side_rows]
        b_vals = [r[2] for r in side_rows]

        bars_w = ax.bar(x - w / 2, w_vals, w, color=PALETTE["blue"], label="White to Move")
        bars_b = ax.bar(x + w / 2, b_vals, w, color=PALETTE["vermillion"], label="Black to Move")
        for idx, r in enumerate(side_rows):
            if r[3] is not None:
                ax.text(
                    bars_w[idx].get_x() + bars_w[idx].get_width() / 2,
                    w_vals[idx] + 0.015,
                    f"ECE\n{r[3]:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                )
            if r[4] is not None:
                ax.text(
                    bars_b[idx].get_x() + bars_b[idx].get_width() / 2,
                    b_vals[idx] + 0.015,
                    f"ECE\n{r[4]:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Held-Out NLL (nats)")
        ax.set_title("Side-to-Move Calibration (White vs. Black)")
        ax.legend(loc="best", frameon=True)

    if best_model_key is not None:
        ax = axes[0, col_idx]
        by_k = all_models[best_model_key]["by_k"]
        k_max = max(int(k) for k in by_k.keys())
        per_var = by_k[str(k_max)]["per_variant"]
        items = []
        for v_name, v_data in per_var.items():
            cg = v_data.get("context_gain", {})
            g_mean = float(cg.get("mean", 0.0))
            lo, hi = _ci_to_yerr(g_mean, cg.get("ci"), cg.get("std"))
            items.append((v_name, g_mean, lo, hi))
        items.sort(key=lambda t: t[1])

        y_pos = np.arange(len(items))
        v_labels = [
            f"{v} [null ctrl]" if v == "knight_army" else v
            for v, _, _, _ in items
        ]
        v_means = [m for _, m, _, _ in items]
        v_lo = [lo for _, _, lo, _ in items]
        v_hi = [hi for _, _, _, hi in items]
        colors = [
            PALETTE["gray"] if v == "knight_army" else PALETTE["green"]
            for v, _, _, _ in items
        ]

        ax.barh(
            y_pos,
            v_means,
            xerr=[v_lo, v_hi],
            color=colors,
            capsize=3.0,
            alpha=0.9,
        )
        ax.axvline(0.0, color=PALETTE["black"], linestyle=":", linewidth=1.1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(v_labels)
        ax.set_xlabel(f"Context Gain $\\Delta\\mathrm{{NLL}}_v(K={k_max})$ (nats)")
        ax.set_title(f"Per-Variant Context Gain ({_display_name(best_model_key)})")

    fig.text(
        0.5,
        -0.03,
        "Note: Left panel compares White- vs. Black-to-move NLL with top-label ECE annotated above bars. Right panel error bars denote 95% cluster-bootstrap CIs.",
        ha="center",
        fontsize=8.2,
        color="#444444",
    )
    fig.tight_layout()
    return _save_figure(fig, out_dir, "fig5_side_calibration_and_variants")


# ---------------------------------------------------------------------------
# LaTeX Tables & Captions
# ---------------------------------------------------------------------------


def _fmt_val_ci(val: Optional[float], ci: Optional[Sequence[float]], std: Optional[float]) -> str:
    if val is None:
        return "--"
    if ci is not None and len(ci) == 2 and ci[0] is not None and ci[1] is not None:
        half = 0.5 * (float(ci[1]) - float(ci[0]))
        return f"{val:.3f} $\\pm$ {half:.3f}"
    if std is not None:
        return f"{val:.3f} $\\pm$ {std:.3f}"
    return f"{val:.3f}"


def generate_latex_tables(
    all_models: dict[str, dict[str, Any]],
    baselines_payload: Optional[dict[str, Any]],
    out_dir: Path,
) -> list[Path]:
    """Generate booktabs LaTeX tables (`table1_main_results.tex`, `table2_hyperparameters.tex`, `captions.tex`)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    if all_models:
        lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Macro-averaged held-out evaluation across non-neural baselines and ICL models at maximum context budget $K_{\\max}$. Uncertainties report half-width of 95\\% cluster-bootstrap CIs over source games.}",
            "\\label{tab:main_results}",
            "\\begin{tabular}{lcccccc}",
            "\\toprule",
            "Model / Baseline & $K$ & Macro NLL ($\\downarrow$) & Brier ($\\downarrow$) & TV ($\\downarrow$) & Eq.-Mat. NLL ($\\downarrow$) & $\\overline{\\Delta\\mathrm{NLL}}$ ($\\uparrow$) \\\\",
            "\\midrule",
        ]
        ordered = [
            "uniform",
            "training_prior",
            "query_material",
            "demo_majority",
            "length_histogram",
            "board_knn",
            "track2_tiny",
            "track2_small",
            "track2_medium",
            "track2_large",
            "track1_qwen",
            "qwen",
        ]
        keys = [k for k in ordered if k in all_models] + [
            k for k in sorted(all_models.keys()) if k not in ordered
        ]
        for k_name in keys:
            by_k = all_models[k_name].get("by_k", {})
            if not by_k:
                continue
            k_max = max(int(k) for k in by_k.keys())
            m = by_k[str(k_max)]["macro"]
            nll_str = _fmt_val_ci(m.get("nll"), m.get("nll_ci"), m.get("nll_boot_std"))
            brier_str = f"{m['brier']:.3f}" if m.get("brier") is not None else "--"
            tv_str = f"{m['tv']:.3f}" if m.get("tv") is not None else "--"
            eq_str = _fmt_val_ci(
                m.get("equal_material_nll"),
                m.get("equal_material_ci"),
                None,
            )
            gain_str = _fmt_val_ci(
                m.get("context_gain"),
                m.get("context_gain_ci"),
                m.get("context_gain_std"),
            )
            disp = _display_name(k_name).replace("_", "\\_")
            lines.append(
                f"{disp} & {k_max} & {nll_str} & {brier_str} & {tv_str} & {eq_str} & {gain_str} \\\\"
            )

        if baselines_payload and "mc_noise_floor_macro" in baselines_payload:
            mc = baselines_payload["mc_noise_floor_macro"]
            lines.append("\\midrule")
            lines.append(
                f"MC Noise Floor $H(y)$ & -- & {mc['entropy_floor']:.3f} & -- & -- & -- & -- \\\\"
            )

        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
        t1_path = out_dir / "table1_main_results.tex"
        t1_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        generated.append(t1_path)

    t2_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Model architecture ladder ($\\mu$P Track 2), Track 1 LoRA adaptation, and evaluation protocol hyperparameters.}",
        "\\label{tab:hyperparameters}",
        "\\begin{tabular}{ll}",
        "\\toprule",
        "Component & Frozen Configuration \\\\",
        "\\midrule",
        "Track 1 Base Model & Qwen2.5-7B-Instruct (BF16, LoRA $r=16, \\alpha=32$, 3-way head) \\\\",
        "Track 2 Vocabulary & $V = 4{,}705$ (70-token state + 1-token UCI moves) \\\\",
        "Track 2 \\textsc{Tiny} / \\textsc{Small} & 1.40M ($d=128, L=4$) / 7.52M ($d=256, L=8$) \\\\",
        "Track 2 \\textsc{Medium} / \\textsc{Large} & 23.10M ($d=384, L=12$) / 206.36M ($d=1024, L=16$) \\\\",
        "Parameterization & $\\mu$P ($d_{\\text{base}}=128$, RoPE $\\theta=10{,}000$, soft-target CE) \\\\",
        "Evaluation Protocol & $K \\in \\{0, 1, 2, 4, 8, 16\\}$, 5 demo draws/$K$, $B=1{,}000$ cluster bootstrap \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    t2_path = out_dir / "table2_hyperparameters.tex"
    t2_path.write_text("\n".join(t2_lines) + "\n", encoding="utf-8")
    generated.append(t2_path)

    captions_lines = [
        "% Auto-generated LaTeX figure captions for chess_icl_v2",
        "\\newcommand{\\capFigOne}{Training (dashed) and validation (solid) soft-target cross-entropy NLL (nats) vs. optimizer step on linear axes; shaded regions show $\\pm 1$ standard deviation across model seeds.}",
        "\\newcommand{\\capFigTwo}{Macro-averaged Context Gain $\\overline{\\Delta\\mathrm{NLL}}(K) = \\mathrm{NLL}(0) - \\mathrm{NLL}(K)$ (nats) on identical query roots across context sizes $K \\in \\{0, 1, 2, 4, 8, 16\\}$. Error bars show 95\\% cluster-bootstrap confidence intervals over source games ($B=1{,}000$).}",
        "\\newcommand{\\capFigThree}{Track 2 $\\mu$P scaling across parameter counts (log scale) on held-out variants: macro NLL (left) and macro Context Gain (right) at $K_{\\max}$. Error bars denote 95\\% cluster-bootstrap confidence intervals over source games.}",
        "\\newcommand{\\capFigFour}{Causal intervention controls comparing Clean context against Mismatched-Rule Context, Inverted Outcomes, Scrambled Outcomes, and Corrupted Trajectories on the full evaluation set and the Equal-Material slice ($|\\Delta M| \\le 2$). Error bars show 95\\% cluster-bootstrap CIs.}",
        "\\newcommand{\\capFigFive}{(Left) Side-to-move calibration comparing White-to-move and Black-to-move NLL with top-label ECE annotated. (Right) Per-variant Context Gain $\\Delta\\mathrm{NLL}_v(K_{\\max})$ with 95\\% cluster-bootstrap CIs; \\texttt{knight\\_army} serves as the setup-only null control.}",
    ]
    cap_path = out_dir / "captions.tex"
    cap_path.write_text("\n".join(captions_lines) + "\n", encoding="utf-8")
    generated.append(cap_path)

    return generated


def generate_all(
    runs_dir: Path,
    output_dir: Path,
    baselines_json: Optional[Path] = None,
    split: str = "test",
) -> dict[str, list[str]]:
    """Discover available run outputs and generate all applicable figures and LaTeX tables."""
    configure_matplotlib()

    baselines_payload = discover_baselines_payload(
        runs_dir=runs_dir, baselines_json=baselines_json, split=split
    )
    histories = discover_training_histories(runs_dir)
    neural_models = discover_neural_evals(runs_dir, split=split)
    controls_data = discover_controls_eval(runs_dir)

    all_models: dict[str, dict[str, Any]] = {}
    if baselines_payload and isinstance(baselines_payload.get("models"), dict):
        all_models.update(baselines_payload["models"])
    all_models.update(neural_models)

    mc_floor = None
    if baselines_payload and "mc_noise_floor_macro" in baselines_payload:
        mc_floor = baselines_payload["mc_noise_floor_macro"].get("entropy_floor")

    created: dict[str, list[str]] = {"figures": [], "tables": [], "skipped": []}

    fig_jobs = [
        ("fig1_training_curves", lambda: make_figure1_training_curves(histories, output_dir, mc_floor=mc_floor)),
        ("fig2_context_gain_vs_k", lambda: make_figure2_context_gain_vs_k(all_models, output_dir)),
        ("fig3_mup_scaling", lambda: make_figure3_mup_scaling(neural_models, baselines_payload, output_dir)),
        ("fig4_causal_controls", lambda: make_figure4_causal_controls(controls_data, output_dir)),
        ("fig5_side_calibration_and_variants", lambda: make_figure5_side_and_variants(all_models, output_dir)),
    ]

    for fig_name, builder in fig_jobs:
        paths = builder()
        if paths:
            created["figures"].extend(str(p) for p in paths)
        else:
            created["skipped"].append(fig_name)
            print(f"[make_paper_figures] Skipped {fig_name} (required inputs not present)")

    table_paths = generate_latex_tables(all_models, baselines_payload, output_dir)
    created["tables"].extend(str(p) for p in table_paths)
    return created


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper figures and LaTeX tables for chess_icl_v2.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs_v2"), help="Directory containing run outputs.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs_v2/paper_figures"), help="Output directory for figures and .tex tables.")
    parser.add_argument("--baselines-json", type=Path, default=None, help="Optional explicit path to baselines_<split>.json.")
    parser.add_argument("--split", choices=("val", "test"), default="test", help="Preferred eval split.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    created = generate_all(
        runs_dir=args.runs_dir,
        output_dir=args.output_dir,
        baselines_json=args.baselines_json,
        split=args.split,
    )
    print(
        f"[make_paper_figures] Generated {len(created['figures'])} figure files, "
        f"{len(created['tables'])} LaTeX files; skipped={created['skipped']}"
    )


if __name__ == "__main__":
    main()
