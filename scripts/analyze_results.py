"""Post-hoc analysis, plotting, and LaTeX table generation from evaluation results."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_model_comparison_by_k(model_results: dict[str, list[dict]], output_path: Path):
    """Plot Accuracy vs K for each variant comparing the three models."""
    variants = [
        "short_knight", "mobile_king", "piece_capture",
        "forward_pawn_capture", "dark_square_immunity", "blocked_squares",
        "long_board", "knight_army", "antichess"
    ]

    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharey=True)
    axes = axes.flatten()

    models = ["chess_only", "move_variants", "all_variants"]
    model_labels = {
        "chess_only": "Standard Chess Pretrain Only (Zero-shot)",
        "move_variants": "Trained on Move Variants",
        "all_variants": "Trained on All Variants (Upper bound)",
    }
    colors = {
        "chess_only": "#d62728",      # red
        "move_variants": "#1f77b4",   # blue
        "all_variants": "#2ca02c",    # green
    }

    k_values = [0, 1, 2, 4, 8, 16]

    for idx, var in enumerate(variants):
        ax = axes[idx]
        for m in models:
            entries = model_results.get(m, [])
            k_dict = defaultdict(list)
            for e in entries:
                if e["variant"] == var:
                    k_dict[e["K"]].append(e["accuracy"])

            means = [np.mean(k_dict[k]) * 100 if k_dict[k] else 0.0 for k in k_values]
            stds = [np.std(k_dict[k]) * 100 if k_dict[k] else 0.0 for k in k_values]

            ax.errorbar(
                k_values, means, yerr=stds,
                marker="o", capsize=3, label=model_labels[m],
                color=colors[m], linewidth=2
            )

        ax.axhline(y=33.33, color="gray", linestyle="--", alpha=0.5, label="Random Baseline (33.3%)" if idx == 0 else "")
        ax.set_title(var.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_xlabel("Demonstrations (K)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="lower left", fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved figure: {output_path}")


def plot_cross_category_transfer(model_results: dict[str, list[dict]], output_path: Path):
    """Compare models across variant categories."""
    categories = ["MOVE", "CAPTURE", "BOARD", "PIECES", "OBJECTIVE"]
    models = ["chess_only", "move_variants", "all_variants"]
    labels = ["Standard Only", "Move Variants", "All Variants"]

    cat_acc = {m: {c: [] for c in categories} for m in models}

    for m in models:
        entries = model_results.get(m, [])
        for e in entries:
            # Only consider K in [1, 8] for in-context evaluation
            if e["K"] in [1, 2, 4, 8]:
                cat = e.get("category", "MOVE")
                if cat in cat_acc[m]:
                    cat_acc[m][cat].append(e["accuracy"] * 100)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(categories))
    width = 0.25

    colors = ["#d62728", "#1f77b4", "#2ca02c"]

    for idx, m in enumerate(models):
        means = [np.mean(cat_acc[m][c]) if cat_acc[m][c] else 0.0 for c in categories]
        stds = [np.std(cat_acc[m][c]) if cat_acc[m][c] else 0.0 for c in categories]
        ax.bar(
            x + (idx - 1) * width, means, width,
            yerr=stds, label=labels[idx],
            color=colors[idx], capsize=4, alpha=0.88, edgecolor="black"
        )

    ax.axhline(y=33.33, color="gray", linestyle="--", alpha=0.7, label="Random (33.3%)")
    ax.set_ylabel("Mean Accuracy (%) (K=1..8)", fontsize=12)
    ax.set_title("Cross-Category In-Context Learning Transfer", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight="medium")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved figure: {output_path}")


def generate_comprehensive_latex_table(model_results: dict[str, list[dict]], output_path: Path):
    """Generate detailed LaTeX table comparing models across K values."""
    variants = [
        "short_knight", "mobile_king", "piece_capture",
        "forward_pawn_capture", "dark_square_immunity", "blocked_squares",
        "long_board", "knight_army", "antichess"
    ]
    models = ["chess_only", "move_variants", "all_variants"]
    model_short = {"chess_only": "Base", "move_variants": "Move", "all_variants": "All"}
    k_values = [0, 1, 2, 4, 8]

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\toprule")
    lines.append(r"Variant & Model & K=0 & K=1 & K=2 & K=4 & K=8 \\")
    lines.append(r"\midrule")

    for v in variants:
        v_name = v.replace("_", r"\_")
        for m_idx, m in enumerate(models):
            entries = model_results.get(m, [])
            k_dict = defaultdict(list)
            for e in entries:
                if e["variant"] == v:
                    k_dict[e["K"]].append(e["accuracy"])

            row_header = v_name if m_idx == 0 else ""
            model_tag = model_short[m]
            vals = []
            for k in k_values:
                accs = k_dict.get(k, [])
                if accs:
                    mean = np.mean(accs) * 100
                    std = np.std(accs) * 100
                    vals.append(f"{mean:.1f} $\\pm$ {std:.1f}")
                else:
                    vals.append("-")
            lines.append(f"{row_header} & {model_tag} & " + " & ".join(vals) + r" \\")
        lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{In-Context Evaluation Accuracy (\%) across demonstration count $K$ for each training condition.}")
    lines.append(r"\label{tab:icl_comprehensive}")
    lines.append(r"\end{table*}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved LaTeX table: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Chess ICL evaluation results.")
    parser.add_argument("--results-dir", type=str, default="runs/results")
    parser.add_argument("--output-dir", type=str, default="runs/analysis")
    args = parser.parse_args()

    res_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group by model type (chess_only, move_variants, all_variants)
    model_types = ["chess_only", "move_variants", "all_variants"]
    model_results = {m: [] for m in model_types}

    for rf in res_dir.glob("*.json"):
        for m in model_types:
            if rf.name.startswith(m):
                try:
                    with open(rf) as f:
                        data = json.load(f)
                        model_results[m].extend(data.get("main_results", []))
                except Exception as e:
                    print(f"Error reading {rf}: {e}")

    plot_model_comparison_by_k(model_results, out_dir / "k_vs_accuracy_comparison.png")
    plot_cross_category_transfer(model_results, out_dir / "cross_category_transfer.png")
    generate_comprehensive_latex_table(model_results, out_dir / "comprehensive_table.tex")
    print(f"All analysis figures and tables generated in {out_dir}")


if __name__ == "__main__":
    main()
