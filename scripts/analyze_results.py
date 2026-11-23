"""Post-hoc analysis, plotting, and LaTeX table generation from evaluation results."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np


def plot_k_vs_accuracy(results: list[dict], output_path: Path):
    """Plot Accuracy vs K for each model and variant."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey=True)
    axes = axes.flatten()

    # Group by variant -> K -> list of accuracies
    variant_data = defaultdict(lambda: defaultdict(list))
    for entry in results:
        v = entry["variant"]
        k = entry["K"]
        variant_data[v][k].append(entry["accuracy"])

    variants = sorted(list(variant_data.keys()))

    for i, var in enumerate(variants[:9]):
        ax = axes[i]
        k_dict = variant_data[var]
        ks = sorted(list(k_dict.keys()))
        means = [np.mean(k_dict[k]) for k in ks]
        stds = [np.std(k_dict[k]) for k in ks]

        ax.errorbar(ks, means, yerr=stds, marker="o", capsize=4, linewidth=2, color="#1f77b4")
        ax.axhline(y=0.333, color="gray", linestyle="--", alpha=0.6, label="Random (33.3%)")
        ax.set_title(var, fontsize=12, fontweight="bold")
        ax.set_xlabel("Number of Demos (K)")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper left")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved figure: {output_path}")


def plot_cross_category_transfer(model_results: dict[str, list[dict]], output_path: Path):
    """Compare models (move_variants vs chess_only vs all_variants) across categories."""
    categories = ["MOVE", "CAPTURE", "BOARD", "PIECES", "OBJECTIVE"]
    models = list(model_results.keys())

    cat_acc = {m: {c: [] for c in categories} for m in models}

    for m, entries in model_results.items():
        for e in entries:
            cat = e.get("category", "MOVE")
            if cat in cat_acc[m]:
                cat_acc[m][cat].append(e["accuracy"])

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(categories))
    width = 0.25

    colors = ["#2ca02c", "#1f77b4", "#ff7f0e"]

    for idx, m in enumerate(models):
        means = [np.mean(cat_acc[m][c]) if cat_acc[m][c] else 0.0 for c in categories]
        stds = [np.std(cat_acc[m][c]) if cat_acc[m][c] else 0.0 for c in categories]
        ax.bar(x + idx * width - width, means, width, yerr=stds, label=m, color=colors[idx % len(colors)], capsize=3)

    ax.axhline(y=0.333, color="gray", linestyle="--", alpha=0.6, label="Random")
    ax.set_ylabel("Average Accuracy", fontsize=12)
    ax.set_title("Cross-Category ICL Generalization", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved figure: {output_path}")


def generate_latex_table(results: list[dict], output_path: Path):
    """Generate LaTeX table of accuracy across K for all variants."""
    variant_k_data = defaultdict(lambda: defaultdict(list))
    for entry in results:
        v = entry["variant"]
        k = entry["K"]
        variant_k_data[v][k].append(entry["accuracy"])

    k_values = [0, 1, 2, 4, 8, 16]

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Variant & K=0 & K=1 & K=2 & K=4 & K=8 & K=16 \\")
    lines.append(r"\midrule")

    for v, k_dict in sorted(variant_k_data.items()):
        row = [v.replace("_", r"\_")]
        for k in k_values:
            accs = k_dict.get(k, [])
            if accs:
                mean = np.mean(accs) * 100
                std = np.std(accs) * 100
                row.append(f"{mean:.1f} $\\pm$ {std:.1f}")
            else:
                row.append("-")
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{In-context evaluation accuracy (\%) across demonstration count $K$.}")
    lines.append(r"\label{tab:icl_results}")
    lines.append(r"\end{table}")

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

    result_files = list(res_dir.glob("*.json"))
    if not result_files:
        print(f"No result files found in {res_dir}")
        return

    all_models_results = {}
    combined_entries = []

    for rf in result_files:
        try:
            with open(rf) as f:
                data = json.load(f)
                entries = data.get("main_results", [])
                name = rf.stem
                all_models_results[name] = entries
                combined_entries.extend(entries)
        except Exception as e:
            print(f"Error reading {rf}: {e}")

    if combined_entries:
        plot_k_vs_accuracy(combined_entries, out_dir / "k_vs_accuracy.png")
        generate_latex_table(combined_entries, out_dir / "results_table.tex")

    if len(all_models_results) > 1:
        plot_cross_category_transfer(all_models_results, out_dir / "cross_category_transfer.png")

    print(f"Analysis complete. Generated artifacts in {out_dir}")


if __name__ == "__main__":
    main()
