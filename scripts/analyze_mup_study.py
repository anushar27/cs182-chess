"""Analysis and plotting for the muP scaling and hyperparameter transfer study."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_mup_pretrain_loss(pretrain_dir: Path, output_path: Path):
    """Plot pretraining loss and validation accuracy curves for Tiny, Small, Medium, and Large under muP."""
    scales = ["tiny", "small", "medium", "large"]
    labels = {
        "tiny": "Tiny (1.3M, d=128)",
        "small": "Small (7.4M, d=256)",
        "medium": "Medium (22.9M, d=384)",
        "large": "Large (205.9M, d=1024)",
    }
    colors = {
        "tiny": "#1f77b4",
        "small": "#ff7f0e",
        "medium": "#2ca02c",
        "large": "#d62728",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for scale in scales:
        log_files = list(pretrain_dir.glob(f"pretrain_log_{scale}_mup_seed*.json"))
        if not log_files:
            continue
        # Use first seed log
        with open(log_files[0]) as f:
            data = json.load(f)

        steps = [entry["step"] for entry in data]
        train_loss = [entry["train_loss"] for entry in data]
        val_acc = [entry["val_acc"] * 100 for entry in data]

        ax1.plot(steps, train_loss, label=labels[scale], color=colors[scale], linewidth=2)
        ax2.plot(steps, val_acc, label=labels[scale], color=colors[scale], linewidth=2)

    ax1.set_title("Pretraining Loss (muP Zero-Shot LR Transfer, lr=3e-4)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Optimization Step")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    ax2.set_title("Validation Accuracy vs Scale", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Optimization Step")
    ax2.set_ylabel("Validation Accuracy (%)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved muP pretrain curves to {output_path}")


def plot_model_scaling_icl(results_dir: Path, output_path: Path):
    """Plot ICL Accuracy vs Number of Demonstrations (K) across model sizes and LLM baseline."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    scales = ["tiny", "small", "medium", "large", "qwen_7b"]
    scale_labels = {
        "tiny": "Tiny (1.3M)",
        "small": "Small (7.4M)",
        "medium": "Medium (22.9M)",
        "large": "Large (205.9M)",
        "qwen_7b": "Qwen2.5-7B-Instruct (Zero-Shot LLM)",
    }
    colors = {
        "tiny": "#1f77b4",
        "small": "#ff7f0e",
        "medium": "#2ca02c",
        "large": "#d62728",
        "qwen_7b": "#9467bd",
    }
    linestyles = {
        "tiny": "-",
        "small": "-",
        "medium": "-",
        "large": "-",
        "qwen_7b": "--",
    }
    k_vals = [0, 1, 2, 4, 8, 16]

    target_groups = [
        ("Overall Test Variants (all_variants trained)", "all", axes[0]),
        ("In-Category Transfer (Move Variants)", "move", axes[1]),
        ("Radical Changes (Antichess & Piece Capture)", "radical", axes[2]),
    ]

    for title, grp, ax in target_groups:
        for scale in scales:
            if scale == "qwen_7b":
                res_file = results_dir / "eval_qwen_7b_icl.json"
            else:
                res_file = results_dir / f"eval_{scale}_mup_all_variants.json"
                if not res_file.exists():
                    alt_files = list(results_dir.glob(f"eval_{scale}*_all_variants*.json"))
                    if alt_files:
                        res_file = alt_files[0]
                    else:
                        continue

            if not res_file.exists():
                continue

            with open(res_file) as f:
                raw = json.load(f)
                entries = raw.get("main_results", raw) if isinstance(raw, dict) else raw

            k_accs = defaultdict(list)
            for e in entries:
                v = e["variant"]
                if grp == "move" and v not in ["short_knight", "mobile_king"]:
                    continue
                if grp == "radical" and v not in ["antichess", "piece_capture"]:
                    continue
                k_accs[e["K"]].append(e["accuracy"])

            # Filter valid K values
            valid_k = [k for k in k_vals if k in k_accs and len(k_accs[k]) > 0]
            if not valid_k:
                continue
            means = [np.mean(k_accs[k]) * 100 for k in valid_k]
            ax.plot(
                valid_k,
                means,
                marker="o" if scale != "qwen_7b" else "s",
                linewidth=2.5 if scale in ["large", "qwen_7b"] else 1.8,
                linestyle=linestyles[scale],
                label=scale_labels[scale],
                color=colors[scale],
            )

        ax.axhline(y=33.33, color="gray", linestyle=":", alpha=0.7, label="Chance Baseline (33.3%)")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Number of Demonstrations (K)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(15, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right" if grp != "radical" else "lower right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved model scaling ICL plot to {output_path}")


def generate_mup_latex_table(results_dir: Path, output_path: Path):
    """Generate a clean publication LaTeX table comparing model sizes under muP across all 13 variants."""
    scales = ["tiny", "small", "medium", "large", "qwen_7b"]
    train_variants = [
        ("diagonal_rook", "Move", "Diagonal Rook"),
        ("straight_bishop", "Move", "Straight Bishop"),
        ("weakened_queen", "Move", "Weakened Queen"),
        ("extended_pawn", "Move", "Extended Pawn"),
    ]
    test_variants = [
        ("short_knight", "Move", "Short Knight"),
        ("mobile_king", "Move", "Mobile King"),
        ("piece_capture", "Capture", "Piece Capture"),
        ("forward_pawn_capture", "Capture", "Forward Pawn"),
        ("dark_square_immunity", "Capture", "Dark Immunity"),
        ("blocked_squares", "Board", "Blocked Squares"),
        ("long_board", "Board", "Long Board"),
        ("knight_army", "Pieces", "Knight Army"),
        ("antichess", "Objective", "Antichess"),
    ]

    # Map scale -> variant -> K -> acc
    data = defaultdict(lambda: defaultdict(dict))
    for scale in scales:
        if scale == "qwen_7b":
            res_file = results_dir / "eval_qwen_7b_icl.json"
        else:
            res_file = results_dir / f"eval_{scale}_mup_all_variants.json"
            if not res_file.exists():
                alt = list(results_dir.glob(f"eval_{scale}*_all_variants*.json"))
                if alt:
                    res_file = alt[0]
                else:
                    continue

        if not res_file.exists():
            continue

        with open(res_file) as f:
            raw = json.load(f)
            entries = raw.get("main_results", raw) if isinstance(raw, dict) else raw

        v_k_acc = defaultdict(lambda: defaultdict(list))
        for e in entries:
            v_k_acc[e["variant"]][e["K"]].append(e["accuracy"])

        for v, k_dict in v_k_acc.items():
            for k, accs in k_dict.items():
                data[scale][v][k] = np.mean(accs) * 100

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{ll|ccc|ccc|ccc|ccc|cc}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{\textbf{Tiny (1.3M)}} & \multicolumn{3}{c}{\textbf{Small (7.4M)}} & \multicolumn{3}{c}{\textbf{Medium (22.9M)}} & \multicolumn{3}{c}{\textbf{Large (205.9M)}} & \multicolumn{2}{c}{\textbf{Qwen-7B (LLM)}} \\",
        r"\textbf{Variant} & \textbf{Category} & $K=0$ & $K=4$ & $K=8$ & $K=0$ & $K=4$ & $K=8$ & $K=0$ & $K=4$ & $K=8$ & $K=0$ & $K=4$ & $K=8$ & $K=0$ & $K=4$ \\",
        r"\midrule",
        r"\multicolumn{16}{l}{\textit{In-Distribution Training Variants}} \\",
    ]

    for v_code, cat, v_disp in train_variants:
        row = [f"{v_disp:<18}", f"{cat:<9}"]
        for scale in ["tiny", "small", "medium", "large"]:
            for k in [0, 4, 8]:
                val = data[scale].get(v_code, {}).get(k, 0.0)
                row.append(f"{val:5.1f}\\%" if val > 0 else "   --  ")
        for k in [0, 4]:
            val = data["qwen_7b"].get(v_code, {}).get(k, 0.0)
            row.append(f"{val:5.1f}\\%" if val > 0 else "   --  ")
        lines.append(" & ".join(row) + r" \\")

    lines.extend([
        r"\midrule",
        r"\multicolumn{16}{l}{\textit{Out-of-Distribution Held-Out Test Variants}} \\",
    ])

    for v_code, cat, v_disp in test_variants:
        row = [f"{v_disp:<18}", f"{cat:<9}"]
        for scale in ["tiny", "small", "medium", "large"]:
            for k in [0, 4, 8]:
                val = data[scale].get(v_code, {}).get(k, 0.0)
                row.append(f"{val:5.1f}\\%" if val > 0 else "   --  ")
        for k in [0, 4]:
            val = data["qwen_7b"].get(v_code, {}).get(k, 0.0)
            row.append(f"{val:5.1f}\\%" if val > 0 else "   --  ")
        lines.append(" & ".join(row) + r" \\")

    # Mean summary rows
    lines.append(r"\midrule")
    for group_name, v_list in [("In-Distribution Mean (4)", train_variants), ("Out-of-Distribution Mean (9)", test_variants), ("Overall Mean (13)", train_variants + test_variants)]:
        row = [f"\\textbf{{{group_name}}}", "All"]
        for scale in ["tiny", "small", "medium", "large"]:
            for k in [0, 4, 8]:
                vals = [data[scale].get(vc, {}).get(k, 0.0) for vc, _, _ in v_list if data[scale].get(vc, {}).get(k, 0.0) > 0]
                m_val = np.mean(vals) if vals else 0.0
                row.append(f"\\textbf{{{m_val:5.1f}\\%}}" if m_val > 0 else "   --  ")
        for k in [0, 4]:
            vals = [data["qwen_7b"].get(vc, {}).get(k, 0.0) for vc, _, _ in v_list if data["qwen_7b"].get(vc, {}).get(k, 0.0) > 0]
            m_val = np.mean(vals) if vals else 0.0
            row.append(f"\\textbf{{{m_val:5.1f}\\%}}" if m_val > 0 else "   --  ")
        lines.append(" & ".join(row) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{\textbf{Model Scaling on In-Context Variant Learning: Domain-Specific muP Transformers vs General LLM.} Evaluation accuracy across 13 chess variants on 1,998 balanced evaluation positions per variant (Chance = 33.3\%). Domain-pretrained muP transformers scale monotonically from 1.3M to 205.8M, while general-purpose Qwen2.5-7B-Instruct remains at chance baseline without domain spatial representations.}",
        r"\label{tab:mup_scaling}",
        r"\end{table*}",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved LaTeX table to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze muP scaling study results.")
    parser.add_argument("--pretrain-dir", type=str, default="runs/pretrain")
    parser.add_argument("--results-dir", type=str, default="runs/results")
    parser.add_argument("--output-dir", type=str, default="runs/analysis")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_mup_pretrain_loss(Path(args.pretrain_dir), out_dir / "mup_pretrain_curves.png")
    plot_model_scaling_icl(Path(args.results_dir), out_dir / "mup_model_scaling_icl.png")
    generate_mup_latex_table(Path(args.results_dir), out_dir / "mup_scaling_table.tex")
    print("All muP scaling analysis artifacts created successfully!")


if __name__ == "__main__":
    main()
