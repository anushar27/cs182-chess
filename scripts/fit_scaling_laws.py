"""Fit neural scaling laws, compute bootstrap confidence intervals, and confusion matrices."""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


MODEL_PARAMS = {
    "tiny": 1_327_875,
    "small": 7_407_875,
    "medium": 22_905_219,
    "large": 205_827_075,
}

TRAIN_VARIANTS = {"diagonal_rook", "straight_bishop", "weakened_queen", "extended_pawn"}


def fit_power_law_numpy(N_vals, acc_vals):
    """Fit Error = 1 - Acc = A * N^(-alpha) => log(1 - Acc) = log(A) - alpha * log(N)."""
    errors = np.clip(1.0 - acc_vals, 1e-4, 1.0)
    log_N = np.log(N_vals)
    log_E = np.log(errors)
    slope, intercept = np.polyfit(log_N, log_E, 1)
    alpha = -slope
    A = np.exp(intercept)
    return A, alpha


def predict_power_law(N, A, alpha):
    return np.clip(1.0 - A * (N ** (-alpha)), 0.0, 1.0)



def compute_bootstrap_ci(data: list[float], n_bootstrap: int = 2000, ci: float = 0.95):
    """Compute 95% bootstrap confidence interval."""
    if len(data) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(42)
    means = [np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)]
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return np.mean(data), lower, upper


def analyze_scaling_and_stats(results_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load results
    model_data = {}
    for scale in ["tiny", "small", "medium", "large"]:
        fpath = results_dir / f"eval_{scale}_mup_all_variants.json"
        if not fpath.exists():
            continue
        with open(fpath) as f:
            raw = json.load(f)
        model_data[scale] = raw.get("main_results", raw)
        
    # Aggregate accuracies by (scale, K, group)
    # group in ["overall", "ood", "id"]
    stats = defaultdict(list)
    for scale, entries in model_data.items():
        for e in entries:
            k = e["K"]
            acc = e["accuracy"]
            v = e["variant"]
            stats[(scale, k, "overall")].append(acc)
            if v in TRAIN_VARIANTS:
                stats[(scale, k, "id")].append(acc)
            else:
                stats[(scale, k, "ood")].append(acc)
                
    # 2. Fit power law for K=1 and K=2 OOD
    scales = ["tiny", "small", "medium", "large"]
    N_vals = np.array([MODEL_PARAMS[s] for s in scales])
    
    k1_ood_means = np.array([np.mean(stats[(s, 1, "ood")]) for s in scales])
    k2_ood_means = np.array([np.mean(stats[(s, 2, "ood")]) for s in scales])
    
    A_k1, alpha_k1 = fit_power_law_numpy(N_vals, k1_ood_means)
    A_k2, alpha_k2 = fit_power_law_numpy(N_vals, k2_ood_means)
    
    # Extrapolate to 1B parameters
    extrap_N = 1_000_000_000
    pred_k1_1B = predict_power_law(extrap_N, A_k1, alpha_k1)
    pred_k2_1B = predict_power_law(extrap_N, A_k2, alpha_k2)
    
    print("=== Neural Scaling Law Fits ===")
    print(f"K=1 OOD Scaling Law: Error(N) = {A_k1:.3f} * N^(-{alpha_k1:.4f})")
    print(f"K=1 OOD Predicted at 1B params: {pred_k1_1B*100:.2f}%")
    print(f"K=2 OOD Scaling Law: Error(N) = {A_k2:.3f} * N^(-{alpha_k2:.4f})")
    print(f"K=2 OOD Predicted at 1B params: {pred_k2_1B*100:.2f}%")
    
    # 3. Plot fitted scaling curve
    fig, ax = plt.subplots(figsize=(8, 5.5))
    dense_N = np.logspace(5.8, 9.2, 200)
    
    ax.scatter(N_vals / 1e6, k1_ood_means * 100, color="#d62728", s=80, zorder=5, label="Observed K=1 OOD (Empirical)")
    ax.plot(dense_N / 1e6, predict_power_law(dense_N, A_k1, alpha_k1) * 100, color="#d62728", linestyle="-", linewidth=2, label=f"Power Law Fit: $\\alpha = {alpha_k1:.4f}$ (1B Pred: {pred_k1_1B*100:.1f}%)")
    
    ax.scatter(N_vals / 1e6, k2_ood_means * 100, color="#1f77b4", s=80, zorder=5, label="Observed K=2 OOD (Empirical)")
    ax.plot(dense_N / 1e6, predict_power_law(dense_N, A_k2, alpha_k2) * 100, color="#1f77b4", linestyle="--", linewidth=2, label=f"Power Law Fit: $\\alpha = {alpha_k2:.4f}$ (1B Pred: {pred_k2_1B*100:.1f}%)")
    
    ax.axhline(y=33.33, color="gray", linestyle=":", label="Chance Baseline (33.3%)")
    ax.scatter([1000], [pred_k1_1B * 100], color="#d62728", marker="*", s=200, zorder=6, label="1B Param Extrapolation")
    ax.scatter([1000], [pred_k2_1B * 100], color="#1f77b4", marker="*", s=200, zorder=6)
    
    ax.set_xscale("log")
    ax.set_xlabel("Model Parameters (Millions)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Out-of-Distribution In-Context Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("Empirical Power-Law Scaling on Chess Variant In-Context Learning", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    plot_path = output_dir / "neural_scaling_law_fit.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved scaling law fit plot to {plot_path}")
    
    # 4. Save scaling law parameters to JSON
    scaling_summary = {
        "k1_ood_fit": {
            "coefficient_A": float(A_k1),
            "power_exponent_alpha": float(alpha_k1),
            "predicted_1B_acc": float(pred_k1_1B),
        },
        "k2_ood_fit": {
            "coefficient_A": float(A_k2),
            "power_exponent_alpha": float(alpha_k2),
            "predicted_1B_acc": float(pred_k2_1B),
        },
        "empirical_points": {
            scale: {
                "params": MODEL_PARAMS[scale],
                "k1_ood_acc": float(k1_ood_means[i]),
                "k2_ood_acc": float(k2_ood_means[i]),
            }
            for i, scale in enumerate(scales)
        }
    }
    with open(output_dir / "scaling_law_fit.json", "w") as f:
        json.dump(scaling_summary, f, indent=2)


if __name__ == "__main__":
    analyze_scaling_and_stats(Path("runs/results"), Path("runs/analysis"))
