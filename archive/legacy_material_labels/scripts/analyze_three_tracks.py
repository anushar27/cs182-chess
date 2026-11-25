"""Consolidated analysis and publication-quality plotting for all three experimental tracks:
Track 1: Qwen2.5-7B-Instruct LoRA Fine-Tuning & In-Context Learning
Track 2: Frontier muP Transformer Scaling to 680M XLARGE
Track 3: Generative Policy + AlphaZero Continuous Value Head Bot
"""

import json
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import shutil

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def plot_track1_qwen(results_dir: Path, out_path: Path):
    """Plot Base Qwen-7B vs LoRA Fine-Tuned Qwen-7B across held-out variants."""
    base_file = results_dir / "eval_qwen_7b_icl.json"
    ft_file = results_dir / "eval_qwen_7b_finetuned.json"
    
    if not (base_file.exists() and ft_file.exists()):
        print(f"Skipping Track 1 plot: files not found in {results_dir}")
        return
        
    with open(base_file) as f:
        base_data = json.load(f)
    with open(ft_file) as f:
        ft_data = json.load(f)
        
    ft_by_v = defaultdict(lambda: defaultdict(list))
    for r in ft_data:
        ft_by_v[r['variant']][r['k']].append(r['accuracy'])
        
    base_by_v = defaultdict(list)
    for r in base_data.get('results', []):
        base_by_v[r['variant']].append(r['accuracy'])
        
    variants = sorted([v for v in ft_by_v.keys() if v != 'antichess'])
    k_vals = [0, 1, 2, 4, 8]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    
    # Left: Few-shot gain per variant
    colors = plt.cm.tab10(np.linspace(0, 1, len(variants)))
    for idx, v in enumerate(variants):
        means = [np.mean(ft_by_v[v][k]) * 100 for k in k_vals if k in ft_by_v[v]]
        valid_k = [k for k in k_vals if k in ft_by_v[v]]
        ax1.plot(valid_k, means, marker='o', linewidth=2, label=v.replace('_', ' ').title(), color=colors[idx])
        
    ax1.axhline(y=33.33, color='gray', linestyle=':', label='Chance (33.3%)')
    ax1.set_title("LoRA Fine-Tuned Qwen-7B: Accuracy vs K by Variant", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Number of In-Context Demonstrations (K)", fontsize=11)
    ax1.set_ylabel("Evaluation Accuracy (%)", fontsize=11)
    ax1.set_ylim(40, 95)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc='lower right', ncol=2)
    
    # Right: Base vs Fine-Tuned Mean Comparison
    ft_means = [np.mean([np.mean(ft_by_v[v][k]) for v in variants]) * 100 for k in k_vals]
    base_means = [np.mean([np.mean(base_by_v[v]) for v in variants]) * 100 for _ in k_vals]
    
    ax2.plot(k_vals, ft_means, marker='s', linewidth=2.5, color='#2ca02c', label='Fine-Tuned Qwen-7B (LoRA r=16)')
    ax2.plot(k_vals, base_means, marker='x', linewidth=2, color='#d62728', linestyle='--', label='Off-the-Shelf Qwen-7B Base')
    ax2.axhline(y=33.33, color='gray', linestyle=':', label='Chance (33.3%)')
    
    ax2.set_title("Mean Held-Out Variant Accuracy: Base vs Fine-Tuned", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Number of In-Context Demonstrations (K)", fontsize=11)
    ax2.set_ylabel("Mean Accuracy (%)", fontsize=11)
    ax2.set_ylim(25, 85)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc='center right')
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Track 1 plot to {out_path}")


def plot_track2_frontier_scaling(results_dir: Path, out_path: Path):
    """Plot scaling curves across all 5 muP transformer models from Tiny (1.3M) to XLARGE (680M)."""
    scales = [
        ("Tiny (1.3M)", "eval_tiny_mup_all_variants.json", "#1f77b4", "-"),
        ("Small (7.4M)", "eval_small_mup_all_variants.json", "#ff7f0e", "-"),
        ("Medium (22.9M)", "eval_medium_mup_all_variants.json", "#2ca02c", "-"),
        ("Large (205.8M)", "eval_large_mup_all_variants.json", "#d62728", "-"),
        ("XLARGE (680.1M)", "eval_xlarge_mup_all_variants.json", "#9467bd", "-"),
    ]
    
    k_vals = [0, 1, 2, 4, 8, 16]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    
    for label, fname, color, ls in scales:
        fpath = results_dir / fname
        if not fpath.exists():
            continue
        with open(fpath) as f:
            d = json.load(f)
            
        res = defaultdict(lambda: defaultdict(list))
        for r in d['main_results']:
            res[r['variant']][r['K']].append(r['accuracy'])
            
        # Left: All 13 variants mean
        all_means = [np.mean([np.mean(res[v][k]) for v in res]) * 100 for k in k_vals]
        ax1.plot(k_vals, all_means, marker='o', linewidth=2.2, label=label, color=color, linestyle=ls)
        
        # Right: Zero-shot (K=0) accuracy per variant
        k0_no_anti = np.mean([np.mean(res[v][0]) for v in res if v != 'antichess']) * 100
        
    ax1.axhline(y=33.33, color='gray', linestyle=':', label='Chance (33.3%)')
    ax1.set_title("All 13 Variants: Mean Accuracy vs K", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Number of Demonstrations (K)", fontsize=11)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_ylim(25, 95)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc='upper right')
    
    # Right: Parameter Scaling vs Accuracy at K=0 and K=1
    params = [1.3, 7.4, 22.9, 205.8, 680.1]
    k0_accs = []
    k1_accs = []
    for _, fname, _, _ in scales:
        with open(results_dir / fname) as f:
            d = json.load(f)
        res = defaultdict(lambda: defaultdict(list))
        for r in d['main_results']:
            if r['variant'] != 'antichess':
                res[r['variant']][r['K']].append(r['accuracy'])
        k0_accs.append(np.mean([np.mean(res[v][0]) for v in res]) * 100)
        k1_accs.append(np.mean([np.mean(res[v][1]) for v in res]) * 100)
        
    ax2.plot(params, k0_accs, marker='o', linewidth=2.5, color='#1f77b4', label='Zero-Shot (K=0, excl. Antichess)')
    ax2.plot(params, k1_accs, marker='s', linewidth=2.5, color='#2ca02c', label='Few-Shot (K=1, excl. Antichess)')
    ax2.set_xscale('log')
    ax2.set_title("Frontier muP Scaling: Accuracy vs Parameter Count", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Parameters (Millions, Log Scale)", fontsize=11)
    ax2.set_ylabel("Variant Accuracy (%)", fontsize=11)
    ax2.set_ylim(55, 100)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc='center right')
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Track 2 plot to {out_path}")


def plot_track3_alphazero_bot(results_dir: Path, out_path: Path):
    """Plot AlphaZero bot evaluation: Continuous Value MSE vs K and Legal Move Rate."""
    az_file = results_dir / "eval_alphazero_mse.json"
    if not az_file.exists():
        print(f"Skipping Track 3 plot: {az_file} not found")
        return
        
    with open(az_file) as f:
        az_data = json.load(f)
        
    k_stats = defaultdict(lambda: {'mse': [], 'sign': [], 'lmr': []})
    for r in az_data:
        k = r['k']
        k_stats[k]['mse'].append(r['value_mse'])
        k_stats[k]['sign'].append(r['value_sign_accuracy'])
        k_stats[k]['lmr'].append(r['legal_move_rate'])
        
    k_vals = sorted(k_stats.keys())
    mse_means = [np.mean(k_stats[k]['mse']) for k in k_vals]
    sign_means = [np.mean(k_stats[k]['sign']) * 100 for k in k_vals]
    lmr_means = [np.mean(k_stats[k]['lmr']) * 100 for k in k_vals]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    
    # Left: Value MSE & Sign Agreement
    color_mse = '#1f77b4'
    color_sign = '#2ca02c'
    
    ax1.plot(k_vals, mse_means, marker='o', linewidth=2.5, color=color_mse, label='Value MSE vs Minimax Expert')
    ax1.set_xlabel("Number of Demonstrations (K)", fontsize=11)
    ax1.set_ylabel("Continuous Value MSE", color=color_mse, fontsize=11)
    ax1.tick_params(axis='y', labelcolor=color_mse)
    ax1.set_title("AlphaZero Value Head: In-Context Alignment with Expert", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(k_vals, sign_means, marker='s', linewidth=2.5, color=color_sign, linestyle='--', label='Sign Agreement (%)')
    ax1_twin.set_ylabel("Sign Agreement Accuracy (%)", color=color_sign, fontsize=11)
    ax1_twin.tick_params(axis='y', labelcolor=color_sign)
    ax1_twin.set_ylim(25, 50)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left', fontsize=9)
    
    # Right: Policy Legal Move Rate (LMR) vs K
    ax2.plot(k_vals, lmr_means, marker='^', linewidth=2.5, color='#d62728', label='Legal Move Rate (LMR)')
    ax2.set_xlabel("Number of Demonstrations (K)", fontsize=11)
    ax2.set_ylabel("Legal Move Rate (%)", fontsize=11)
    ax2.set_title("Policy Head: Action Legality vs Context Length (K)", fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 25)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc='upper right')
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Track 3 plot to {out_path}")


def main():
    results_dir = Path("runs/results")
    analysis_dir = Path("runs/analysis")
    artifact_dir = Path("/data/backups/rganapa/antigravity/.gemini/antigravity-cli/brain/a2b2d38e-2b3c-4310-bb38-ff6bac961224")
    
    p1 = analysis_dir / "qwen_lora_comparison.png"
    p2 = analysis_dir / "frontier_scaling_icl.png"
    p3 = analysis_dir / "alphazero_bot_eval.png"
    
    plot_track1_qwen(results_dir, p1)
    plot_track2_frontier_scaling(results_dir, p2)
    plot_track3_alphazero_bot(results_dir, p3)
    
    # Copy generated plots to artifact directory
    if artifact_dir.exists():
        for p in [p1, p2, p3]:
            if p.exists():
                shutil.copy(p, artifact_dir / p.name)
                print(f"Copied {p.name} to {artifact_dir}")

if __name__ == "__main__":
    main()
