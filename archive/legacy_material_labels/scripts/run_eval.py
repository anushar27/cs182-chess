"""Phase 4: Full evaluation matrix across test variants, K values, and seeds."""

import argparse
import json
import sys
from pathlib import Path

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Sequence

import torch

from chess_icl.data import ChessTokenizer, load_positions, load_games
from chess_icl.dataset import ChessEvalDataset
from chess_icl.model import ChessTransformer
from chess_icl.evaluate import compute_accuracy, demo_quality_comparison, position_reevaluation_test
from chess_icl.variants import TEST_VARIANTS, TRAIN_VARIANTS, VARIANTS


def evaluate_variant(
    model: ChessTransformer,
    tokenizer: ChessTokenizer,
    variant_name: str,
    data_dir: Path,
    k_values: Sequence[int],
    seeds: Sequence[int],
    device: str,
    max_len: int = 2048,
    batch_size: int = 32,
) -> list[dict]:
    """Evaluate model on a variant across K values and seeds."""
    v_dir = data_dir / variant_name
    if not (v_dir / "positions_eval.jsonl").exists():
        print(f"Warning: No eval data found for variant {variant_name} in {v_dir}")
        return []

    eval_ds = ChessEvalDataset(v_dir, tokenizer)
    results = []

    for K in k_values:
        for seed in seeds:
            inputs, labels = eval_ds.get_eval_batch(K, seed, max_len=max_len)
            if not inputs:
                continue

            predictions = []
            # Batch inference
            for i in range(0, len(inputs), batch_size):
                batch_inps = inputs[i : i + batch_size]
                # Pad batch
                max_b_len = max(len(inp) for inp in batch_inps)
                padded_batch = [
                    inp + [tokenizer.pad_id] * (max_b_len - len(inp))
                    for inp in batch_inps
                ]
                inp_t = torch.tensor(padded_batch, dtype=torch.long, device=device)

                with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    logits = model(inp_t)
                    preds = logits.argmax(-1).cpu().tolist()
                    predictions.extend(preds)

            metrics = compute_accuracy(predictions, labels)
            entry = {
                "variant": variant_name,
                "category": VARIANTS[variant_name].category.name if variant_name in VARIANTS else "UNKNOWN",
                "split": "train" if variant_name in TRAIN_VARIANTS else "test",
                "K": K,
                "seed": seed,
                "accuracy": metrics["accuracy"],
                "per_class": metrics["per_class"],
                "total": metrics["total"],
            }
            print(f"[{variant_name}] K={K:2d} seed={seed:3d} -> Acc: {metrics['accuracy']:.4f}")
            results.append(entry)

    return results


def run_eval(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint {args.checkpoint} on {device}...")

    model = ChessTransformer.load(args.checkpoint, map_location=device).to(device)
    model.eval()
    tokenizer = ChessTokenizer()
    model.pad_token_id = tokenizer.pad_id

    data_dir = Path(args.data_dir)
    if args.test_variants and len(args.test_variants) == 1 and args.test_variants[0] == "all":
        test_variants = list(VARIANTS.keys())
    elif args.test_variants:
        test_variants = args.test_variants
    else:
        test_variants = list(VARIANTS.keys())

    all_results = []
    for var in test_variants:
        print(f"\nEvaluating on {var}...")
        res = evaluate_variant(
            model=model,
            tokenizer=tokenizer,
            variant_name=var,
            data_dir=data_dir,
            k_values=args.k_values,
            seeds=args.seeds,
            device=device,
            max_len=args.max_seq_len,
            batch_size=args.batch_size,
        )
        all_results.extend(res)

    # Experiment 3: Demo quality comparison (heuristic vs random)
    demo_quality_results = {}
    if not getattr(args, "skip_auxiliary", False):
        print("\nRunning Experiment 3: Demo quality comparison...")
        for var in test_variants:
            v_dir = data_dir / var
            if (v_dir / "games_demo_weak.jsonl").exists():
                dq = demo_quality_comparison(model, tokenizer, v_dir, K_values=[4, 8], device=device)
                demo_quality_results[var] = dq
                print(f"[{var}] Demo quality (K=4,8): {dq}")

    # Experiment 4: Position re-evaluation test (Antichess & reversed outcome positions)
    reeval_results = {}
    if not getattr(args, "skip_auxiliary", False):
        print("\nRunning Experiment 4: Position re-evaluation test...")
        if (data_dir / "antichess").exists() and (data_dir / "standard").exists():
            std_pos = load_positions(data_dir / "standard" / "positions_val.jsonl")
            antichess_pos = load_positions(data_dir / "antichess" / "positions_eval.jsonl")
            antichess_demos = load_games(data_dir / "antichess" / "games_demo.jsonl")
            for K in [0, 4, 8, 16]:
                rev = position_reevaluation_test(
                    model, tokenizer, std_pos, antichess_pos, antichess_demos, K=K, device=device
                )
                reeval_results[f"antichess_K{K}"] = rev
                print(f"[antichess] Re-evaluation K={K}: {rev}")

    output_data = {
        "checkpoint": args.checkpoint,
        "main_results": all_results,
        "demo_quality_results": demo_quality_results,
        "reevaluation_results": reeval_results,
    }

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nEvaluation complete. Results saved to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Chess ICL checkpoints.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test-variants", nargs="+", default=None, help="Variants to evaluate")
    parser.add_argument("--k-values", nargs="+", type=int, default=[0, 1, 2, 4, 8, 16])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output", type=str, default="runs/results/eval_results.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--skip-auxiliary", action="store_true", help="Skip demo quality and re-evaluation auxiliary tests")
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
