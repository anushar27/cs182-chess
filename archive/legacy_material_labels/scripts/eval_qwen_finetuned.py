"""Evaluate LoRA Fine-Tuned Qwen-7B on Chess Variant In-Context Learning."""

import argparse
import json
import sys
from pathlib import Path

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from chess_icl.data import load_games, load_positions
from chess_icl.evaluate import compute_accuracy
from chess_icl.variants import VARIANTS, TEST_VARIANTS
from scripts.run_qwen_icl import build_qwen_prompt


def evaluate_finetuned_qwen_variant(
    model,
    tokenizer,
    variant_name: str,
    data_dir: Path,
    k_values: list[int],
    seeds: list[int],
    num_samples: int = 200,
    batch_size: int = 8,
    device: str = "cuda:1",
) -> list[dict]:
    """Evaluate fine-tuned Qwen on a variant across K values and seeds."""
    v_dir = data_dir / variant_name
    pos_file = v_dir / "positions_eval.jsonl"
    demo_file = v_dir / "games_demo.jsonl"

    if not pos_file.exists() or not demo_file.exists():
        print(f"Warning: Missing data for {variant_name} in {v_dir}")
        return []

    positions = load_positions(pos_file)
    demos_pool = load_games(demo_file)

    if num_samples and len(positions) > num_samples:
        wins = [p for p in positions if p.get("label") == "win"]
        draws = [p for p in positions if p.get("label") == "draw"]
        losses = [p for p in positions if p.get("label") == "loss"]
        per_cls = num_samples // 3
        positions = wins[:per_cls] + draws[:per_cls] + losses[:per_cls]

    win_tokens = [7526, 3164, 16970, 12190]
    draw_tokens = [7633, 4038, 8137, 11992]
    loss_tokens = [9379, 4709, 39838, 24704]

    label_to_idx = {"win": 0, "draw": 1, "loss": 2}
    ground_truth = [label_to_idx.get(p.get("label", "draw"), 1) for p in positions]

    results = []

    for K in k_values:
        for seed in seeds:
            # Deterministic demo selection
            import random
            rng = random.Random(seed)
            if K > 0 and len(demos_pool) >= K:
                chosen_demos = rng.sample(demos_pool, K)
            elif K > 0:
                chosen_demos = demos_pool[:K]
            else:
                chosen_demos = []

            prompts = []
            for p in positions:
                fen = p["fen"]
                side = "w" if " w " in fen else "b"
                prompt = build_qwen_prompt(chosen_demos, fen, side, variant_name, tokenizer)
                prompts.append(prompt)

            predictions = []
            with torch.inference_mode():
                for i in range(0, len(prompts), batch_size):
                    batch_prompts = prompts[i : i + batch_size]
                    enc = tokenizer(
                        batch_prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=2048,
                    ).to(device)

                    outputs = model(**enc)
                    next_token_logits = outputs.logits[:, -1, :]

                    win_logits = next_token_logits[:, win_tokens].max(dim=-1).values
                    draw_logits = next_token_logits[:, draw_tokens].max(dim=-1).values
                    loss_logits = next_token_logits[:, loss_tokens].max(dim=-1).values

                    cls_logits = torch.stack([win_logits, draw_logits, loss_logits], dim=-1)
                    preds = cls_logits.argmax(dim=-1).cpu().tolist()
                    predictions.extend(preds)

            correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
            accuracy = correct / len(ground_truth) if ground_truth else 0.0

            cls_acc = {}
            for c_name, c_idx in label_to_idx.items():
                c_mask = [g == c_idx for g in ground_truth]
                c_total = sum(c_mask)
                c_corr = sum(1 for p, g, m in zip(predictions, ground_truth, c_mask) if m and p == g)
                cls_acc[c_name] = c_corr / c_total if c_total > 0 else 0.0

            results.append({
                "variant": variant_name,
                "k": K,
                "seed": seed,
                "accuracy": accuracy,
                "num_samples": len(ground_truth),
                "class_accuracies": cls_acc,
            })
            print(f"  {variant_name} | K={K} seed={seed} | Acc: {accuracy * 100:.2f}% | per-class: {cls_acc}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Fine-Tuned Qwen on Chess Variants")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora-path", type=str, default="runs/qwen_lora/qwen_lora_final")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output", type=str, default="runs/results/eval_qwen_7b_finetuned.json")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--k-values", nargs="+", type=int, default=[0, 1, 2, 4, 8])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123])
    parser.add_argument("--num-samples", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Loading Qwen base model {args.base_model} and LoRA from {args.lora_path} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )

    lora_path = Path(args.lora_path)
    if lora_path.exists():
        print(f"Applying LoRA weights from {lora_path}...")
        model = PeftModel.from_pretrained(base_model, str(lora_path))
    else:
        print(f"Warning: {lora_path} not found. Running with base model.")
        model = base_model

    model.eval()

    all_results = []
    print(f"Evaluating across held-out variants: {TEST_VARIANTS}")
    for variant in TEST_VARIANTS:
        print(f"\n--- Evaluating {variant} ---")
        v_res = evaluate_finetuned_qwen_variant(
            model=model,
            tokenizer=tokenizer,
            variant_name=variant,
            data_dir=Path(args.data_dir),
            k_values=args.k_values,
            seeds=args.seeds,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            device=device,
        )
        all_results.extend(v_res)

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved fine-tuned Qwen evaluation results to {out_file}")


if __name__ == "__main__":
    main()
