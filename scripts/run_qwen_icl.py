"""Evaluate Qwen-7B (Qwen2.5-7B-Instruct) on Chess Variant In-Context Learning."""

import argparse
import json
import sys
from pathlib import Path

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from chess_icl.data import load_games, load_positions
from chess_icl.evaluate import compute_accuracy
from chess_icl.variants import VARIANTS, TRAIN_VARIANTS, TEST_VARIANTS


def build_qwen_prompt(
    demos: list[dict],
    query_fen: str,
    query_side: str,
    variant_name: str,
    tokenizer,
) -> str:
    """Format in-context variant demonstration games and query position for Qwen."""
    system_msg = (
        "You are a grandmaster chess engine evaluating positions under custom chess variants. "
        "In this variant, pieces may have non-standard movement, capture rules, board geometry, or win conditions. "
        "Study the demonstration games below to infer the variant rules, then evaluate the query position."
    )

    demo_text = ""
    if demos:
        demo_text += f"\n\nHere are {len(demos)} demonstration game(s) played under this variant's rules:\n"
        for i, g in enumerate(demos, 1):
            moves_str = " ".join(g["moves"][:50])  # Cap moves for token efficiency
            res_str = g.get("result", "draw")
            demo_text += f"\n[Demo Game {i}]\nMoves: {moves_str}\nOutcome for White: {res_str}\n"

    query_side_full = "White" if query_side.lower().startswith("w") else "Black"
    user_prompt = (
        f"{demo_text}\n"
        f"[Query Position]\n"
        f"FEN: {query_fen}\n"
        f"Side to move: {query_side_full}\n\n"
        f"Based on the variant rules demonstrated, evaluate whether this position is a 'win', 'draw', or 'loss' "
        f"for {query_side_full}.\n"
        f"Answer strictly with exactly one word: 'win', 'draw', or 'loss'."
    )

    # Use chat template if available
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def evaluate_qwen_variant(
    model,
    tokenizer,
    variant_name: str,
    data_dir: Path,
    k_values: list[int],
    seeds: list[int],
    num_samples: int = 200,
    batch_size: int = 8,
    device: str = "cuda:5",
) -> list[dict]:
    """Evaluate Qwen on a variant across K values and seeds."""
    v_dir = data_dir / variant_name
    pos_file = v_dir / "positions_eval.jsonl"
    demo_file = v_dir / "games_demo.jsonl"

    if not pos_file.exists() or not demo_file.exists():
        print(f"Warning: Missing data for {variant_name} in {v_dir}")
        return []

    positions = load_positions(pos_file)
    demos_pool = load_games(demo_file)

    if num_samples and len(positions) > num_samples:
        # Balanced sampling across classes
        wins = [p for p in positions if p.get("label") == "win"]
        draws = [p for p in positions if p.get("label") == "draw"]
        losses = [p for p in positions if p.get("label") == "loss"]
        per_cls = num_samples // 3
        positions = wins[:per_cls] + draws[:per_cls] + losses[:per_cls]

    # Target token IDs for Qwen vocabulary
    win_tokens = [7526, 3164, 16970, 12190]   # win,  win, Win,  Win
    draw_tokens = [7633, 4038, 8137, 11992]  # draw,  draw, Draw,  Draw
    loss_tokens = [9379, 4709, 39838, 24704] # loss,  loss, Loss,  Loss

    label_to_idx = {"win": 0, "draw": 1, "loss": 2}
    ground_truth = [label_to_idx.get(p.get("label", "draw"), 1) for p in positions]

    results = []

    for K in k_values:
        for seed in seeds:
            import random
            rng = random.Random(seed)
            if K == 0 or not demos_pool:
                sampled_demos = []
            else:
                sampled_demos = rng.sample(demos_pool, min(K, len(demos_pool)))

            # Build prompts
            prompts = [
                build_qwen_prompt(
                    demos=sampled_demos,
                    query_fen=p["fen"],
                    query_side=p.get("side_to_move", "w"),
                    variant_name=variant_name,
                    tokenizer=tokenizer,
                )
                for p in positions
            ]

            predictions = []
            # Batch inference
            for i in range(0, len(prompts), batch_size):
                b_prompts = prompts[i : i + batch_size]
                inputs = tokenizer(b_prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)

                with torch.inference_mode():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    # Get logits at the final prompt token
                    last_logits = outputs.logits[:, -1, :]

                    score_win = torch.logsumexp(last_logits[:, win_tokens], dim=-1)
                    score_draw = torch.logsumexp(last_logits[:, draw_tokens], dim=-1)
                    score_loss = torch.logsumexp(last_logits[:, loss_tokens], dim=-1)

                    scores = torch.stack([score_win, score_draw, score_loss], dim=-1)
                    preds = scores.argmax(dim=-1).cpu().tolist()
                    predictions.extend(preds)

            metrics = compute_accuracy(predictions, ground_truth)
            entry = {
                "variant": variant_name,
                "category": VARIANTS[variant_name].category.name if variant_name in VARIANTS else "UNKNOWN",
                "split": "train" if variant_name in TRAIN_VARIANTS else "test",
                "K": K,
                "seed": seed,
                "accuracy": metrics["accuracy"],
                "total": metrics["total"],
                "per_class": metrics["per_class"],
            }
            print(f"[Qwen-7B] [{variant_name:20s}] K={K:2d} seed={seed:3d} -> Acc: {metrics['accuracy']:.4f} (N={metrics['total']})")
            results.append(entry)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen-7B on Chess Variant ICL.")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--k-values", nargs="+", type=int, default=[0, 1, 2, 4, 8])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123])
    parser.add_argument("--num-samples", type=int, default=300, help="Number of eval samples per variant")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--output", type=str, default="runs/results/eval_qwen_7b_icl.json")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_id} on {device} in bfloat16...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    variants = args.variants if args.variants and args.variants != ["all"] else list(VARIANTS.keys())
    print(f"Evaluating {len(variants)} variants: {variants}")

    data_dir = Path(args.data_dir)
    all_results = []

    for v in variants:
        print(f"\n--- Evaluating Variant: {v} ---")
        res = evaluate_qwen_variant(
            model=model,
            tokenizer=tokenizer,
            variant_name=v,
            data_dir=data_dir,
            k_values=args.k_values,
            seeds=args.seeds,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            device=device,
        )
        all_results.extend(res)

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({"model": args.model_id, "main_results": all_results}, f, indent=2)

    print(f"\nQwen-7B evaluation complete! Results saved to {out_file}")


if __name__ == "__main__":
    main()
