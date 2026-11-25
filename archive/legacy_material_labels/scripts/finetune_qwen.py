"""LoRA fine-tuning for Qwen2.5-7B-Instruct on Chess Domain & Variant Demonstrations."""

import argparse
import json
import math
import sys
from pathlib import Path

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from chess_icl.data import load_games, load_positions
from chess_icl.variants import TRAIN_VARIANTS, VARIANTS


def build_training_prompt(
    demos: list[dict],
    query_fen: str,
    query_side: str,
    label: str,
    tokenizer,
) -> tuple[str, str]:
    """Builds prompt and target completion for instruction tuning."""
    system_msg = (
        "You are a grandmaster chess engine evaluating positions under custom chess variants. "
        "In this variant, pieces may have non-standard movement, capture rules, board geometry, or win conditions. "
        "Study the demonstration games below to infer the variant rules, then evaluate the query position."
    )

    demo_text = ""
    if demos:
        demo_text += f"\n\nHere are {len(demos)} demonstration game(s) played under this variant's rules:\n"
        for i, g in enumerate(demos, 1):
            moves_str = " ".join(g.get("moves", [])[:40])
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

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_prompt},
    ]
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt_str, label.strip().lower()


class ChessQwenDataset(Dataset):
    """Dataset combining standard chess positions and in-context variant demonstrations."""

    def __init__(
        self,
        data_dir: Path,
        tokenizer,
        max_samples: int = 15000,
        max_len: int = 1536,
        include_variants: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.items = []

        # 1. Standard chess positions
        std_pos_file = data_dir / "standard" / "positions_train.jsonl"
        if std_pos_file.exists():
            std_positions = load_positions(std_pos_file)
            # Sample balanced standard positions
            wins = [p for p in std_positions if p.get("label") == "win"]
            draws = [p for p in std_positions if p.get("label") == "draw"]
            losses = [p for p in std_positions if p.get("label") == "loss"]
            n_each = min(len(wins), len(draws), len(losses), max_samples // 6)
            for p in (wins[:n_each] + draws[:n_each] + losses[:n_each]):
                fen = p["fen"]
                side = "w" if " w " in fen else "b"
                self.items.append(([], fen, side, p["label"]))

        # 2. In-context variant demonstrations from training variants
        if include_variants:
            for v_name in TRAIN_VARIANTS:
                v_dir = data_dir / v_name
                p_file = v_dir / "positions_train.jsonl"
                if not p_file.exists():
                    p_file = v_dir / "positions_eval.jsonl"
                d_file = v_dir / "games_demo.jsonl"
                if p_file.exists() and d_file.exists():
                    pos_list = load_positions(p_file)
                    demo_list = load_games(d_file)
                    v_wins = [p for p in pos_list if p.get("label") == "win"]
                    v_draws = [p for p in pos_list if p.get("label") == "draw"]
                    v_losses = [p for p in pos_list if p.get("label") == "loss"]
                    n_v = min(len(v_wins), len(v_draws), len(v_losses), 300)
                    for p in (v_wins[:n_v] + v_draws[:n_v] + v_losses[:n_v]):
                        fen = p["fen"]
                        side = "w" if " w " in fen else "b"
                        # pick 1 or 2 demos
                        demos = demo_list[:2]
                        self.items.append((demos, fen, side, p["label"]))

        print(f"Total training examples for Qwen fine-tuning: {len(self.items)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        demos, fen, side, label = self.items[idx]
        prompt, target = build_training_prompt(demos, fen, side, label, self.tokenizer)

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target + "<|im_end|>\n", add_special_tokens=False)

        input_ids = prompt_ids + target_ids
        # Mask out prompt tokens in labels (loss only on completion)
        labels = [-100] * len(prompt_ids) + target_ids

        if len(input_ids) > self.max_len:
            input_ids = input_ids[-self.max_len :]
            labels = labels[-self.max_len :]

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collate_fn(batch, pad_token_id):
    input_ids_list, labels_list = zip(*batch)
    max_len = max(len(ids) for ids in input_ids_list)

    padded_inputs = []
    padded_labels = []
    attention_masks = []

    for in_ids, labs in zip(input_ids_list, labels_list):
        pad_len = max_len - len(in_ids)
        # Left pad for causal LM generation compatibility
        p_ids = torch.cat([torch.full((pad_len,), pad_token_id, dtype=torch.long), in_ids])
        p_labs = torch.cat([torch.full((pad_len,), -100, dtype=torch.long), labs])
        p_mask = torch.cat([torch.zeros(pad_len, dtype=torch.long), torch.ones(len(in_ids), dtype=torch.long)])

        padded_inputs.append(p_ids)
        padded_labels.append(p_labs)
        attention_masks.append(p_mask)

    return (
        torch.stack(padded_inputs),
        torch.stack(attention_masks),
        torch.stack(padded_labels),
    )


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning Qwen-7B for Chess ICL")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="runs/qwen_lora")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=6000)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=200)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Fine-tuning Qwen on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model {args.model_name} in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = ChessQwenDataset(Path(args.data_dir), tokenizer, max_samples=args.max_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=1e-6)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    running_loss = 0.0
    model.train()
    print(f"Starting LoRA training for {args.max_steps} steps (grad accum = {args.grad_accum})...")

    optimizer.zero_grad()
    while step < args.max_steps:
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            running_loss += loss.item() * args.grad_accum

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % args.log_interval == 0:
                    avg_loss = running_loss / args.log_interval
                    running_loss = 0.0
                    print(f"Step {step}/{args.max_steps} | loss={avg_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

                if step % args.save_interval == 0 or step >= args.max_steps:
                    save_path = out_dir / f"checkpoint_step_{step}"
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    print(f"  -> Saved LoRA checkpoint to {save_path}")

                if step >= args.max_steps:
                    break

    final_path = out_dir / "qwen_lora_final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Fine-tuning complete! Final model saved to {final_path}")


if __name__ == "__main__":
    main()
