"""Train Generative Chess Bot with Policy Head & AlphaZero Value Head."""

import argparse
import copy
import json
import math
import sys
from pathlib import Path

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from chess_icl.data import ChessTokenizer, load_games
from chess_icl.model import ModelConfig, SMALL, MEDIUM
from chess_icl.bot_model import ChessBotPolicyValue


class ChessGameTrajectoryDataset(Dataset):
    """Dataset of full game move sequences with AlphaZero scalar outcome targets."""

    def __init__(self, games_files: list[Path], tokenizer: ChessTokenizer, max_len: int = 1024, max_games: int = 10000):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        val_map = {"win": 1.0, "draw": 0.0, "loss": -1.0}

        for g_file in games_files:
            if not g_file.exists():
                continue
            games = load_games(g_file)
            for g in games[:max_games]:
                moves = g.get("moves", [])
                if len(moves) < 4:
                    continue
                # Truncate moves if needed to fit max_len
                moves = moves[:120]
                res_str = g.get("result", "draw")
                val_target = val_map.get(res_str, 0.0)

                # Build token sequence: [CLS] [GAME_START] m1 m2 ... [result] [GAME_END]
                tokens = [self.tokenizer.cls_id, self.tokenizer.game_start_id]
                for m in moves:
                    tokens.extend(self.tokenizer.encode_move(m))
                res_token = self.tokenizer.token_to_id[self.tokenizer.result_token_map.get(res_str, self.tokenizer.DRAW_TOKEN)]
                tokens.append(res_token)
                tokens.append(self.tokenizer.game_end_id)

                if len(tokens) > max_len:
                    tokens = tokens[:max_len]

                self.samples.append((tokens, val_target))

        print(f"Loaded {len(self.samples)} game trajectories for policy/value training.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, target_val = self.samples[idx]
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(target_val, dtype=torch.float32)


def collate_fn(batch, pad_id):
    seqs, targets = zip(*batch)
    max_len = max(len(s) for s in seqs)

    padded_inps = []
    padded_targets = []

    for s in seqs:
        pad_len = max_len - len(s)
        # Right pad
        p_in = torch.cat([s, torch.full((pad_len,), pad_id, dtype=torch.long)])
        # Target for next-token prediction: shifted tokens
        # Target at position t is token at t+1; at the end, pad with -100 (ignored in loss)
        t_shift = torch.cat([s[1:], torch.tensor([pad_id], dtype=torch.long), torch.full((pad_len,), -100, dtype=torch.long)])
        # Mask out pad tokens from target
        t_shift[t_shift == pad_id] = -100

        padded_inps.append(p_in)
        padded_targets.append(t_shift)

    return (
        torch.stack(padded_inps),
        torch.stack(padded_targets),
        torch.stack(targets),
    )


def main():
    parser = argparse.ArgumentParser(description="Train ChessBot Policy & AlphaZero Value Network")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="runs/chess_bot")
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--model-size", type=str, default="medium", choices=["small", "medium"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=100)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Training ChessBot Policy/Value model on device: {device}")

    tokenizer = ChessTokenizer()
    base_cfg = SMALL if args.model_size == "small" else MEDIUM
    config = copy.deepcopy(base_cfg)
    config.vocab_size = tokenizer.vocab_size
    config.max_seq_len = args.max_len
    config.use_mup = True

    model = ChessBotPolicyValue(config, pad_token_id=tokenizer.pad_id).to(device)
    print(f"ChessBot ({args.model_size}) parameters: {model.count_params():,}")

    data_dir = Path(args.data_dir)
    game_files = [
        data_dir / "standard" / "games_train.jsonl",
        Path("data_large/standard/games_train.jsonl"),
    ]
    dataset = ChessGameTrajectoryDataset(game_files, tokenizer, max_len=args.max_seq_len if hasattr(args, "max_seq_len") else args.max_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_id),
    )

    optimizer = model.get_optimizer(base_lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=1e-6)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    running_tot_loss = 0.0
    running_pol_loss = 0.0
    running_val_mse = 0.0

    model.train()
    print(f"Starting AlphaZero-style training (policy + value MSE) for {args.max_steps} steps...")

    while step < args.max_steps:
        for in_ids, target_ids, target_val in dataloader:
            in_ids = in_ids.to(device)
            target_ids = target_ids.to(device)
            target_val = target_val.to(device)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                total_loss, pol_loss, val_mse = model.compute_loss(
                    in_ids, target_ids, target_val, value_weight=args.value_weight
                )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            running_tot_loss += total_loss.item()
            running_pol_loss += pol_loss.item()
            running_val_mse += val_mse.item()

            if step % args.eval_interval == 0:
                avg_tot = running_tot_loss / args.eval_interval
                avg_pol = running_pol_loss / args.eval_interval
                avg_mse = running_val_mse / args.eval_interval
                running_tot_loss, running_pol_loss, running_val_mse = 0.0, 0.0, 0.0
                print(
                    f"Step {step}/{args.max_steps} | Total Loss={avg_tot:.4f} | "
                    f"Policy NLL={avg_pol:.4f} | Value MSE={avg_mse:.4f} | lr={scheduler.get_last_lr()[0]:.2e}"
                )

            if step >= args.max_steps:
                break

    save_path = out_dir / "chess_bot_policy_value.pt"
    model.save(save_path)
    print(f"Training complete! Saved model to {save_path}")


if __name__ == "__main__":
    main()
