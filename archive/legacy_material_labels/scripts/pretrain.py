"""Phase 2: Pretrain the transformer on standard chess evaluation."""

import argparse
import copy
import json
import sys
from pathlib import Path

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from chess_icl.data import ChessTokenizer
from chess_icl.dataset import ChessPretrainDataset
from chess_icl.model import ChessTransformer, TINY, SMALL, MEDIUM, LARGE, XLARGE, ModelConfig


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.inference_mode():
        for input_ids, labels in dataloader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                logits = model(input_ids)
                total_loss += criterion(logits, labels).item() * len(labels)
            correct += (logits.argmax(-1) == labels).sum().item()
            total += len(labels)
    model.train()
    acc = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    return avg_loss, acc


def pretrain(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = ChessTokenizer()

    # Model configuration
    config_map = {"tiny": TINY, "small": SMALL, "medium": MEDIUM, "large": LARGE, "xlarge": XLARGE}
    base_cfg = config_map.get(args.config, SMALL)
    config = copy.deepcopy(base_cfg)
    config.vocab_size = tokenizer.vocab_size
    if args.use_mup:
        config.use_mup = True
        print(f"Enabling Maximal Update Parametrization (muP) with base_width={config.d_base}")

    model = ChessTransformer(config, pad_token_id=tokenizer.pad_id).to(device)
    if getattr(args, "resume", None) and Path(args.resume).exists():
        print(f"Resuming model weights from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    print(f"Model ({args.config}) params: {model.count_params():,}")

    train_path = Path(args.data_dir) / "positions_train.jsonl"
    val_path = Path(args.data_dir) / "positions_val.jsonl"

    train_ds = ChessPretrainDataset(train_path, tokenizer, max_len=args.max_seq_len)
    val_ds = ChessPretrainDataset(val_path, tokenizer, max_len=args.max_seq_len)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = model.get_optimizer(base_lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    best_val_loss = float("inf")
    log_history = []
    mup_tag = "_mup" if args.use_mup else ""

    print(f"Starting pretraining: {len(train_ds)} train positions, {len(val_ds)} val positions.")
    for epoch in range(args.max_epochs):
        model.train()
        for input_ids, labels in train_dl:
            input_ids, labels = input_ids.to(device), labels.to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                logits = model(input_ids)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % args.eval_interval == 0:
                val_loss, val_acc = evaluate(model, val_dl, criterion, device)
                train_acc = (logits.argmax(-1) == labels).float().mean().item()
                print(
                    f"Epoch {epoch+1} | Step {step}/{args.max_steps}: "
                    f"train_loss={loss.item():.4f} train_acc={train_acc:.3f} "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
                )
                log_entry = {
                    "step": step,
                    "epoch": epoch + 1,
                    "train_loss": loss.item(),
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
                log_history.append(log_entry)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_ckpt_path = out_dir / f"chess_base_{args.config}{mup_tag}_seed{args.seed}.pt"
                    model.save(best_ckpt_path)
                    # Also save default alias for backward compatibility if not mup
                    if not args.use_mup and args.config == "small":
                        model.save(out_dir / f"chess_base_seed{args.seed}.pt")
                    print(f"  -> Saved best checkpoint: {best_ckpt_path}")

            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break

    # Save final logs
    with open(out_dir / f"pretrain_log_{args.config}{mup_tag}_seed{args.seed}.json", "w") as f:
        json.dump(log_history, f, indent=2)

    # Save final checkpoint if none saved
    final_ckpt = out_dir / f"chess_base_{args.config}{mup_tag}_seed{args.seed}.pt"
    if not final_ckpt.exists():
        model.save(final_ckpt)
        if not args.use_mup and args.config == "small":
            model.save(out_dir / f"chess_base_seed{args.seed}.pt")
    print(f"Pretraining complete. Checkpoint saved to {final_ckpt}")


def main():
    parser = argparse.ArgumentParser(description="Pretrain Chess Transformer.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="small", choices=["tiny", "small", "medium", "large", "xlarge"])
    parser.add_argument("--use-mup", action="store_true", help="Enable Maximal Update Parametrization (muP)")
    parser.add_argument("--data-dir", type=str, default="data/standard")
    parser.add_argument("--output", type=str, default="runs/pretrain")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume weights from")
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    pretrain(args)


if __name__ == "__main__":
    main()
