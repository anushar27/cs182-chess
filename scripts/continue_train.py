"""Phase 3: Continue training with in-context variant demonstrations."""

import argparse
import json
import sys
from pathlib import Path

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from chess_icl.data import ChessTokenizer
from chess_icl.dataset import ChessVariantDataset
from chess_icl.model import ChessTransformer
from chess_icl.variants import VARIANTS, TRAIN_VARIANTS


def continue_train(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Continue-training config: {args.config}, seed: {args.seed}, device: {device}")

    # Load pretrain checkpoint
    pretrain_ckpt = Path(args.pretrain_checkpoint)
    if not pretrain_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {pretrain_ckpt}")

    model = ChessTransformer.load(pretrain_ckpt, map_location=device).to(device)
    if args.use_mup:
        model.config.use_mup = True
        print(f"muP enabled for continue-training (base_width={model.config.d_base})")
    tokenizer = ChessTokenizer()
    model.pad_token_id = tokenizer.pad_id

    # Determine variant paths based on config:
    # 1. "move_variants"  -> 4 training Move variants
    # 2. "all_variants"   -> all 13 variants (upper bound)
    # 3. "chess_only"     -> standard chess only (lower bound / control)
    data_root = Path(args.data_dir)
    if args.config == "chess_only":
        variant_paths = []
    elif args.config == "move_variants":
        variant_paths = [data_root / v for v in TRAIN_VARIANTS if (data_root / v).exists()]
    elif args.config == "all_variants":
        variant_paths = [data_root / v for v in VARIANTS.keys() if (data_root / v).exists()]
    else:
        raise ValueError(f"Unknown config: {args.config}")

    print(f"Active variant paths ({len(variant_paths)}): {[p.name for p in variant_paths]}")

    dataset = ChessVariantDataset(
        standard_path=data_root / "standard",
        variant_paths=variant_paths,
        tokenizer=tokenizer,
        k_max=args.k_max,
        max_len=args.max_seq_len,
        mix_ratio=0.5 if variant_paths else 1.0,
        dataset_size=args.dataset_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Optimizer with muP parameter groups or standard groups
    optimizer = model.get_optimizer(base_lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=1e-7)
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    log_history = []
    running_loss = 0.0
    running_acc = 0.0

    model.train()
    print(f"Starting continue-training for {args.max_steps} steps...")
    while step < args.max_steps:
        for input_ids, labels in dataloader:
            input_ids, labels = input_ids.to(device), labels.to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            acc = (logits.argmax(-1) == labels).float().mean().item()
            running_loss += loss.item()
            running_acc += acc

            if step % args.log_interval == 0:
                avg_loss = running_loss / args.log_interval
                avg_acc = running_acc / args.log_interval
                running_loss, running_acc = 0.0, 0.0
                print(f"Step {step}/{args.max_steps} | loss={avg_loss:.4f} acc={avg_acc:.3f}")
                log_history.append({"step": step, "loss": avg_loss, "acc": avg_acc})

            if step % args.save_interval == 0 or step >= args.max_steps:
                ckpt_path = out_dir / f"{args.config}_seed{args.seed}.pt"
                model.save(ckpt_path)

            if step >= args.max_steps:
                break

    # Save training log
    with open(out_dir / f"ctrain_{args.config}_seed{args.seed}_log.json", "w") as f:
        json.dump(log_history, f, indent=2)

    final_path = out_dir / f"{args.config}_seed{args.seed}.pt"
    model.save(final_path)
    print(f"Continue-training complete. Saved checkpoint to {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Continue-train Chess Transformer on variants.")
    parser.add_argument("--config", type=str, required=True, choices=["move_variants", "all_variants", "chess_only"])
    parser.add_argument("--pretrain-checkpoint", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-mup", action="store_true", help="Enable/force Maximal Update Parametrization (muP)")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output", type=str, default="runs/variant_train")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--dataset-size", type=int, default=10000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    continue_train(args)


if __name__ == "__main__":
    main()
