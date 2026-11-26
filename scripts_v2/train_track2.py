"""Train Track 2 ChessTransformerV2 (standard chess warmup -> variant mixture continuation)."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chess_icl_v2.data.dataset_v2 import (
    EpisodeDatasetV2,
    EvalEpisodeBuilder,
    collate_episode_batch,
    discover_variants,
)
from chess_icl_v2.data.tokenizer_v2 import ChessV2Tokenizer
from chess_icl_v2.eval.metrics import (
    compute_argmax_accuracy,
    compute_brier,
    compute_nll,
)
from chess_icl_v2.models.small_transformer import (
    ChessTransformerV2,
    LARGE,
    MEDIUM,
    ModelConfigV2,
    SMALL,
    TINY,
)

PRESET_CONFIGS: dict[str, ModelConfigV2] = {
    "tiny": TINY,
    "small": SMALL,
    "medium": MEDIUM,
    "large": LARGE,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(gpu: Optional[int]) -> torch.device:
    if gpu is not None and gpu >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    if gpu is None and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@torch.no_grad()
def evaluate_validation(
    model: ChessTransformerV2,
    tokenizer: ChessV2Tokenizer,
    builder: EvalEpisodeBuilder,
    variants: Sequence[str],
    val_k_values: Sequence[int],
    max_seq_len: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    max_val_roots: Optional[int] = None,
    use_bf16: bool = False,
) -> dict[str, Any]:
    """Compute macro validation NLL, Brier, and argmax accuracy across variants and K."""
    was_training = model.training
    model.eval()

    per_vk: dict[str, dict[str, float]] = {}
    nlls: list[float] = []
    briers: list[float] = []
    accs: list[float] = []
    by_k_nlls: dict[int, list[float]] = {int(k): [] for k in val_k_values}

    for v in variants:
        for k in val_k_values:
            k_int = int(k)
            episodes, _ = builder.build(
                variant=v,
                split="val",
                k=k_int,
                draw_seed=seed,
                max_roots=max_val_roots,
                prefer_precomputed=False,
            )
            encoded_items = []
            y_list = []
            for ep in episodes:
                enc = tokenizer.encode_episode(
                    demos=ep["demos"],
                    query_state={
                        "fen": ep["query_fen"],
                        "blocked_squares": ep.get("blocked_squares"),
                    },
                    max_len=max_seq_len,
                )
                encoded_items.append(
                    (
                        enc["input_ids"],
                        enc["segment_ids"],
                        ep["y_true"],
                    )
                )
                y_list.append(ep["y_true"])

            probs_chunks: list[np.ndarray] = []
            for start in range(0, len(encoded_items), batch_size):
                batch = collate_episode_batch(
                    encoded_items[start : start + batch_size],
                    pad_id=tokenizer.pad_id,
                    pad_segment_id=-1,
                )
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                segment_ids = batch["segment_ids"].to(device)
                if use_bf16 and device.type == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        probs = model.predict_proba(
                            input_ids,
                            attention_mask=attention_mask,
                            segment_ids=segment_ids,
                        )
                else:
                    probs = model.predict_proba(
                        input_ids,
                        attention_mask=attention_mask,
                        segment_ids=segment_ids,
                    )
                probs_chunks.append(probs.float().cpu().numpy())

            p_pred = np.vstack(probs_chunks) if probs_chunks else np.empty((0, 3))
            y_true = np.asarray(y_list, dtype=np.float64)

            mean_nll, _ = compute_nll(p_pred, y_true)
            mean_brier = compute_brier(p_pred, y_true)
            acc = compute_argmax_accuracy(p_pred, y_true)["accuracy"]

            key = f"{v}/k={k_int}"
            per_vk[key] = {
                "nll": float(mean_nll),
                "brier": float(mean_brier),
                "acc": float(acc),
                "n_roots": int(len(y_true)),
            }
            nlls.append(float(mean_nll))
            briers.append(float(mean_brier))
            accs.append(float(acc))
            by_k_nlls[k_int].append(float(mean_nll))

    if was_training:
        model.train()

    val_nll = float(np.mean(nlls)) if nlls else float("inf")
    val_brier = float(np.mean(briers)) if briers else float("inf")
    val_acc = float(np.mean(accs)) if accs else 0.0
    per_k_macro_nll = {
        str(k): float(np.mean(vals)) for k, vals in by_k_nlls.items() if vals
    }

    return {
        "val_nll": val_nll,
        "val_brier": val_brier,
        "val_acc": val_acc,
        "per_k_nll": per_k_macro_nll,
        "per_variant_k": per_vk,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Track 2 ChessTransformerV2")
    parser.add_argument("--mode", choices=["warmup", "continue"], default="warmup")
    parser.add_argument(
        "--config", choices=["tiny", "small", "medium", "large"], default="small"
    )
    parser.add_argument(
        "--use-mup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Maximal Update Parametrization (muP)",
    )
    parser.add_argument(
        "--attention-mode",
        choices=["causal", "segment_bidirectional"],
        default="causal",
    )
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--val-variants", nargs="+", default=None)
    parser.add_argument("--init-checkpoint", type=str, default=None)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--k-values", nargs="+", type=int, default=[0, 1, 2, 4, 8, 16])
    parser.add_argument("--k-probs", nargs="+", type=float, default=None)
    parser.add_argument("--val-k-values", nargs="+", type=int, default=[0, 4])
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--max-val-roots", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use bfloat16 autocast on CUDA",
    )
    return parser


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    device = resolve_device(args.gpu)
    use_bf16 = bool(args.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    tokenizer = ChessV2Tokenizer()

    if args.mode == "warmup":
        train_variants = args.variants if args.variants is not None else ["standard"]
        ds_mode = "standard_warmup" if train_variants == ["standard"] else "variant_mixture"
    else:
        train_variants = (
            args.variants
            if args.variants is not None
            else discover_variants(data_dir, split="train")
        )
        ds_mode = "variant_mixture"

    val_variants = args.val_variants if args.val_variants is not None else train_variants

    # Build model (optionally initializing weights from --init-checkpoint)
    cfg = copy.deepcopy(PRESET_CONFIGS[args.config])
    cfg.vocab_size = tokenizer.vocab_size
    cfg.max_seq_len = int(args.max_seq_len)
    cfg.attention_mode = args.attention_mode
    cfg.use_mup = bool(args.use_mup)

    if args.init_checkpoint is not None:
        init_model = ChessTransformerV2.load(args.init_checkpoint, map_location="cpu")
        if (
            init_model.config.d_model != cfg.d_model
            or init_model.config.n_layers != cfg.n_layers
            or init_model.config.n_heads != cfg.n_heads
            or init_model.config.vocab_size != cfg.vocab_size
        ):
            raise ValueError(
                f"Checkpoint config mismatch: init={init_model.config} vs requested={cfg}"
            )
        model = ChessTransformerV2(cfg, pad_token_id=tokenizer.pad_id)
        model.load_state_dict(init_model.state_dict())
    else:
        model = ChessTransformerV2(cfg, pad_token_id=tokenizer.pad_id)

    model.to(device)
    optimizer = model.get_optimizer(base_lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(step: int) -> float:
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return float(step + 1) / float(args.warmup_steps)
        if args.max_steps <= args.warmup_steps:
            return 1.0
        progress = float(step - args.warmup_steps) / float(
            max(1, args.max_steps - args.warmup_steps)
        )
        progress = min(max(progress, 0.0), 1.0)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_dataset = EpisodeDatasetV2(
        data_dir=data_dir,
        split="train",
        mode=ds_mode,
        variants=train_variants,
        k_values=args.k_values,
        k_probs=args.k_probs,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        length=max(args.batch_size * args.grad_accum * 4, 256),
        standard_variant_name=train_variants[0],
    )

    def _collate(batch):
        return collate_episode_batch(batch, pad_id=tokenizer.pad_id, pad_segment_id=-1)

    eval_builder = EvalEpisodeBuilder(data_dir)

    history: dict[str, Any] = {
        "mode": args.mode,
        "config": args.config,
        "use_mup": bool(args.use_mup),
        "attention_mode": args.attention_mode,
        "train_variants": list(train_variants),
        "val_variants": list(val_variants),
        "param_count": model.count_params(),
        "train_steps": [],
        "eval_steps": [],
        "best_step": 0,
        "best_val_nll": float("inf"),
    }

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    best_val_nll = float("inf")
    epoch = 0
    train_dataset.set_epoch(epoch)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate,
    )
    data_iter = iter(loader)

    def next_batch() -> dict[str, torch.Tensor]:
        nonlocal epoch, loader, data_iter
        try:
            return next(data_iter)
        except StopIteration:
            epoch += 1
            train_dataset.set_epoch(epoch)
            data_iter = iter(loader)
            return next(data_iter)

    model.train()
    running_nll = 0.0
    running_count = 0
    t0 = time.time()

    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss_val = 0.0

        for _ in range(args.grad_accum):
            batch = next_batch()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            segment_ids = batch["segment_ids"].to(device)
            label_probs = batch["label_probs"].to(device)

            if use_bf16:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(
                        input_ids,
                        attention_mask=attention_mask,
                        segment_ids=segment_ids,
                    )
                    loss = model.compute_loss(logits, label_probs) / float(args.grad_accum)
            else:
                logits = model(
                    input_ids,
                    attention_mask=attention_mask,
                    segment_ids=segment_ids,
                )
                loss = model.compute_loss(logits, label_probs) / float(args.grad_accum)

            loss.backward()
            step_loss_val += float(loss.detach().item())

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        scheduler.step()

        running_nll += step_loss_val
        running_count += 1

        history["train_steps"].append(
            {
                "step": step,
                "train_nll": step_loss_val,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        should_eval = (step % args.eval_interval == 0) or (step == args.max_steps)
        if should_eval:
            avg_train_nll = running_nll / max(1, running_count)
            running_nll = 0.0
            running_count = 0

            val_res = evaluate_validation(
                model=model,
                tokenizer=tokenizer,
                builder=eval_builder,
                variants=val_variants,
                val_k_values=args.val_k_values,
                max_seq_len=args.max_seq_len,
                batch_size=args.batch_size,
                device=device,
                seed=args.seed,
                max_val_roots=args.max_val_roots,
                use_bf16=use_bf16,
            )

            eval_entry = {
                "step": step,
                "train_nll": float(avg_train_nll),
                "val_nll": val_res["val_nll"],
                "val_brier": val_res["val_brier"],
                "val_acc": val_res["val_acc"],
                "per_k_nll": val_res["per_k_nll"],
                "per_variant_k": val_res["per_variant_k"],
                "elapsed_sec": round(time.time() - t0, 2),
            }
            history["eval_steps"].append(eval_entry)

            if val_res["val_nll"] < best_val_nll:
                best_val_nll = val_res["val_nll"]
                history["best_step"] = step
                history["best_val_nll"] = best_val_nll
                model.save(output_dir / "best.pt")

            print(
                f"[step {step:>5d}/{args.max_steps}] "
                f"train_nll={avg_train_nll:.4f} val_nll={val_res['val_nll']:.4f} "
                f"val_brier={val_res['val_brier']:.4f} val_acc={val_res['val_acc']:.4f} "
                f"(best={best_val_nll:.4f} @ {history['best_step']})"
            )

    model.save(output_dir / "last.pt")
    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return history


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return train(args)


if __name__ == "__main__":
    main()
