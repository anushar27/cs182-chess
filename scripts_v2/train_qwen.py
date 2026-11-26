"""Train Track 1 Qwen2.5-7B-Instruct LoRA + 3-way probability head (warmup or continue)."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Literal, Mapping, Optional, Sequence

import chess
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_icl_v2.data.dataset_v2 import (  # noqa: E402
    EvalEpisodeBuilder,
    discover_variants,
    load_jsonl,
    normalize_demo_record,
)
from chess_icl_v2.eval.metrics import (  # noqa: E402
    compute_argmax_accuracy,
    compute_brier,
    compute_nll,
)
from chess_icl_v2.models.qwen_model import (  # noqa: E402
    STANDARD_INITIAL_FEN,
    QwenICLClassifier,
    QwenPromptFormatter,
    get_preferred_attn_implementation,
    load_qwen_tokenizer,
    resolve_qwen_model_path,
)
from chess_icl_v2.variants import ALL_VARIANTS, SINGLE_VARIANTS, TRAIN_VARIANTS, get_board  # noqa: E402


def resolve_variants(
    variants_arg: Optional[Sequence[str]],
    mode: Literal["warmup", "continue"],
) -> list[str]:
    """Expand preset variant group names or validate explicit variant lists."""
    if not variants_arg:
        return ["standard"] if mode == "warmup" else list(TRAIN_VARIANTS)

    resolved: list[str] = []
    for item in variants_arg:
        if item in ("move_variants", "train_variants"):
            resolved.extend(TRAIN_VARIANTS)
        elif item == "all_variants":
            resolved.extend(SINGLE_VARIANTS)
        elif item == "chess_only":
            resolved.append("standard")
        else:
            resolved.append(item)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for v in resolved:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


def _uci_moves_to_san(
    variant: str,
    uci_moves: Sequence[str],
    initial_fen: Optional[str] = None,
    blocked_squares: Optional[Sequence[int]] = None,
) -> list[str]:
    """Convert a sequence of UCI moves into SAN moves using VariantBoard rules."""
    if variant not in ALL_VARIANTS:
        return [str(m) for m in uci_moves]

    fen_arg = None if (not initial_fen or initial_fen == STANDARD_INITIAL_FEN) else initial_fen
    kwargs: dict[str, Any] = {}
    if blocked_squares:
        kwargs["blocked"] = [int(sq) for sq in blocked_squares]
    try:
        board = get_board(variant, fen=fen_arg, **kwargs)
    except TypeError:
        board = get_board(variant, fen=fen_arg)

    san_moves: list[str] = []
    for uci in uci_moves:
        uci_str = str(uci).strip()
        try:
            move = chess.Move.from_uci(uci_str)
            san_moves.append(board.san(move))
            board.push(move)
        except Exception:
            san_moves.append(uci_str)
    return san_moves


def prepare_demo_for_notation(
    demo: Mapping[str, Any],
    variant: str,
    notation: Literal["uci", "san"],
) -> dict[str, Any]:
    """Return a normalized demo dict whose `moves` field matches `notation`.

    Removing stale alternate move keys guarantees downstream control perturbations
    (which rewrite `demo['moves']`) stay consistent with `QwenPromptFormatter`.
    """
    out = normalize_demo_record(demo)
    raw_moves = [str(m) for m in out.get("moves", [])]
    if notation == "san":
        pre_san = out.get("moves_san") or out.get("san_moves")
        if pre_san is not None and len(pre_san) == len(raw_moves):
            san_list = [str(m) for m in pre_san]
        else:
            san_list = _uci_moves_to_san(
                variant=out.get("variant", variant),
                uci_moves=raw_moves,
                initial_fen=out.get("initial_fen"),
                blocked_squares=out.get("blocked_squares"),
            )
        out["moves"] = san_list
    else:
        pre_uci = out.get("moves_uci") or out.get("uci_moves")
        if pre_uci is not None and len(pre_uci) == len(raw_moves):
            out["moves"] = [str(m) for m in pre_uci]
        else:
            out["moves"] = raw_moves

    for stale_key in ("moves_san", "san_moves", "moves_uci", "uci_moves"):
        out.pop(stale_key, None)
    return out


def prepare_episodes_for_notation(
    episodes: Sequence[Mapping[str, Any]],
    notation: Literal["uci", "san"],
) -> list[dict[str, Any]]:
    """Normalize all demo records inside an episode list for `notation`."""
    prepared: list[dict[str, Any]] = []
    for ep in episodes:
        ep_copy = dict(ep)
        variant = str(ep_copy.get("variant", "standard"))
        ep_copy["demos"] = [
            prepare_demo_for_notation(d, variant=variant, notation=notation)
            for d in ep_copy.get("demos", [])
        ]
        prepared.append(ep_copy)
    return prepared


def build_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine decay down to 0."""
    eff_total = max(1, int(total_steps))
    eff_warmup = max(0, min(int(warmup_steps), eff_total))

    def lr_lambda(step: int) -> float:
        if eff_warmup > 0 and step < eff_warmup:
            return float(step + 1) / float(eff_warmup)
        if eff_total <= eff_warmup:
            return 1.0
        progress = float(step - eff_warmup) / float(eff_total - eff_warmup)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(gpu: int) -> torch.device:
    if gpu >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def evaluate_validation(
    model: QwenICLClassifier,
    formatter: QwenPromptFormatter,
    builder: EvalEpisodeBuilder,
    variants: Sequence[str],
    val_k_values: Sequence[int],
    notation: Literal["uci", "san"],
    batch_size: int = 4,
    max_roots: Optional[int] = None,
) -> dict[str, Any]:
    """Compute unweighted macro validation NLL, Brier, and Argmax Acc across variants & K."""
    per_variant_k_nll: dict[str, dict[int, float]] = {}
    per_variant_k_brier: dict[str, dict[int, float]] = {}
    per_variant_k_acc: dict[str, dict[int, float]] = {}

    for variant in variants:
        per_variant_k_nll[variant] = {}
        per_variant_k_brier[variant] = {}
        per_variant_k_acc[variant] = {}
        for k in val_k_values:
            episodes, _ = builder.build(
                variant=variant,
                split="val",
                k=int(k),
                draw_seed=0,
                max_roots=max_roots,
                prefer_precomputed=True,
            )
            prepared = prepare_episodes_for_notation(episodes, notation=notation)
            prompts = [
                formatter.format_prompt(
                    ep["demos"],
                    {
                        "fen": ep["query_fen"],
                        "blocked_squares": ep.get("blocked_squares"),
                    },
                )
                for ep in prepared
            ]
            y_true = np.asarray([ep["y_true"] for ep in prepared], dtype=np.float64)
            p_pred = model.predict_proba(prompts, batch_size=max(1, batch_size))
            mean_nll, _ = compute_nll(p_pred, y_true)
            mean_brier = compute_brier(p_pred, y_true)
            acc = float(compute_argmax_accuracy(p_pred, y_true)["accuracy"])

            per_variant_k_nll[variant][int(k)] = float(mean_nll)
            per_variant_k_brier[variant][int(k)] = float(mean_brier)
            per_variant_k_acc[variant][int(k)] = acc

    macro_by_k: dict[str, dict[str, float]] = {}
    for k in val_k_values:
        k_int = int(k)
        macro_by_k[str(k_int)] = {
            "nll": float(np.mean([per_variant_k_nll[v][k_int] for v in variants])),
            "brier": float(np.mean([per_variant_k_brier[v][k_int] for v in variants])),
            "acc": float(np.mean([per_variant_k_acc[v][k_int] for v in variants])),
        }

    val_nll = float(np.mean([macro_by_k[str(int(k))]["nll"] for k in val_k_values]))
    val_brier = float(np.mean([macro_by_k[str(int(k))]["brier"] for k in val_k_values]))
    val_acc = float(np.mean([macro_by_k[str(int(k))]["acc"] for k in val_k_values]))

    delta_nll: Optional[float] = None
    sorted_ks = sorted(int(k) for k in val_k_values)
    if 0 in sorted_ks and sorted_ks[-1] > 0:
        delta_nll = macro_by_k["0"]["nll"] - macro_by_k[str(sorted_ks[-1])]["nll"]

    return {
        "val_nll": val_nll,
        "val_brier": val_brier,
        "val_acc": val_acc,
        "delta_nll": delta_nll,
        "by_k": macro_by_k,
        "per_variant_nll": {
            v: {str(k): val for k, val in ks.items()} for v, ks in per_variant_k_nll.items()
        },
    }


def _sample_training_microbatch(
    rng: np.random.Generator,
    variants: Sequence[str],
    demos_by_variant: Mapping[str, list[dict[str, Any]]],
    positions_by_variant: Mapping[str, list[dict[str, Any]]],
    k_values: Sequence[int],
    formatter: QwenPromptFormatter,
    batch_size: int,
) -> tuple[list[str], torch.Tensor]:
    prompts: list[str] = []
    targets: list[list[float]] = []

    for _ in range(batch_size):
        variant = str(variants[int(rng.integers(0, len(variants)))])
        pos_pool = positions_by_variant[variant]
        pos = pos_pool[int(rng.integers(0, len(pos_pool)))]
        k = int(k_values[int(rng.integers(0, len(k_values)))])

        demo_pool = demos_by_variant[variant]
        k_eff = min(k, len(demo_pool))
        if k_eff > 0:
            src_gid = pos.get("source_game_id")
            cand = [i for i, d in enumerate(demo_pool) if d.get("game_id") != src_gid]
            if len(cand) < k_eff:
                cand = list(range(len(demo_pool)))
            chosen = rng.choice(cand, size=k_eff, replace=False)
            demos = [demo_pool[int(i)] for i in chosen]
        else:
            demos = []

        prompt = formatter.format_prompt(
            demos,
            {
                "fen": pos["fen"],
                "blocked_squares": pos.get("blocked_squares"),
            },
        )
        prompts.append(prompt)
        targets.append([float(x) for x in pos["label_probs"]])

    return prompts, torch.tensor(targets, dtype=torch.float32)


def run_training(
    *,
    mode: Literal["warmup", "continue"],
    lora_mode: Literal["handoff", "full_lora"],
    notation: Literal["uci", "san"],
    variants: Sequence[str],
    data_dir: Path,
    output_dir: Path,
    init_adapter_dir: Optional[Path] = None,
    seed: int = 42,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.05,
    max_grad_norm: float = 1.0,
    batch_size: int = 1,
    grad_accum: int = 8,
    max_steps: int = 500,
    eval_interval: int = 50,
    k_values: Sequence[int] = (0, 1, 2, 4, 8, 16),
    val_k_values: Sequence[int] = (0, 4),
    val_max_roots: Optional[int] = 100,
    max_seq_len: int = 16384,
    gpu: int = 0,
    model_path: Optional[str] = None,
    save_final: bool = False,
    model: Optional[QwenICLClassifier] = None,
    tokenizer: Any = None,
) -> dict[str, Any]:
    """Execute Track 1 LoRA training loop with periodic macro val_nll checkpointing."""
    set_global_seed(seed)
    device = resolve_device(gpu)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_variants = resolve_variants(variants, mode=mode)
    if not resolved_variants:
        raise ValueError("Resolved variant list cannot be empty")

    if tokenizer is None:
        resolved_path = resolve_qwen_model_path(model_path)
        tokenizer = load_qwen_tokenizer(resolved_path, local_files_only=True)

    if model is None:
        attn_impl = get_preferred_attn_implementation()
        torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        if init_adapter_dir is not None:
            model = QwenICLClassifier.from_adapter_and_head(
                init_adapter_dir,
                tokenizer=tokenizer,
                model_path=model_path,
                torch_dtype=torch_dtype,
                attn_implementation=attn_impl,
                local_files_only=True,
                is_trainable=True,
                device=device,
            )
            model.gradient_checkpointing_enable()
        else:
            model = QwenICLClassifier(
                tokenizer=tokenizer,
                model_path=model_path,
                lora_mode=lora_mode,
                torch_dtype=torch_dtype,
                head_dtype=torch.float32,
                attn_implementation=attn_impl,
                local_files_only=True,
                gradient_checkpointing=True,
            ).to(device)
    else:
        if init_adapter_dir is not None:
            model.load_adapter_and_head(init_adapter_dir)
        model = model.to(device)
        if not getattr(model, "_gradient_checkpointing_enabled", False):
            model.gradient_checkpointing_enable()

    # Load training pools per variant and prepare notation once
    demos_by_variant: dict[str, list[dict[str, Any]]] = {}
    positions_by_variant: dict[str, list[dict[str, Any]]] = {}
    for v in resolved_variants:
        v_dir = data_dir / v
        raw_demos = load_jsonl(v_dir / "games_demo.jsonl")
        raw_train = load_jsonl(v_dir / "positions_train.jsonl")
        if not raw_train:
            raise ValueError(f"Empty positions_train.jsonl for variant {v!r}")
        demos_by_variant[v] = [
            prepare_demo_for_notation(d, variant=v, notation=notation) for d in raw_demos
        ]
        positions_by_variant[v] = raw_train

    eff_max_len = min(int(max_seq_len), int(model.max_position_embeddings))
    formatter = QwenPromptFormatter(
        tokenizer=tokenizer,
        notation=notation,
        max_length=eff_max_len,
    )
    val_builder = EvalEpisodeBuilder(data_dir)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found on QwenICLClassifier")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    warmup_steps = int(round(warmup_ratio * max_steps))
    scheduler = build_cosine_warmup_scheduler(
        optimizer, total_steps=max_steps, warmup_steps=warmup_steps
    )

    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    best_val_nll = float("inf")
    best_step = -1
    best_dir = output_dir / "best_adapter"
    train_losses: list[float] = []

    run_config = {
        "mode": mode,
        "lora_mode": model.lora_mode,
        "notation": notation,
        "variants": resolved_variants,
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "init_adapter_dir": str(init_adapter_dir) if init_adapter_dir else None,
        "seed": int(seed),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "warmup_steps": int(warmup_steps),
        "max_grad_norm": float(max_grad_norm),
        "batch_size": int(batch_size),
        "grad_accum": int(grad_accum),
        "max_steps": int(max_steps),
        "eval_interval": int(eval_interval),
        "k_values": [int(k) for k in k_values],
        "val_k_values": [int(k) for k in val_k_values],
        "val_max_roots": val_max_roots,
        "max_seq_len": eff_max_len,
        "device": str(device),
        "parameter_counts": model.parameter_counts(),
    }
    (output_dir / "train_config.json").write_text(json.dumps(run_config, indent=2))

    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    for step in range(1, max_steps + 1):
        t_step = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for _ in range(grad_accum):
            prompts, target_probs = _sample_training_microbatch(
                rng=rng,
                variants=resolved_variants,
                demos_by_variant=demos_by_variant,
                positions_by_variant=positions_by_variant,
                k_values=k_values,
                formatter=formatter,
                batch_size=batch_size,
            )
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            target_probs = target_probs.to(device)

            with amp_ctx:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    target_probs=target_probs,
                )
                loss = out.loss / float(grad_accum)
            loss.backward()
            step_loss += float(loss.detach().cpu().item())

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
        optimizer.step()
        scheduler.step()
        step_ms = (time.perf_counter() - t_step) * 1000.0
        train_losses.append(step_loss)

        should_eval = (eval_interval > 0 and step % eval_interval == 0) or (step == max_steps)
        if should_eval:
            val_metrics = evaluate_validation(
                model=model,
                formatter=formatter,
                builder=val_builder,
                variants=resolved_variants,
                val_k_values=val_k_values,
                notation=notation,
                batch_size=max(1, batch_size),
                max_roots=val_max_roots if (val_max_roots is not None and val_max_roots > 0) else None,
            )
            is_best = val_metrics["val_nll"] < best_val_nll
            if is_best:
                best_val_nll = float(val_metrics["val_nll"])
                best_step = step
                model.save_adapter_and_head(best_dir)

            record = {
                "step": step,
                "train_loss": float(np.mean(train_losses[-eval_interval:])),
                "last_step_loss": step_loss,
                "lr": float(scheduler.get_last_lr()[0]),
                "step_ms": round(step_ms, 2),
                "is_best": is_best,
                "best_val_nll": best_val_nll,
                "best_step": best_step,
                **val_metrics,
            }
            history.append(record)
            (output_dir / "history.json").write_text(json.dumps(history, indent=2))
            print(
                f"[Step {step:4d}/{max_steps}] train_loss={record['train_loss']:.4f} | "
                f"val_nll={record['val_nll']:.4f} | val_brier={record['val_brier']:.4f} | "
                f"val_acc={record['val_acc']:.4f} | best={best_val_nll:.4f} (@{best_step})"
            )

    if save_final:
        model.save_adapter_and_head(output_dir / "final_adapter")

    return {
        "config": run_config,
        "best_step": best_step,
        "best_val_nll": best_val_nll,
        "best_adapter_dir": str(best_dir),
        "history": history,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Track 1 Qwen2.5-7B-Instruct LoRA classifier")
    parser.add_argument("--mode", choices=["warmup", "continue"], required=True)
    parser.add_argument("--lora-mode", choices=["handoff", "full_lora"], default="handoff")
    parser.add_argument("--notation", choices=["uci", "san"], default="san")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Variant names or presets (move_variants, all_variants, chess_only)",
    )
    parser.add_argument("--init-adapter-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--k-values", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16])
    parser.add_argument("--val-k-values", type=int, nargs="+", default=[0, 4])
    parser.add_argument("--val-max-roots", type=int, default=100)
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--save-final", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    variants = args.variants or (["standard"] if args.mode == "warmup" else list(TRAIN_VARIANTS))
    run_training(
        mode=args.mode,
        lora_mode=args.lora_mode,
        notation=args.notation,
        variants=variants,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        init_adapter_dir=args.init_adapter_dir,
        seed=args.seed,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        k_values=args.k_values,
        val_k_values=args.val_k_values,
        val_max_roots=args.val_max_roots,
        max_seq_len=args.max_seq_len,
        gpu=args.gpu,
        model_path=args.model_path,
        save_final=args.save_final,
    )


if __name__ == "__main__":
    main()
