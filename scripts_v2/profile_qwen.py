"""Profile Qwen2.5-7B-Instruct (Track 1) on 1 H100 GPU:
1. Verify pinned model config (max_position_embeddings=32768, etc.).
2. Measure UCI vs SAN Qwen token counts over 20 random 200-ply standard chess games.
3. Run CUDA smoke test: 1 optimizer step, save adapter+head, reload into fresh instance,
   and assert deterministic predict_proba (max abs diff < 1e-5).
4. Profile peak GPU memory (MB) and latency/step (ms) for forward+backward+step at
   sequence lengths [4096, 8192, 16384, 24576] with batch_size=1, grad_checkpointing=True,
   Mode A ('handoff') LoRA.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import random
import shutil
import statistics
import sys
import tempfile
import time
from typing import Any

import chess
import numpy as np
import torch

# Ensure repo root is importable when invoked as `python3 scripts_v2/profile_qwen.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_icl_v2.models.qwen_model import (  # noqa: E402
    MODEL_ID,
    REVISION,
    STANDARD_INITIAL_FEN,
    DemoGame,
    QueryPosition,
    QwenICLClassifier,
    QwenPromptFormatter,
    get_preferred_attn_implementation,
    load_qwen_tokenizer,
    resolve_qwen_model_path,
)


def sample_random_200_ply_games(
    num_games: int = 20,
    plies: int = 200,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], int]:
    """Sample `num_games` standard chess games that reach exactly `plies` half-moves."""
    rng = random.Random(seed)
    games: list[dict[str, Any]] = []
    attempts = 0
    while len(games) < num_games:
        attempts += 1
        board = chess.Board()
        uci_moves: list[str] = []
        san_moves: list[str] = []
        for _ in range(plies):
            if board.is_game_over(claim_draw=True):
                break
            legal = list(board.legal_moves)
            if not legal:
                break
            move = rng.choice(legal)
            san_moves.append(board.san(move))
            uci_moves.append(move.uci())
            board.push(move)
        if len(uci_moves) == plies:
            games.append(
                {
                    "initial_fen": STANDARD_INITIAL_FEN,
                    "moves_uci": uci_moves,
                    "moves_san": san_moves,
                    "outcome": "draw",
                }
            )
    return games, attempts


def measure_uci_vs_san_tokens(
    tokenizer: Any,
    num_games: int = 20,
    plies: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare Qwen token counts for UCI vs SAN across 200-ply random games."""
    games, attempts = sample_random_200_ply_games(num_games=num_games, plies=plies, seed=seed)
    fmt_uci = QwenPromptFormatter(tokenizer=tokenizer, notation="uci", max_length=None)
    fmt_san = QwenPromptFormatter(tokenizer=tokenizer, notation="san", max_length=None)

    raw_uci_lens: list[int] = []
    raw_san_lens: list[int] = []
    demo_uci_lens: list[int] = []
    demo_san_lens: list[int] = []

    for g in games:
        uci_str = " ".join(g["moves_uci"])
        san_str = " ".join(g["moves_san"])
        raw_uci_lens.append(len(tokenizer.encode(uci_str, add_special_tokens=False)))
        raw_san_lens.append(len(tokenizer.encode(san_str, add_special_tokens=False)))
        demo_uci_lens.append(
            len(tokenizer.encode(fmt_uci.format_demo(g), add_special_tokens=False))
        )
        demo_san_lens.append(
            len(tokenizer.encode(fmt_san.format_demo(g), add_special_tokens=False))
        )

    def _stats(vals: list[int]) -> dict[str, float]:
        return {
            "mean": round(statistics.mean(vals), 2),
            "stdev": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0,
            "min": float(min(vals)),
            "max": float(max(vals)),
        }

    return {
        "num_games": num_games,
        "plies_per_game": plies,
        "sampling_attempts": attempts,
        "raw_moves_uci": _stats(raw_uci_lens),
        "raw_moves_san": _stats(raw_san_lens),
        "formatted_demo_uci": _stats(demo_uci_lens),
        "formatted_demo_san": _stats(demo_san_lens),
        "san_vs_uci_raw_ratio": round(statistics.mean(raw_san_lens) / statistics.mean(raw_uci_lens), 4),
        "san_vs_uci_demo_ratio": round(
            statistics.mean(demo_san_lens) / statistics.mean(demo_uci_lens), 4
        ),
    }


def run_smoke_test(
    model_path: str,
    tokenizer: Any,
    device: torch.device,
    attn_impl: str,
) -> tuple[QwenICLClassifier, dict[str, Any]]:
    """Run 1 optimizer update step, save adapter+head, reload into fresh instance, verify determinism."""
    print("\n[Smoke Test] Loading Qwen2.5-7B-Instruct (BF16, Mode A 'handoff' LoRA)...")
    t0 = time.perf_counter()
    model = QwenICLClassifier(
        tokenizer=tokenizer,
        model_path=model_path,
        lora_mode="handoff",
        torch_dtype=torch.bfloat16,
        head_dtype=torch.float32,
        attn_implementation=attn_impl,
        local_files_only=True,
        gradient_checkpointing=True,
    ).to(device)
    load_sec = time.perf_counter() - t0

    cfg = model.config
    config_info = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "resolved_path": model_path,
        "architectures": getattr(cfg, "architectures", None),
        "max_position_embeddings": int(cfg.max_position_embeddings),
        "hidden_size": int(cfg.hidden_size),
        "num_hidden_layers": int(cfg.num_hidden_layers),
        "num_attention_heads": int(cfg.num_attention_heads),
        "num_key_value_heads": int(cfg.num_key_value_heads),
        "vocab_size": int(cfg.vocab_size),
        "attn_implementation": attn_impl,
        "load_time_sec": round(load_sec, 2),
        "parameter_counts": model.parameter_counts(),
    }
    assert config_info["max_position_embeddings"] == 32768, (
        f"Expected max_position_embeddings=32768, got {config_info['max_position_embeddings']}"
    )

    formatter = QwenPromptFormatter(tokenizer=tokenizer, notation="san", max_length=4096)
    demos = [
        DemoGame(moves=["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"], outcome="win"),
        DemoGame(moves=["d4", "d5", "c4", "e6", "Nc3", "Nf6"], outcome="draw"),
    ]
    prompts = [
        formatter.format_prompt(
            demos,
            QueryPosition(fen="rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
        ),
        formatter.format_prompt(
            demos[:1],
            QueryPosition(
                fen="rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
                blocked_squares=["d5", "e4"],
            ),
        ),
    ]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    target_probs = torch.tensor([[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]], dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    out = model(input_ids=input_ids, attention_mask=attention_mask, target_probs=target_probs)
    loss_val = float(out.loss.detach().cpu().item())
    assert np.isfinite(loss_val), f"Non-finite smoke test loss: {loss_val}"
    out.loss.backward()

    nonzero_lora_grads = sum(
        1
        for n, p in model.backbone.named_parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum().item() > 0
    )
    head_grad_norm = float(model.classifier.weight.grad.norm().item())
    assert nonzero_lora_grads > 0, "Expected nonzero LoRA gradients"
    assert head_grad_norm > 0, "Expected nonzero classifier head gradient"

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    probs_before = model.predict_proba(prompts, batch_size=2)
    assert probs_before.shape == (2, 3)

    ckpt_dir = Path(tempfile.mkdtemp(prefix="qwen_smoke_"))
    try:
        model.save_adapter_and_head(ckpt_dir)

        # Free first instance to confirm clean fresh-instance reload within 80GB H100 memory
        del optimizer, out, model
        gc.collect()
        torch.cuda.empty_cache()

        reloaded = QwenICLClassifier.from_adapter_and_head(
            ckpt_dir,
            tokenizer=tokenizer,
            model_path=model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            local_files_only=True,
            is_trainable=False,
            device=device,
        )
        probs_after = reloaded.predict_proba(prompts, batch_size=2)
        max_abs_diff = float(np.max(np.abs(probs_before - probs_after)))
        assert max_abs_diff < 1e-5, f"Reload max abs diff {max_abs_diff} >= 1e-5"

        # Re-enable training + grad checkpointing on reloaded instance for seq-len profiling
        for n, p in reloaded.backbone.named_parameters():
            if "lora_" in n:
                p.requires_grad_(True)
        for p in reloaded.classifier.parameters():
            p.requires_grad_(True)
        reloaded.gradient_checkpointing_enable()

        smoke_summary = {
            "passed": True,
            "smoke_loss": round(loss_val, 6),
            "nonzero_lora_grad_tensors_step1": nonzero_lora_grads,
            "head_grad_norm_step1": round(head_grad_norm, 6),
            "probs_before_reload": [[round(float(x), 6) for x in row] for row in probs_before],
            "probs_after_reload": [[round(float(x), 6) for x in row] for row in probs_after],
            "reload_max_abs_diff": max_abs_diff,
        }
        return reloaded, {"config": config_info, "smoke_test": smoke_summary}
    finally:
        shutil.rmtree(ckpt_dir, ignore_errors=True)


def profile_sequence_lengths(
    model: QwenICLClassifier,
    device: torch.device,
    seq_lengths: list[int],
    warmup_steps: int = 1,
    timed_steps: int = 3,
) -> list[dict[str, Any]]:
    """Profile peak GPU memory (MB) and step latency (ms) across sequence lengths."""
    results: list[dict[str, Any]] = []
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    target_probs = torch.tensor([[0.6, 0.2, 0.2]], dtype=torch.float32, device=device)
    model.train()

    for seq_len in seq_lengths:
        gc.collect()
        torch.cuda.empty_cache()

        input_ids = torch.randint(100, 5000, (1, seq_len), dtype=torch.long, device=device)
        attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)

        try:
            for _ in range(warmup_steps):
                optimizer.zero_grad(set_to_none=True)
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    target_probs=target_probs,
                )
                out.loss.backward()
                optimizer.step()
            torch.cuda.synchronize(device)

            torch.cuda.reset_peak_memory_stats(device)
            step_times_ms: list[float] = []
            for _ in range(timed_steps):
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    target_probs=target_probs,
                )
                out.loss.backward()
                optimizer.step()
                torch.cuda.synchronize(device)
                step_times_ms.append((time.perf_counter() - t0) * 1000.0)

            peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)
            results.append(
                {
                    "seq_len": seq_len,
                    "status": "ok",
                    "mean_step_ms": round(statistics.mean(step_times_ms), 2),
                    "min_step_ms": round(min(step_times_ms), 2),
                    "max_step_ms": round(max(step_times_ms), 2),
                    "tokens_per_sec": round(seq_len / (statistics.mean(step_times_ms) / 1000.0), 1),
                    "peak_allocated_mb": round(peak_alloc_mb, 1),
                    "peak_reserved_mb": round(peak_reserved_mb, 1),
                }
            )
            print(
                f"[Profile] seq_len={seq_len:5d} | mean={results[-1]['mean_step_ms']:8.2f} ms | "
                f"peak_alloc={results[-1]['peak_allocated_mb']:8.1f} MB | "
                f"peak_reserved={results[-1]['peak_reserved_mb']:8.1f} MB"
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            results.append({"seq_len": seq_len, "status": f"OOM: {exc}"})
            print(f"[Profile] seq_len={seq_len:5d} | OOM")
        finally:
            del input_ids, attention_mask
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile Qwen2.5-7B-Instruct LoRA on CUDA")
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[4096, 8192, 16384, 24576],
    )
    parser.add_argument("--num-token-games", type=int, default=20)
    parser.add_argument("--plies", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for profile_qwen.py")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    gpu_total_mb = torch.cuda.get_device_properties(device).total_memory / (1024.0 * 1024.0)
    attn_impl = get_preferred_attn_implementation()
    model_path = resolve_qwen_model_path()
    tokenizer = load_qwen_tokenizer(model_path, local_files_only=True)

    print(f"GPU: {gpu_name} ({gpu_total_mb:.1f} MB), attn_implementation={attn_impl}")
    print(f"Model path: {model_path}")

    token_stats = measure_uci_vs_san_tokens(
        tokenizer=tokenizer,
        num_games=args.num_token_games,
        plies=args.plies,
        seed=args.seed,
    )
    print("\n[Tokenization UCI vs SAN]")
    print(json.dumps(token_stats, indent=2))

    model, smoke_and_cfg = run_smoke_test(
        model_path=model_path,
        tokenizer=tokenizer,
        device=device,
        attn_impl=attn_impl,
    )
    print("\n[Config + Smoke Test]")
    print(json.dumps(smoke_and_cfg, indent=2))

    seq_profiles = profile_sequence_lengths(
        model=model,
        device=device,
        seq_lengths=args.seq_lengths,
    )

    report = {
        "gpu": {"name": gpu_name, "total_memory_mb": round(gpu_total_mb, 1)},
        "tokenization_uci_vs_san": token_stats,
        **smoke_and_cfg,
        "sequence_profiles": seq_profiles,
    }
    print("\n=== FULL JSON REPORT ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
