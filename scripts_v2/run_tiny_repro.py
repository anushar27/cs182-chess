"""Tiny end-to-end reproducibility script for chess_icl_v2 (< 60s on CPU or GPU).

Pipeline:
1. Generate a deterministic simulated dataset under `data_v2_tiny/` for
   `standard`, `diagonal_rook`, and `short_knight` (20 demo games + 24 train +
   12 val + 12 test positions each) using `chess_icl_v2.variants` + `ChessV2Tokenizer`.
2. Run Track 2 TINY warmup (`standard`) -> continuation (`diagonal_rook`, `short_knight`).
3. Run `eval_track2.py` (5 demo draws per K, controls, equal-material slice, bootstrap CIs).
4. Run `run_baselines.py` (all 6 non-neural baselines).
5. Run `make_paper_figures.py` to produce PDF/PNG figures and LaTeX tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any, Optional, Sequence

import chess
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chess_icl_v2.data.tokenizer_v2 import ChessV2Tokenizer
from chess_icl_v2.variants import get_board
from scripts_v2 import eval_track2, make_paper_figures, run_baselines, train_track2

REPRO_VARIANTS = ("standard", "diagonal_rook", "short_knight")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _terminal_white_outcome(board: Any) -> str:
    """Resolve White-perspective outcome ('win' | 'draw' | 'loss') at terminal or horizon."""
    outcome = board.outcome(claim_draw=False)
    if outcome is not None:
        if outcome.winner is None:
            return "draw"
        return "win" if outcome.winner == chess.WHITE else "loss"
    mat = board.evaluate_material(chess.WHITE)
    if mat >= 3:
        return "win"
    if mat <= -3:
        return "loss"
    return "draw"


def _rollout_outcome_for_stm(
    variant: str,
    fen: str,
    blocked: list[int],
    rng: random.Random,
    horizon_plies: int,
) -> int:
    """Run one rollout from `fen` and return 0 (win), 1 (draw), or 2 (loss) for root STM."""
    board = get_board(variant, fen=fen, blocked=blocked if blocked else None)
    root_turn = board.turn
    for _ in range(horizon_plies):
        if board.is_game_over(claim_draw=False):
            break
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))

    w_res = _terminal_white_outcome(board)
    if w_res == "draw":
        return 1
    if root_turn == chess.WHITE:
        return 0 if w_res == "win" else 2
    return 2 if w_res == "win" else 0


def _simulate_game_and_trajectory(
    variant: str,
    game_idx: int,
    rng: random.Random,
    max_plies: int,
    tokenizer: ChessV2Tokenizer,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Simulate one game with `chess_icl_v2.variants` and record candidate states."""
    board = get_board(variant, seed=game_idx)
    init_fen = board.fen()
    blocked = sorted(board.blocked)
    moves_uci: list[str] = []
    snapshots: list[dict[str, Any]] = []

    for ply in range(max_plies):
        if board.is_game_over(claim_draw=False):
            break
        legal = list(board.legal_moves)
        if not legal:
            break
        mv = rng.choice(legal)
        uci = mv.uci()
        # Validate every simulated move against ChessV2Tokenizer
        tokenizer.encode_move(uci)
        board.push(mv)
        moves_uci.append(uci)

        if ply >= 3 and not board.is_game_over(claim_draw=False):
            snapshots.append(
                {
                    "fen": board.fen(),
                    "blocked_squares": blocked if blocked else None,
                    "side_to_move": "w" if board.turn == chess.WHITE else "b",
                    "ply": ply + 1,
                    "material_diff_white": int(board.evaluate_material(chess.WHITE)),
                    "source_game_id": f"{variant}_src_{game_idx}",
                }
            )

    demo = {
        "game_id": f"{variant}_demo_{game_idx}",
        "variant": variant,
        "moves": moves_uci,
        "result": _terminal_white_outcome(board),
        "init_fen": init_fen,
        "blocked_squares": blocked if blocked else None,
        "policy_id": "uniform_legal_tiny_v1",
        "move_count": len(moves_uci),
    }
    return demo, snapshots


def generate_tiny_dataset(
    data_dir: Path,
    seed: int = 42,
    n_demos: int = 20,
    n_train: int = 24,
    n_val: int = 12,
    n_test: int = 12,
    n_rollouts: int = 6,
    horizon_plies: int = 24,
) -> None:
    """Generate deterministic `data_v2_tiny/` matching the Stage 1 JSONL contract."""
    tokenizer = ChessV2Tokenizer()
    rng = random.Random(seed)
    n_total_pos = n_train + n_val + n_test

    for variant in REPRO_VARIANTS:
        v_dir = data_dir / variant
        demos: list[dict[str, Any]] = []
        pool_snapshots: list[dict[str, Any]] = []

        # Generate 20 demo games (used as demo context)
        for g_idx in range(n_demos):
            demo, _ = _simulate_game_and_trajectory(
                variant=variant,
                game_idx=g_idx,
                rng=rng,
                max_plies=32,
                tokenizer=tokenizer,
            )
            demos.append(demo)

        # Generate disjoint source games for train/val/test query roots
        src_idx = n_demos
        while len(pool_snapshots) < n_total_pos:
            _, snaps = _simulate_game_and_trajectory(
                variant=variant,
                game_idx=src_idx,
                rng=rng,
                max_plies=30,
                tokenizer=tokenizer,
            )
            if snaps:
                # Take up to 2 well-spaced snapshots per source game so cluster bootstrap has multiple games
                pool_snapshots.append(snaps[0])
                if len(snaps) > 4 and len(pool_snapshots) < n_total_pos:
                    pool_snapshots.append(snaps[min(len(snaps) - 1, 5)])
            src_idx += 1

        selected = pool_snapshots[:n_total_pos]
        split_slices = {
            "train": selected[:n_train],
            "val": selected[n_train : n_train + n_val],
            "test": selected[n_train + n_val :],
        }

        _write_jsonl(v_dir / "games_demo.jsonl", demos)

        for split_name, items in split_slices.items():
            pos_rows: list[dict[str, Any]] = []
            for idx, snap in enumerate(items):
                counts = [0, 0, 0]
                blocked = snap["blocked_squares"] or []
                for _ in range(n_rollouts):
                    cls_idx = _rollout_outcome_for_stm(
                        variant=variant,
                        fen=snap["fen"],
                        blocked=blocked,
                        rng=rng,
                        horizon_plies=horizon_plies,
                    )
                    counts[cls_idx] += 1
                probs = [c / float(n_rollouts) for c in counts]
                # Validate query state encoding with ChessV2Tokenizer
                tokenizer.encode_state(snap["fen"], blocked_squares=snap["blocked_squares"])

                pos_rows.append(
                    {
                        "root_id": f"{variant}_{split_name}_{idx}",
                        "variant": variant,
                        "fen": snap["fen"],
                        "blocked_squares": snap["blocked_squares"],
                        "side_to_move": snap["side_to_move"],
                        "label_probs": probs,
                        "y_true": probs,
                        "R": n_rollouts,
                        "source_game_id": snap["source_game_id"],
                        "ply": snap["ply"],
                        "material_diff_white": snap["material_diff_white"],
                        "equal_material": abs(snap["material_diff_white"]) <= 2,
                    }
                )
            _write_jsonl(v_dir / f"positions_{split_name}.jsonl", pos_rows)


def _export_figure_adapters(
    runs_dir: Path,
    warmup_history: dict[str, Any],
    cont_history: dict[str, Any],
    eval_summary: dict[str, Any],
) -> None:
    """Write lightweight adapter JSONs so make_paper_figures.py renders Track 2 + control plots."""
    warmup_evals = warmup_history.get("eval_steps", [])
    cont_evals = cont_history.get("eval_steps", [])
    warmup_end = warmup_evals[-1]["step"] if warmup_evals else 0

    steps = [e["step"] for e in warmup_evals] + [warmup_end + e["step"] for e in cont_evals]
    train_nll = [e["train_nll"] for e in warmup_evals] + [e["train_nll"] for e in cont_evals]
    val_nll = [e["val_nll"] for e in warmup_evals] + [e["val_nll"] for e in cont_evals]

    (runs_dir / "track2_tiny_history.json").write_text(
        json.dumps(
            {
                "model_name": "track2_tiny",
                "steps": steps,
                "train_nll": train_nll,
                "val_nll": val_nll,
                "warmup_end_step": warmup_end,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    by_k: dict[str, Any] = {}
    for k_str, k_data in eval_summary.get("main_by_k", {}).items():
        nll_s = k_data["nll"]
        brier_s = k_data["brier"]
        gain_s = k_data.get("context_gain")
        side_s = k_data.get("side_calibration", {})
        eq_s = eval_summary.get("equal_material_by_k", {}).get(k_str, {})

        per_var: dict[str, Any] = {}
        for v_name, v_nll in nll_s.get("per_variant", {}).items():
            v_entry: dict[str, Any] = {"nll": v_nll}
            if gain_s and v_name in gain_s.get("per_variant", {}):
                v_entry["context_gain"] = gain_s["per_variant"][v_name]
            per_var[v_name] = v_entry

        by_k[k_str] = {
            "macro": {
                "nll": nll_s["macro_mean"],
                "nll_ci": nll_s["ci"],
                "brier": brier_s["macro_mean"],
                "white_nll": side_s.get("white", {}).get("nll"),
                "black_nll": side_s.get("black", {}).get("nll"),
                "white_ece": side_s.get("white", {}).get("ece"),
                "black_ece": side_s.get("black", {}).get("ece"),
                "equal_material_nll": eq_s.get("nll", {}).get("macro_mean"),
                "equal_material_ci": eq_s.get("nll", {}).get("ci"),
                "context_gain": gain_s["macro_mean"] if gain_s else 0.0,
                "context_gain_ci": gain_s["ci"] if gain_s else [0.0, 0.0],
            },
            "per_variant": per_var,
        }

    (runs_dir / "track2_tiny_eval_test.json").write_text(
        json.dumps(
            {
                "split": eval_summary["split"],
                "models": {
                    "track2_tiny": {
                        "n_params": eval_summary["model_config"]["param_count"],
                        "by_k": by_k,
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    k_ctrl = str(max(int(k) for k in eval_summary["k_values"]))
    clean_k = eval_summary["main_by_k"][k_ctrl]
    eq_clean_k = eval_summary.get("equal_material_by_k", {}).get(k_ctrl, {})
    conditions: dict[str, Any] = {
        "clean": {
            "context_gain": clean_k.get("context_gain", {}).get("macro_mean", 0.0),
            "context_gain_ci": clean_k.get("context_gain", {}).get("ci", [0.0, 0.0]),
            "equal_material_context_gain": eq_clean_k.get("context_gain", {}).get("macro_mean"),
            "equal_material_ci": eq_clean_k.get("context_gain", {}).get("ci"),
        }
    }
    ctrl_name_map = {
        "mismatched_context": "mismatched_context",
        "corrupted_outcomes": "inverted_outcomes",
        "corrupted_trajectories": "corrupted_trajectories",
    }
    for src_name, dst_name in ctrl_name_map.items():
        ctrl_by_k = eval_summary.get("controls_by_k", {}).get(src_name, {})
        if k_ctrl in ctrl_by_k:
            c_entry = ctrl_by_k[k_ctrl]
            conditions[dst_name] = {
                "context_gain": c_entry.get("context_gain", {}).get("macro_mean", 0.0),
                "context_gain_ci": c_entry.get("context_gain", {}).get("ci", [0.0, 0.0]),
            }

    (runs_dir / "controls_eval_test.json").write_text(
        json.dumps({"conditions": conditions}, indent=2),
        encoding="utf-8",
    )


def run_tiny_repro(
    data_dir: Path,
    runs_dir: Path,
    seed: int = 42,
    gpu: int = 0,
    clean: bool = True,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    timings: dict[str, float] = {}

    if clean:
        if data_dir.exists():
            shutil.rmtree(data_dir)
        if runs_dir.exists():
            shutil.rmtree(runs_dir)

    gpu_arg = gpu if (gpu >= 0 and torch.cuda.is_available()) else -1

    # 1. Data generation
    t0 = time.perf_counter()
    generate_tiny_dataset(data_dir=data_dir, seed=seed)
    timings["data_gen_sec"] = round(time.perf_counter() - t0, 3)

    # 2. Track 2 TINY warmup -> continue
    t0 = time.perf_counter()
    warmup_dir = runs_dir / "track2_tiny_warmup"
    warmup_hist = train_track2.main(
        [
            "--mode",
            "warmup",
            "--config",
            "tiny",
            "--use-mup",
            "--attention-mode",
            "causal",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(warmup_dir),
            "--seed",
            str(seed),
            "--lr",
            "1e-3",
            "--batch-size",
            "4",
            "--grad-accum",
            "1",
            "--max-steps",
            "8",
            "--warmup-steps",
            "2",
            "--eval-interval",
            "4",
            "--k-values",
            "0",
            "2",
            "4",
            "--val-k-values",
            "0",
            "4",
            "--max-seq-len",
            "1024",
            "--gpu",
            str(gpu_arg),
        ]
    )
    timings["train_warmup_sec"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    cont_dir = runs_dir / "track2_tiny_continue"
    cont_hist = train_track2.main(
        [
            "--mode",
            "continue",
            "--config",
            "tiny",
            "--use-mup",
            "--attention-mode",
            "causal",
            "--variants",
            "diagonal_rook",
            "short_knight",
            "--init-checkpoint",
            str(warmup_dir / "best.pt"),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(cont_dir),
            "--seed",
            str(seed + 1),
            "--lr",
            "5e-4",
            "--batch-size",
            "4",
            "--grad-accum",
            "1",
            "--max-steps",
            "8",
            "--warmup-steps",
            "2",
            "--eval-interval",
            "4",
            "--k-values",
            "0",
            "2",
            "4",
            "--val-k-values",
            "0",
            "4",
            "--max-seq-len",
            "1024",
            "--gpu",
            str(gpu_arg),
        ]
    )
    timings["train_continue_sec"] = round(time.perf_counter() - t0, 3)

    # 3. Evaluate Track 2 TINY
    t0 = time.perf_counter()
    eval_dir = runs_dir / "eval_track2_tiny"
    eval_summary = eval_track2.main(
        [
            "--checkpoint",
            str(cont_dir / "best.pt"),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(eval_dir),
            "--split",
            "test",
            "--variants",
            "diagonal_rook",
            "short_knight",
            "--k-values",
            "0",
            "2",
            "4",
            "--n-draws",
            "5",
            "--control-k",
            "4",
            "--n-boot",
            "100",
            "--batch-size",
            "8",
            "--seed",
            str(seed),
            "--gpu",
            str(gpu_arg),
        ]
    )
    timings["eval_track2_sec"] = round(time.perf_counter() - t0, 3)

    # 4. Run all 6 non-neural baselines
    t0 = time.perf_counter()
    run_baselines.main(
        [
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(runs_dir),
            "--split",
            "test",
            "--variants",
            "diagonal_rook",
            "short_knight",
            "--k-values",
            "0",
            "2",
            "4",
            "--n-draws",
            "5",
            "--n-boot",
            "100",
            "--R",
            "6",
            "--seed",
            str(seed),
        ]
    )
    timings["run_baselines_sec"] = round(time.perf_counter() - t0, 3)

    # 5. Generate paper figures & LaTeX tables
    t0 = time.perf_counter()
    _export_figure_adapters(runs_dir, warmup_hist, cont_hist, eval_summary)
    fig_dir = runs_dir / "paper_figures"
    make_paper_figures.main(
        [
            "--runs-dir",
            str(runs_dir),
            "--output-dir",
            str(fig_dir),
            "--split",
            "test",
        ]
    )
    timings["make_paper_figures_sec"] = round(time.perf_counter() - t0, 3)
    timings["total_sec"] = round(time.perf_counter() - t_start, 3)

    manifest = {
        "device": f"cuda:{gpu_arg}" if gpu_arg >= 0 else "cpu",
        "variants": list(REPRO_VARIANTS),
        "timings_sec": timings,
        "data_dir": str(data_dir),
        "runs_dir": str(runs_dir),
        "paper_figures_dir": str(fig_dir),
    }
    (runs_dir / "repro_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[run_tiny_repro] Completed full end-to-end v2 pipeline in {timings['total_sec']:.2f}s")
    print(f"[run_tiny_repro] Timings: {json.dumps(timings)}")
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny end-to-end v2 reproducibility runner")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data_v2_tiny")
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs_v2_tiny")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (-1 for CPU)")
    parser.add_argument("--no-clean", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    args = parse_args(argv)
    return run_tiny_repro(
        data_dir=args.data_dir,
        runs_dir=args.runs_dir,
        seed=args.seed,
        gpu=args.gpu,
        clean=not args.no_clean,
    )


if __name__ == "__main__":
    main()
