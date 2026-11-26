"""End-to-end tests for Track 2 dataset loader, training, and evaluation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from chess_icl_v2.data.dataset_v2 import (
    EpisodeDatasetV2,
    EvalEpisodeBuilder,
    collate_episode_batch,
)
from chess_icl_v2.data.tokenizer_v2 import ChessV2Tokenizer, STANDARD_STARTING_FEN
from chess_icl_v2.eval.baselines import DemoMajorityBaseline
from chess_icl_v2.eval.controls import (
    make_corrupted_outcome_episodes,
    make_corrupted_trajectory_episodes,
    make_mismatched_context_episodes,
)
from scripts_v2 import eval_track2, train_track2


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _build_synthetic_dataset(root: Path) -> Path:
    """Create a tiny valid Stage-1-contract dataset under `root`."""
    sample_fens = [
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "b", 0, True),
        ("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2", "b", 0, True),
        ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "w", 0, True),
        ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "w", 3, False),
    ]
    demo_moves_pool = [
        ["e2e4", "e7e5", "g1f3", "b8c6"],
        ["d2d4", "d7d5", "c2c4", "e7e6"],
        ["c2c4", "e7e5", "b1c3", "g8f6"],
        ["g1f3", "d7d5", "d2d4", "c7c5"],
        ["e2e4", "c7c5", "g1f3", "d7d6"],
        ["d2d4", "g8f6", "c2c4", "g7g6"],
    ]
    results = ["win", "draw", "loss", "win", "loss", "draw"]
    probs_choices = [
        [0.65, 0.20, 0.15],
        [0.20, 0.55, 0.25],
        [0.15, 0.25, 0.60],
        [0.70, 0.15, 0.15],
    ]

    for v_idx, variant in enumerate(["standard", "diagonal_rook", "blocked_squares"]):
        v_dir = root / variant
        blocked = [27, 36] if variant == "blocked_squares" else None

        demos = []
        for i in range(6):
            demos.append(
                {
                    "game_id": f"{variant}_demo_{i}",
                    "variant": variant,
                    "moves": demo_moves_pool[i],
                    "result": results[(i + v_idx) % len(results)],
                    "init_fen": STANDARD_STARTING_FEN,
                    "blocked_squares": blocked,
                    "policy_id": "terminal_aware_v1",
                    "move_count": len(demo_moves_pool[i]),
                }
            )
        _write_jsonl(v_dir / "games_demo.jsonl", demos)

        for split in ("train", "val", "test"):
            pos_rows = []
            for r_idx, (fen, side, mat_diff, eq_mat) in enumerate(sample_fens):
                pos_rows.append(
                    {
                        "root_id": f"{variant}_{split}_root_{r_idx}",
                        "variant": variant,
                        "fen": fen,
                        "blocked_squares": blocked,
                        "side_to_move": side,
                        "label_probs": probs_choices[r_idx],
                        "source_game_id": f"{variant}_{split}_game_{r_idx // 2}",
                        "ply": 2 + r_idx,
                        "material_diff_white": mat_diff,
                        "equal_material": eq_mat,
                    }
                )
            _write_jsonl(v_dir / f"positions_{split}.jsonl", pos_rows)

    return root


def test_episode_dataset_and_eval_builder(tmp_path: Path):
    data_dir = _build_synthetic_dataset(tmp_path / "data")
    tok = ChessV2Tokenizer()

    # Standard warmup dataset
    ds_warmup = EpisodeDatasetV2(
        data_dir=data_dir,
        split="train",
        mode="standard_warmup",
        k_values=[0, 2, 4],
        k_probs=[0.2, 0.4, 0.4],
        tokenizer=tok,
        max_seq_len=512,
        seed=7,
        length=12,
    )
    assert ds_warmup.variants == ["standard"]
    assert len(ds_warmup) == 12
    ids, segs, lbl = ds_warmup[0]
    assert ids.ndim == 1 and segs.shape == ids.shape and lbl.shape == (3,)

    # Variant mixture dataset + collate_episode_batch
    ds_mix = EpisodeDatasetV2(
        data_dir=data_dir,
        split="train",
        mode="variant_mixture",
        variants=["diagonal_rook", "blocked_squares"],
        k_values=[0, 1, 3],
        tokenizer=tok,
        max_seq_len=512,
        seed=11,
        length=10,
    )
    batch = collate_episode_batch([ds_mix[0], ds_mix[1], ds_mix[2]], pad_id=tok.pad_id)
    assert batch["input_ids"].shape[0] == 3
    assert batch["segment_ids"].shape == batch["input_ids"].shape
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["label_probs"].shape == (3, 3)

    # EvalEpisodeBuilder determinism & baseline/control integration
    builder = EvalEpisodeBuilder(data_dir)
    eps_k0, meta_k0 = builder.build("blocked_squares", "test", k=0, draw_seed=42)
    eps_k4_a, meta_k4_a = builder.build("blocked_squares", "test", k=4, draw_seed=42)
    eps_k4_b, _ = builder.build("blocked_squares", "test", k=4, draw_seed=42)

    assert [m["root_id"] for m in meta_k0] == [m["root_id"] for m in meta_k4_a]
    assert eps_k4_a == eps_k4_b
    assert len(eps_k0[0]["demos"]) == 0
    assert len(eps_k4_a[0]["demos"]) == 4

    # Seamless integration with baselines and controls
    baseline = DemoMajorityBaseline()
    probs = baseline.predict_proba(eps_k4_a)
    assert probs.shape == (len(eps_k4_a), 3)

    pools = {v: builder.get_demo_pool(v) for v in ["diagonal_rook", "blocked_squares"]}
    mismatched = make_mismatched_context_episodes(eps_k4_a, pools, seed=1)
    corrupted_out = make_corrupted_outcome_episodes(eps_k4_a, mode="inverted", seed=2)
    corrupted_traj = make_corrupted_trajectory_episodes(eps_k4_a, seed=3)
    assert len(mismatched) == len(corrupted_out) == len(corrupted_traj) == len(eps_k4_a)


def test_precomputed_eval_episodes_support(tmp_path: Path):
    data_dir = _build_synthetic_dataset(tmp_path / "data")
    builder = EvalEpisodeBuilder(data_dir)
    eps, _ = builder.build("diagonal_rook", "test", k=2, draw_seed=0)
    pre_path = data_dir / "diagonal_rook" / "eval_episodes_2.jsonl"
    _write_jsonl(pre_path, eps)

    loaded_eps, _ = builder.build(
        "diagonal_rook", "test", k=2, draw_seed=0, prefer_precomputed=True
    )
    assert len(loaded_eps) == len(eps)
    assert [e["root_id"] for e in loaded_eps] == [e["root_id"] for e in eps]


@pytest.mark.parametrize("attention_mode", ["causal", "segment_bidirectional"])
def test_train_warmup_continue_and_eval_end_to_end(tmp_path: Path, attention_mode: str):
    data_dir = _build_synthetic_dataset(tmp_path / "data")
    warmup_dir = tmp_path / f"run_warmup_{attention_mode}"
    cont_dir = tmp_path / f"run_continue_{attention_mode}"
    eval_dir = tmp_path / f"eval_{attention_mode}"

    # 1. Warmup on standard chess
    warmup_hist = train_track2.main(
        [
            "--mode",
            "warmup",
            "--config",
            "tiny",
            "--use-mup",
            "--attention-mode",
            attention_mode,
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(warmup_dir),
            "--seed",
            "42",
            "--lr",
            "1e-3",
            "--batch-size",
            "2",
            "--grad-accum",
            "1",
            "--max-steps",
            "4",
            "--warmup-steps",
            "1",
            "--eval-interval",
            "2",
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
            "-1",
        ]
    )
    assert (warmup_dir / "best.pt").exists()
    assert (warmup_dir / "last.pt").exists()
    assert (warmup_dir / "history.json").exists()
    assert len(warmup_hist["eval_steps"]) == 2
    assert "train_nll" in warmup_hist["eval_steps"][-1]
    assert "val_nll" in warmup_hist["eval_steps"][-1]
    assert "val_brier" in warmup_hist["eval_steps"][-1]
    assert "val_acc" in warmup_hist["eval_steps"][-1]

    # 2. Continue on variant mixture initialized from warmup best.pt
    cont_hist = train_track2.main(
        [
            "--mode",
            "continue",
            "--config",
            "tiny",
            "--use-mup",
            "--attention-mode",
            attention_mode,
            "--variants",
            "diagonal_rook",
            "blocked_squares",
            "--init-checkpoint",
            str(warmup_dir / "best.pt"),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(cont_dir),
            "--seed",
            "43",
            "--lr",
            "5e-4",
            "--batch-size",
            "2",
            "--grad-accum",
            "1",
            "--max-steps",
            "4",
            "--warmup-steps",
            "1",
            "--eval-interval",
            "2",
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
            "-1",
        ]
    )
    assert (cont_dir / "best.pt").exists()
    assert (cont_dir / "history.json").exists()
    best_eval_nll = min(e["val_nll"] for e in cont_hist["eval_steps"])
    assert cont_hist["best_val_nll"] == pytest.approx(best_eval_nll)

    # 3. Evaluate best checkpoint with 5 demo draws, controls, equal_material slice, and bootstrap
    summary = eval_track2.main(
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
            "blocked_squares",
            "--k-values",
            "0",
            "2",
            "4",
            "--n-draws",
            "5",
            "--control-k",
            "4",
            "--n-boot",
            "50",
            "--batch-size",
            "4",
            "--gpu",
            "-1",
        ]
    )

    assert (eval_dir / "predictions.jsonl").exists()
    assert (eval_dir / "eval_summary.json").exists()

    assert set(summary["main_by_k"].keys()) == {"0", "2", "4"}
    assert "context_gain" in summary["main_by_k"]["4"]
    assert "macro_mean" in summary["main_by_k"]["4"]["context_gain"]
    assert "ci" in summary["main_by_k"]["4"]["nll"]

    assert set(summary["equal_material_by_k"].keys()) == {"0", "2", "4"}
    assert set(summary["controls_by_k"].keys()) == {
        "mismatched_context",
        "corrupted_outcomes",
        "corrupted_trajectories",
    }
    assert "4" in summary["controls_by_k"]["mismatched_context"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_train_and_eval_smoke(tmp_path: Path):
    data_dir = _build_synthetic_dataset(tmp_path / "data")
    out_dir = tmp_path / "gpu_run"
    eval_dir = tmp_path / "gpu_eval"

    train_track2.main(
        [
            "--mode",
            "warmup",
            "--config",
            "tiny",
            "--attention-mode",
            "segment_bidirectional",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(out_dir),
            "--batch-size",
            "2",
            "--max-steps",
            "2",
            "--eval-interval",
            "2",
            "--k-values",
            "0",
            "2",
            "--val-k-values",
            "0",
            "2",
            "--max-seq-len",
            "1024",
            "--gpu",
            "0",
        ]
    )
    summary = eval_track2.main(
        [
            "--checkpoint",
            str(out_dir / "best.pt"),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(eval_dir),
            "--split",
            "test",
            "--variants",
            "standard",
            "diagonal_rook",
            "--k-values",
            "0",
            "2",
            "--n-draws",
            "2",
            "--n-boot",
            "20",
            "--gpu",
            "0",
        ]
    )
    assert np.isfinite(summary["main_by_k"]["2"]["nll"]["macro_mean"])
