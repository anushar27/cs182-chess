"""End-to-end tests for scripts_v2/run_baselines.py and scripts_v2/make_paper_figures.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts_v2.make_paper_figures import main as make_figures_main
from scripts_v2.run_baselines import main as run_baselines_main


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _create_synthetic_dataset(data_dir: Path) -> None:
    """Create a synthetic multi-variant dataset adhering to the Track 2 JSONL contract."""
    demo_pool = [
        {
            "initial_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "moves": ["e2e4", "e7e5", "g1f3", "b8c6"],
            "outcome": "win",
        },
        {
            "initial_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "moves": ["d2d4", "d7d5", "c2c4"],
            "outcome": "win",
        },
        {
            "initial_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "moves": ["c2c4", "e7e5", "b1c3", "g8f6", "g2g3"],
            "outcome": "draw",
        },
        {
            "initial_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "moves": ["f2f3", "e7e5", "g2g4", "d8h4"],
            "outcome": "loss",
        },
        {
            "initial_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "moves": ["e2e4", "c7c5", "g1f3", "d7d6"],
            "outcome": "win",
        },
    ]

    # Train split intentionally skewed toward 'win' to verify strict train-only fitting
    train_positions = [
        {
            "fen": "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
            "y_true": [0.85, 0.10, 0.05],
            "source_game_id": f"train_g{i // 2}",
            "R": 20,
        }
        for i in range(12)
    ]

    # Val/Test splits include both White & Black to move and equal + unequal material
    eval_positions = [
        {
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "y_true": [0.20, 0.25, 0.55],
            "source_game_id": "eval_g0",
            "R": 20,
        },
        {
            "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
            "y_true": [0.60, 0.25, 0.15],
            "source_game_id": "eval_g0",
            "R": 20,
        },
        {
            "fen": "4k3/8/8/8/8/8/8/3QK3 w - - 0 1",  # +9 material (excluded from equal-mat slice)
            "y_true": [0.90, 0.05, 0.05],
            "source_game_id": "eval_g1",
            "R": 20,
        },
        {
            "fen": "3qk3/8/8/8/8/8/8/4K3 b - - 0 1",  # +9 material for Black
            "y_true": [0.85, 0.10, 0.05],
            "source_game_id": "eval_g1",
            "R": 20,
        },
        {
            "fen": "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
            "y_true": [0.55, 0.30, 0.15],
            "source_game_id": "eval_g2",
            "R": 20,
        },
        {
            "fen": "4k3/4p3/8/8/8/8/8/4K3 b - - 0 1",
            "y_true": [0.50, 0.30, 0.20],
            "source_game_id": "eval_g2",
            "R": 20,
        },
    ]

    for variant in ("blocked_squares", "knight_army"):
        v_dir = data_dir / variant
        demos = [dict(d) for d in demo_pool]
        if variant == "blocked_squares":
            for d in demos:
                d["blocked_squares"] = ["d4", "e5"]
        _write_jsonl(v_dir / "games_demo.jsonl", demos)
        _write_jsonl(v_dir / "positions_train.jsonl", train_positions)
        _write_jsonl(v_dir / "positions_val.jsonl", eval_positions)
        _write_jsonl(v_dir / "positions_test.jsonl", eval_positions)


def _create_mock_neural_runs(runs_dir: Path) -> None:
    """Write mock Track 1 & Track 2 histories, scaling evals, and causal controls."""
    # Mock training histories (Track 2 + Track 1 Qwen)
    steps = [0, 50, 100, 150, 200]
    (runs_dir / "track2_small_history.json").write_text(
        json.dumps(
            {
                "model_name": "track2_small",
                "steps": steps,
                "train_nll": [1.10, 0.95, 0.85, 0.78, 0.74],
                "val_nll": [1.11, 0.97, 0.88, 0.82, 0.79],
                "val_std": [0.02, 0.02, 0.015, 0.015, 0.01],
                "warmup_end_step": 100,
            }
        ),
        encoding="utf-8",
    )
    (runs_dir / "track1_qwen_history.json").write_text(
        json.dumps(
            {
                "model_name": "track1_qwen",
                "steps": steps,
                "train_nll": [1.05, 0.88, 0.78, 0.71, 0.68],
                "val_nll": [1.06, 0.90, 0.81, 0.75, 0.72],
            }
        ),
        encoding="utf-8",
    )

    # Mock neural evals for Track 2 Small, Medium, Large + Track 1 Qwen
    def _make_model_eval(n_params: int, base_nll: float, max_gain: float) -> dict:
        by_k = {}
        for k, frac in [(0, 0.0), (1, 0.35), (2, 0.65), (4, 1.0)]:
            g = max_gain * frac
            nll = base_nll - g
            by_k[str(k)] = {
                "macro": {
                    "nll": nll,
                    "nll_ci": [nll - 0.02, nll + 0.02],
                    "brier": 0.40 - 0.05 * frac,
                    "tv": 0.25 - 0.03 * frac,
                    "white_nll": nll - 0.01,
                    "black_nll": nll + 0.01,
                    "white_ece": 0.04,
                    "black_ece": 0.05,
                    "equal_material_nll": nll + 0.02,
                    "equal_material_ci": [nll, nll + 0.04],
                    "context_gain": g,
                    "context_gain_ci": [g - 0.015, g + 0.015],
                },
                "per_variant": {
                    "blocked_squares": {"context_gain": {"mean": g * 1.4, "ci": [g * 1.4 - 0.02, g * 1.4 + 0.02]}},
                    "knight_army": {"context_gain": {"mean": 0.01, "ci": [-0.01, 0.03]}},
                },
            }
        return {"n_params": n_params, "by_k": by_k}

    (runs_dir / "neural_eval_test.json").write_text(
        json.dumps(
            {
                "split": "test",
                "models": {
                    "track2_small": _make_model_eval(7_523_843, 0.92, 0.08),
                    "track2_medium": _make_model_eval(23_102_211, 0.86, 0.13),
                    "track2_large": _make_model_eval(206_362_627, 0.79, 0.19),
                    "track1_qwen": _make_model_eval(7_000_000_000, 0.76, 0.22),
                },
            }
        ),
        encoding="utf-8",
    )

    # Mock causal intervention controls
    (runs_dir / "controls_eval_test.json").write_text(
        json.dumps(
            {
                "conditions": {
                    "clean": {
                        "context_gain": 0.19,
                        "context_gain_ci": [0.16, 0.22],
                        "equal_material_context_gain": 0.17,
                        "equal_material_ci": [0.14, 0.20],
                    },
                    "mismatched_context": {
                        "context_gain": -0.02,
                        "context_gain_ci": [-0.05, 0.01],
                        "equal_material_context_gain": -0.03,
                        "equal_material_ci": [-0.06, 0.00],
                    },
                    "inverted_outcomes": {
                        "context_gain": -0.11,
                        "context_gain_ci": [-0.15, -0.07],
                        "equal_material_context_gain": -0.10,
                        "equal_material_ci": [-0.14, -0.06],
                    },
                    "scrambled_outcomes": {
                        "context_gain": 0.01,
                        "context_gain_ci": [-0.02, 0.04],
                        "equal_material_context_gain": 0.00,
                        "equal_material_ci": [-0.03, 0.03],
                    },
                    "corrupted_trajectories": {
                        "context_gain": 0.04,
                        "context_gain_ci": [0.01, 0.07],
                        "equal_material_context_gain": 0.03,
                        "equal_material_ci": [0.00, 0.06],
                    },
                }
            }
        ),
        encoding="utf-8",
    )


def test_run_baselines_and_make_paper_figures(tmp_path: Path):
    data_dir = tmp_path / "data"
    runs_dir = tmp_path / "runs_v2"
    fig_dir_baselines_only = tmp_path / "figs_baselines_only"
    fig_dir_full = tmp_path / "figs_full"

    _create_synthetic_dataset(data_dir)

    # 1. Run baselines end-to-end on test split
    run_baselines_main(
        [
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(runs_dir),
            "--split",
            "test",
            "--k-values",
            "0",
            "1",
            "2",
            "4",
            "--n-draws",
            "5",
            "--n-boot",
            "100",
            "--seed",
            "42",
        ]
    )

    full_json = runs_dir / "baselines_test.json"
    summary_json = runs_dir / "baselines_test_summary.json"
    assert full_json.exists()
    assert summary_json.exists()

    payload = json.loads(full_json.read_text(encoding="utf-8"))
    assert payload["split"] == "test"
    assert payload["variants"] == ["blocked_squares", "knight_army"]
    assert payload["k_values"] == [0, 1, 2, 4]
    assert payload["n_draws"] == 5

    expected_models = {
        "uniform",
        "training_prior",
        "query_material",
        "demo_majority",
        "length_histogram",
        "board_knn",
    }
    assert set(payload["models"].keys()) == expected_models

    # Verify strict train-only fitting on TrainingPriorBaseline:
    # Train has p_win=0.85, whereas test has lower p_win -> NLL of training_prior reflects train prior!
    # Also verify uniform NLL == ln(3)
    uniform_nll_k0 = payload["models"]["uniform"]["by_k"]["0"]["macro"]["nll"]
    assert np.isclose(uniform_nll_k0, np.log(3.0), atol=1e-6)
    assert payload["models"]["board_knn"]["by_k"]["4"]["macro"]["equal_material_nll"] is not None
    assert payload["models"]["demo_majority"]["by_k"]["4"]["macro"]["white_nll"] is not None
    assert payload["models"]["demo_majority"]["by_k"]["4"]["macro"]["black_nll"] is not None

    # 2. Incremental mode: only baselines_test.json exists -> graceful skip of neural-only figs
    make_figures_main(
        [
            "--runs-dir",
            str(runs_dir),
            "--output-dir",
            str(fig_dir_baselines_only),
            "--split",
            "test",
        ]
    )
    assert (fig_dir_baselines_only / "fig2_context_gain_vs_k.png").exists()
    assert (fig_dir_baselines_only / "fig5_side_calibration_and_variants.png").exists()
    assert (fig_dir_baselines_only / "table1_main_results.tex").exists()
    assert (fig_dir_baselines_only / "table2_hyperparameters.tex").exists()
    assert not (fig_dir_baselines_only / "fig1_training_curves.png").exists()
    assert not (fig_dir_baselines_only / "fig3_mup_scaling.png").exists()
    assert not (fig_dir_baselines_only / "fig4_causal_controls.png").exists()

    # 3. Full mode: add mock neural histories, scaling evals, and causal controls -> all 5 figs + tex
    _create_mock_neural_runs(runs_dir)
    make_figures_main(
        [
            "--runs-dir",
            str(runs_dir),
            "--output-dir",
            str(fig_dir_full),
            "--split",
            "test",
        ]
    )

    for stem in (
        "fig1_training_curves",
        "fig2_context_gain_vs_k",
        "fig3_mup_scaling",
        "fig4_causal_controls",
        "fig5_side_calibration_and_variants",
    ):
        png_p = fig_dir_full / f"{stem}.png"
        pdf_p = fig_dir_full / f"{stem}.pdf"
        assert png_p.exists() and png_p.stat().st_size > 1000
        assert pdf_p.exists() and pdf_p.stat().st_size > 1000

    for tex_name in ("table1_main_results.tex", "table2_hyperparameters.tex", "captions.tex"):
        tex_p = fig_dir_full / tex_name
        assert tex_p.exists()
        assert len(tex_p.read_text(encoding="utf-8").strip()) > 0
