"""Comprehensive tests for long-study evaluation, 10% token-matched foreign controls,
7 declared non-neural baselines, full-val checkpoint selection, canonical caching,
cohort & regime reporting, and outcome-independent R256 subset selection.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pytest
import torch
import torch.nn as nn

from chess_icl_v2.eval.baselines import (
    DECLARED_BASELINE_NAMES,
    BoardKNNFinalBaseline,
    BoardKNNTrajectoriesBaseline,
    build_declared_baselines,
)
from chess_icl_v2.eval.controls import (
    make_corrupted_outcome_episodes,
    make_corrupted_trajectory_episodes,
    make_token_matched_foreign_episodes,
)
from scripts_v2.eval_long_study import (
    CanonicalQwenPredictor,
    aggregate_multi_seed_evaluations,
    compute_categorical_brier_per_sample,
    compute_checkpoint_sha256,
    compute_snapshot_sha256,
    discover_candidate_checkpoints,
    evaluate_from_manifest,
    main as eval_long_study_main,
    run_declared_baselines,
    select_checkpoint,
    select_r256_subset,
    verify_model_frozen,
)


# ---------------------------------------------------------------------------
# Lightweight Mock Tokenizer & Mock Frozen Model for CPU Tests
# ---------------------------------------------------------------------------


class _WhitespaceMockTokenizer:
    """Deterministic word-level tokenizer supporting HF encode() and __call__() APIs."""

    def __init__(self) -> None:
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.padding_side = "right"

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = text.split()
        ids = [
            10 + (int.from_bytes(hashlib.sha256(tok.encode("utf-8")).digest()[:2], "big") % 5000)
            for tok in tokens
        ]
        if add_special_tokens:
            ids = ids + [1]
        return ids

    def __call__(
        self,
        texts: str | Sequence[str],
        return_tensors: str | None = None,
        padding: bool = False,
        truncation: bool = False,
        add_special_tokens: bool = True,
    ) -> dict[str, Any]:
        is_single = isinstance(texts, str)
        batch = [texts] if is_single else list(texts)
        encoded = [self.encode(t, add_special_tokens=add_special_tokens) for t in batch]
        if padding and len(encoded) > 1:
            max_l = max(len(x) for x in encoded)
            padded = [x + [0] * (max_l - len(x)) for x in encoded]
            masks = [[1] * len(x) + [0] * (max_l - len(x)) for x in encoded]
        else:
            padded = encoded
            masks = [[1] * len(x) for x in encoded]

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        if is_single:
            return {"input_ids": padded[0], "attention_mask": masks[0]}
        return {"input_ids": padded, "attention_mask": masks}


class _MockFrozenClassifier(nn.Module):
    """Tiny deterministic nn.Module returning (B, 3) probs and recording batch shapes."""

    def __init__(self, bias_logits: Sequence[float] = (0.0, 0.0, 0.0), jitter: float = 0.0):
        super().__init__()
        self.emb = nn.Embedding(5020, 8)
        self.head = nn.Linear(8, 3)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor(list(bias_logits), dtype=torch.float32))
        self.jitter = float(jitter)
        self.forward_calls: list[tuple[int, int, bool]] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        has_pad = bool(attention_mask is not None and torch.any(attention_mask == 0).item())
        self.forward_calls.append((int(input_ids.shape[0]), int(input_ids.shape[1]), has_pad))
        emb = self.emb(input_ids).mean(dim=1)
        logits = self.head(emb)
        if self.jitter > 0.0:
            logits = logits + self.jitter * torch.randn_like(logits)
        probs = torch.softmax(logits, dim=-1)
        return {"logits": logits, "probs": probs}


def _save_mock_checkpoint(ckpt_dir: Path, model: nn.Module) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 16}))
    torch.save(model.state_dict(), ckpt_dir / "classifier_head.pt")


def _load_mock_checkpoint(ckpt_dir: Path, bias_hint: Sequence[float] = (0.0, 0.0, 0.0)) -> _MockFrozenClassifier:
    m = _MockFrozenClassifier(bias_logits=bias_hint)
    sd = torch.load(ckpt_dir / "classifier_head.pt", map_location="cpu", weights_only=True)
    m.load_state_dict(sd)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


# ---------------------------------------------------------------------------
# Synthetic Dataset Helper
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(dict(r)) + "\n")


def _build_synthetic_long_study_dataset(
    data_dir: Path,
    *,
    include_split_demos: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Create dataset spanning standard, familiar TRAIN_VARIANTS, and held-out TEST_VARIANTS."""
    variants = ["standard", "diagonal_rook", "short_knight", "antichess"]

    # Base demos of similar lengths (around 6-10 moves) across variants
    base_demos = [
        {
            "game_id": f"d_{i}",
            "initial_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"][: 4 + (i % 3)],
            "outcome": ["win", "draw", "loss"][i % 3],
        }
        for i in range(8)
    ]

    # Train positions skewed strongly toward win ([0.85, 0.10, 0.05]) to test zero-leakage fit
    train_positions = [
        {
            "root_id": f"train_r{i}",
            "family_id": f"train_fam_{i // 2}",
            "source_game_id": f"train_fam_{i // 2}",
            "fen": "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
            "side_to_move": "w",
            "label_probs": [0.85, 0.10, 0.05],
            "R": 20,
            "equal_material": True,
        }
        for i in range(6)
    ]

    # Val/Test positions with soft targets, both sides, multiple families, equal & unequal material
    eval_template = [
        ("fam_A", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "b", [0.20, 0.30, 0.50], True),
        ("fam_A", "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1", "w", [0.60, 0.25, 0.15], True),
        ("fam_B", "4k3/8/8/8/8/8/8/3QK3 w - - 0 1", "w", [0.80, 0.10, 0.10], False),
        ("fam_B", "3qk3/8/8/8/8/8/8/4K3 b - - 0 1", "b", [0.75, 0.15, 0.10], False),
        ("fam_C", "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", "w", [0.50, 0.30, 0.20], True),
        ("fam_C", "4k3/4p3/8/8/8/8/8/4K3 b - - 0 1", "b", [0.45, 0.35, 0.20], True),
    ]

    test_by_var: dict[str, list[dict[str, Any]]] = {}
    for v in variants:
        v_dir = data_dir / v
        demos_train = [{**d, "game_id": f"{v}_tr_{i}"} for i, d in enumerate(base_demos)]
        demos_val = [{**d, "game_id": f"{v}_va_{i}"} for i, d in enumerate(base_demos)]
        demos_test = [{**d, "game_id": f"{v}_te_{i}"} for i, d in enumerate(base_demos)]

        if include_split_demos:
            _write_jsonl(v_dir / "games_demo_train.jsonl", demos_train)
            _write_jsonl(v_dir / "games_demo_val.jsonl", demos_val)
            _write_jsonl(v_dir / "games_demo_test.jsonl", demos_test)
        _write_jsonl(v_dir / "games_demo.jsonl", demos_train)

        _write_jsonl(v_dir / "positions_train.jsonl", train_positions)

        val_pos = [
            {
                "root_id": f"{v}_val_{i}",
                "variant": v,
                "family_id": f"{v}_{fam}",
                "source_game_id": f"{v}_{fam}",
                "fen": fen,
                "side_to_move": stm,
                "label_probs": probs,
                "R": 20,
                "equal_material": eq,
            }
            for i, (fam, fen, stm, probs, eq) in enumerate(eval_template)
        ]
        test_pos = [
            {
                "root_id": f"{v}_test_{i}",
                "variant": v,
                "family_id": f"{v}_{fam}",
                "source_game_id": f"{v}_{fam}",
                "fen": fen,
                "side_to_move": stm,
                "label_probs": probs,
                "R": 20,
                "equal_material": eq,
            }
            for i, (fam, fen, stm, probs, eq) in enumerate(eval_template)
        ]
        _write_jsonl(v_dir / "positions_val.jsonl", val_pos)
        _write_jsonl(v_dir / "positions_test.jsonl", test_pos)
        test_by_var[v] = test_pos

    return test_by_var


# ---------------------------------------------------------------------------
# 1. 10% Token-Length Matched Foreign Control Tests
# ---------------------------------------------------------------------------


def test_token_matched_foreign_control_success_and_failure_audit():
    tok = _WhitespaceMockTokenizer()

    def _make_demo(n_moves: int, outcome: str) -> dict[str, Any]:
        return {
            "initial_fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "moves": ["e2e4"] * n_moves,
            "outcome": outcome,
        }

    episodes = [
        {
            "root_id": "ep_matchable",
            "variant": "diagonal_rook",
            "query_fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "y_true": [0.6, 0.2, 0.2],
            "demos": [_make_demo(40, "win"), _make_demo(60, "draw")],
        },
        {
            "root_id": "ep_unmatchable",
            "variant": "diagonal_rook",
            "query_fen": "4k3/8/8/8/8/8/8/4K3 b - - 0 1",
            "y_true": [0.3, 0.3, 0.4],
            "demos": [_make_demo(300, "loss"), _make_demo(320, "win")],
        },
    ]
    orig_snapshot = copy.deepcopy(episodes)

    foreign_pools = {
        "diagonal_rook": [_make_demo(300, "win"), _make_demo(320, "loss")],  # same variant: must be excluded!
        "short_knight": [
            _make_demo(38, "loss"),  # within 10% of 40-move demo
            _make_demo(62, "win"),   # within 10% of 60-move demo
            _make_demo(5, "draw"),
        ],
        "antichess": [
            _make_demo(10, "draw"),
        ],
    }

    matched, report = make_token_matched_foreign_episodes(
        episodes=episodes,
        foreign_demos_by_variant=foreign_pools,
        tokenizer=tok,
        notation="san",
        tolerance=0.10,
        max_search_attempts=50,
        seed=42,
    )

    assert episodes == orig_snapshot  # never mutate inputs
    assert report["n_total"] == 2
    assert report["n_matched"] == 1
    assert report["n_failed"] == 1
    assert np.isclose(report["failure_rate"], 0.5)
    assert "does not prove" in report["note"].lower()

    # Check matched episode meets BOTH per-demo +/-10% and total context +/-10%
    assert len(matched) == 1
    m_ep = matched[0]
    assert m_ep["root_id"] == "ep_matchable"
    assert m_ep["match_found"] is True
    assert m_ep["match_failure_reason"] is None
    assert m_ep["control"] == "token_matched_foreign"
    assert m_ep["control_source_variant"] == "short_knight"
    for o_tok, c_tok in zip(m_ep["orig_demo_tokens"], m_ep["matched_demo_tokens"]):
        assert 0.90 * o_tok - 1e-9 <= c_tok <= 1.10 * o_tok + 1e-9
    assert (
        0.90 * m_ep["orig_context_tokens"] - 1e-9
        <= m_ep["matched_context_tokens"]
        <= 1.10 * m_ep["orig_context_tokens"] + 1e-9
    )

    # Check failed episode status explicitly recorded
    status_by_id = {s["root_id"]: s for s in report["per_episode_status"]}
    assert status_by_id["ep_matchable"]["match_found"] is True
    assert status_by_id["ep_unmatchable"]["match_found"] is False
    assert status_by_id["ep_unmatchable"]["match_failure_reason"] == "insufficient_eligible_foreign_demos"

    # Also test bounded_search_exhausted_without_match when context tolerance fails under custom tokenizer
    class _WeirdContextTokenizer(_WhitespaceMockTokenizer):
        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            # Single demos ([Demo] only appears once) match; multi-demo context inflates length for foreign move 'a2a3'
            if add_special_tokens and "a2a3" in text:
                return [1] * 2000
            return super().encode(text, add_special_tokens=add_special_tokens)

    matched_fail, report_fail = make_token_matched_foreign_episodes(
        episodes=[episodes[0]],
        foreign_demos_by_variant={
            "short_knight": [
                {"initial_fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1", "moves": ["a2a3"] * 40, "outcome": "win"},
                {"initial_fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1", "moves": ["a2a3"] * 60, "outcome": "loss"},
            ]
        },
        tokenizer=_WeirdContextTokenizer(),
        notation="san",
        tolerance=0.10,
        max_search_attempts=3,
        seed=7,
    )
    assert matched_fail == []
    assert report_fail["n_failed"] == 1
    assert (
        report_fail["per_episode_status"][0]["match_failure_reason"]
        == "bounded_search_exhausted_without_match"
    )
    assert report_fail["per_episode_status"][0]["attempts_used"] == 3


def test_corrupted_outcomes_and_shuffled_trajectories_compatibility():
    eps = [
        {
            "variant": "short_knight",
            "query_fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "demos": [
                {"moves": ["e2e4", "e7e5", "g1f3"], "outcome": "win"},
                {"moves": ["d2d4", "d7d5", "c2c4"], "outcome": "loss"},
            ],
        }
    ]
    for mode in ("inverted", "scrambled", "all_draw"):
        out = make_corrupted_outcome_episodes(eps, mode=mode, seed=42)
        assert len(out) == 1
    shuf = make_corrupted_trajectory_episodes(eps, mode="shuffled", seed=42)
    assert len(shuf[0]["demos"][0]["moves"]) == 3
    assert shuf[0]["demos"][0]["outcome"] == "win"


# ---------------------------------------------------------------------------
# 2. Seven Declared Baselines Training-Only Fit & Evaluation
# ---------------------------------------------------------------------------


def test_seven_declared_baselines_train_only_fit_and_eval(tmp_path: Path):
    data_dir = tmp_path / "data"
    _build_synthetic_long_study_dataset(data_dir, include_split_demos=True)

    # Verify BoardKNNTrajectoriesBaseline vs BoardKNNFinalBaseline behavior
    demo_traj = {
        "initial_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "moves": ["e2e4", "e7e5"],
        "outcome": "win",
    }
    ep_mid = {
        "query_fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "demos": [demo_traj],
    }
    knn_traj = BoardKNNTrajectoriesBaseline(top_k=1, alpha=0.0)
    knn_final = BoardKNNFinalBaseline(top_k=1, alpha=0.0)
    assert knn_traj.state_mode == "trajectories"
    assert knn_final.state_mode == "final"
    # Mid-trajectory state exact match for Black-to-move -> White 'win' flips to Black 'loss' (index 2 = 1.0)
    assert np.isclose(knn_traj.predict_proba([ep_mid])[0, 2], 1.0)

    # Fit declared baselines on synthetic train episodes
    train_eps = [
        {
            "query_fen": "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
            "demos": [demo_traj],
            "y_true": [0.9, 0.05, 0.05],
        }
        for _ in range(4)
    ]
    declared = build_declared_baselines(train_eps)
    assert tuple(declared.keys()) == DECLARED_BASELINE_NAMES
    assert len(declared) == 7

    # Run full run-baselines subcommand on test split
    out_file = tmp_path / "baselines_long_study_test.json"
    res = run_declared_baselines(
        data_dir=data_dir,
        output_file=out_file,
        split="test",
        k_values=[0, 1, 2, 4],
        num_draws=2,
        n_boot=50,
        seed=42,
    )
    assert out_file.exists()
    assert set(res["baselines"].keys()) == set(DECLARED_BASELINE_NAMES)
    assert len(res["baselines"]) == 7

    # Verify training_prior did NOT leak test/val labels (train has 0.85 win prior!)
    # Uniform NLL at K=0 must be ln(3)
    uni_heldout = res["baselines"]["uniform"]["feasible_cohort"]["heldout_transfer"]["clean"]["0"]
    assert np.isclose(uni_heldout["nll"]["macro_mean"], np.log(3.0), atol=1e-6)
    assert uni_heldout["brier_vector"]["macro_mean"] > 0.0
    assert uni_heldout["brier_categorical"]["macro_mean"] > 0.0
    assert uni_heldout["mc_noise_floor"]["macro"]["entropy_floor"] > 0.0

    # Check regime separation in baselines output
    fam_block = res["baselines"]["board_knn_final"]["feasible_cohort"]["familiar_adaptation"]
    tr_block = res["baselines"]["board_knn_final"]["feasible_cohort"]["heldout_transfer"]
    std_block = res["baselines"]["board_knn_final"]["feasible_cohort"]["standard_control"]
    assert fam_block["is_transfer"] is False
    assert tr_block["is_transfer"] is True
    assert std_block["is_transfer"] is False


# ---------------------------------------------------------------------------
# 3. Full-Val Checkpoint Selection, Canonical Caching, Cohorts & Controls
# ---------------------------------------------------------------------------


def test_val_selection_canonical_eval_caching_cohorts_and_regimes(tmp_path: Path):
    data_dir = tmp_path / "data"
    _build_synthetic_long_study_dataset(data_dir, include_split_demos=True)
    tok = _WhitespaceMockTokenizer()

    # Create 3 candidate checkpoints: step_500 (bad), step_1000 (best on val!), step_1500 (medium)
    ckpts_dir = tmp_path / "checkpoints"
    bias_map = {
        "step_500": (-2.0, -2.0, 2.0),   # predicts mostly loss -> bad NLL on val
        "step_1000": (0.4, -0.1, -0.3),  # well-aligned with val -> best NLL
        "step_1500": (-0.5, 1.5, -0.5),  # predicts mostly draw -> worse NLL
    }
    for name, bias in bias_map.items():
        _save_mock_checkpoint(ckpts_dir / name, _MockFrozenClassifier(bias_logits=bias))

    def _loader(p: Path) -> _MockFrozenClassifier:
        return _load_mock_checkpoint(p, bias_hint=bias_map[p.name])

    manifest_path = tmp_path / "val_selection_manifest.json"
    manifest = select_checkpoint(
        data_dir=data_dir,
        output_manifest=manifest_path,
        checkpoints_dir=ckpts_dir,
        selection_regime="familiar",
        k_values=[0, 1, 2],
        num_draws=2,
        controls=["token_matched_foreign", "corrupted_outcomes", "corrupted_trajectories"],
        notation="san",
        max_seq_len=4096,
        seed=42,
        tokenizer=tok,
        model_loader=_loader,
    )
    assert manifest_path.exists()
    assert manifest["frozen"] is True
    assert manifest["selection_variants"] == ["diagonal_rook"]
    assert manifest["selected_checkpoint"]["name"] == "step_1000"
    assert len(manifest["candidates"]) == 3
    assert manifest["selected_checkpoint_sha256"] == compute_checkpoint_sha256(
        ckpts_dir / "step_1000"
    )
    assert manifest["snapshot_sha256"] == compute_snapshot_sha256(data_dir)

    # Verify weight-freeze check rejects unfrozen model
    unfrozen = _MockFrozenClassifier()
    unfrozen.train()
    with pytest.raises(RuntimeError, match="eval\\(\\)"):
        verify_model_frozen(unfrozen)
    unfrozen.eval()
    with pytest.raises(RuntimeError, match="requires_grad"):
        verify_model_frozen(unfrozen)

    # Verify uncached repeat determinism rejects noisy model (> 1e-5)
    noisy_model = _MockFrozenClassifier(jitter=0.5)
    noisy_model.eval()
    for p in noisy_model.parameters():
        p.requires_grad_(False)
    noisy_pred = CanonicalQwenPredictor(noisy_model, tok, max_seq_len=4096)
    with pytest.raises(RuntimeError, match="repeat determinism"):
        noisy_pred.verify_uncached_repeat_determinism(["hello world"], tol=1e-5)

    # Verify canonical unpadded batch_size=1 + model-bound cache behavior
    best_model = _loader(ckpts_dir / "step_1000")
    pred_a = CanonicalQwenPredictor(best_model, tok, model_sha256="sha_A", max_seq_len=4096)
    ep_sample = {
        "query_fen": "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
        "demos": [],
    }
    out1 = pred_a.predict_episode(ep_sample)
    out2 = pred_a.predict_episode(ep_sample)
    assert out1["cache_hit"] is False
    assert out2["cache_hit"] is True
    assert np.allclose(out1["p_pred"], out2["p_pred"])
    # Every forward call must have batch_size=1 and has_pad=False!
    assert len(best_model.forward_calls) == 1
    assert best_model.forward_calls[0][0] == 1
    assert best_model.forward_calls[0][2] is False

    # Changing model_sha256 invalidates cache key (model-bound!)
    pred_a._cache[("sha_B", out1["input_ids_sha256"])] = np.array([0.1, 0.8, 0.1])
    pred_b = CanonicalQwenPredictor(best_model, tok, model_sha256="sha_B", max_seq_len=4096)
    pred_b._cache = pred_a._cache
    out_b = pred_b.predict_episode(ep_sample)
    assert out_b["cache_hit"] is True
    assert np.allclose(out_b["p_pred"], [0.1, 0.8, 0.1])

    # Run frozen test evaluation from manifest
    eval_out_file = tmp_path / "eval_test_seed0.json"
    best_model.forward_calls.clear()
    eval_payload = evaluate_from_manifest(
        manifest_path=manifest_path,
        data_dir=data_dir,
        output_file=eval_out_file,
        split="test",
        n_boot=50,
        seed=42,
        model_seed=0,
        model=best_model,
        tokenizer=tok,
    )
    assert eval_out_file.exists()
    assert eval_payload["metadata"]["inference_mode"] == "canonical_unpadded_batch_size_1"
    assert eval_payload["metadata"]["determinism_check"]["passed"] is True
    assert eval_payload["metadata"]["cache_stats"]["hits"] > 0
    assert all(bsz == 1 and not has_pad for bsz, _, has_pad in best_model.forward_calls)

    # Check separate familiar / 9-rule transfer / standard reporting
    feasible = eval_payload["summary"]["feasible_cohort"]
    common = eval_payload["summary"]["common_cohort"]
    assert set(feasible.keys()) == {"familiar_adaptation", "heldout_transfer", "standard_control"}
    assert feasible["familiar_adaptation"]["is_transfer"] is False
    assert feasible["familiar_adaptation"]["variants"] == ["diagonal_rook"]
    assert feasible["heldout_transfer"]["is_transfer"] is True
    assert feasible["heldout_transfer"]["variants"] == ["short_knight", "antichess"]
    assert feasible["standard_control"]["is_transfer"] is False
    assert feasible["standard_control"]["variants"] == ["standard"]

    # Check vector vs categorical Brier, context gain, calibration with counts, equal_material, caveat
    tr_k2 = feasible["heldout_transfer"]["clean"]["2"]
    assert tr_k2["brier_vector"]["macro_mean"] != tr_k2["brier_categorical"]["macro_mean"]
    assert "context_gain" in tr_k2
    assert tr_k2["calibration"]["n_samples"] == tr_k2["n_records"]
    assert sum(tr_k2["calibration"]["bin_counts"]) == tr_k2["n_records"]
    assert tr_k2["calibration"]["side_calibration"]["white"]["n_samples"] > 0
    assert tr_k2["calibration"]["side_calibration"]["black"]["n_samples"] > 0
    assert tr_k2["equal_material"]["n_records"] > 0
    assert tr_k2["uncertainty_metadata"]["small_seed_caveat"] is not None
    assert tr_k2["uncertainty_metadata"]["mc_label_uncertainty_kept_separate"] is True

    # Check relevance comparison vs clean on controls
    for ctrl in ("token_matched_foreign", "corrupted_outcomes", "corrupted_trajectories"):
        ctrl_k2 = common["heldout_transfer"]["controls"][ctrl]["2"]
        assert "relevance_comparison_vs_clean" in ctrl_k2
        assert ctrl_k2["relevance_comparison_vs_clean"]["n_paired_records"] > 0

    # Check manifest tampering protection
    tampered_man = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered_man["selected_checkpoint_sha256"] = "0" * 64
    tampered_path = tmp_path / "tampered_manifest.json"
    tampered_path.write_text(json.dumps(tampered_man), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        evaluate_from_manifest(
            manifest_path=tampered_path,
            data_dir=data_dir,
            output_file=tmp_path / "bad.json",
            model=best_model,
            tokenizer=tok,
        )

    # Check multi-seed (S=3) aggregation
    eval_files = []
    for s in range(3):
        p = tmp_path / f"eval_seed_{s}.json"
        if s == 0:
            eval_files.append(eval_out_file)
        else:
            copy_payload = copy.deepcopy(eval_payload)
            copy_payload["metadata"]["model_seed"] = s
            for r in copy_payload["per_root_predictions"]:
                r["model_seed"] = s
            p.write_text(json.dumps(copy_payload), encoding="utf-8")
            eval_files.append(p)
    agg_out = aggregate_multi_seed_evaluations(
        eval_files, output_file=tmp_path / "eval_3seeds.json", n_boot=50, seed=42
    )
    assert agg_out["metadata"]["n_seeds"] == 3
    tr_s3 = agg_out["summary"]["feasible_cohort"]["heldout_transfer"]["clean"]["2"]
    assert tr_s3["nll"]["n_variants"] == 2
    assert tr_s3["uncertainty_metadata"]["n_seeds"] == 3
    assert "S=3 <= 3" in tr_s3["uncertainty_metadata"]["small_seed_caveat"]


def test_feasible_vs_common_cohort_when_one_root_infeasible(tmp_path: Path):
    """Verify feasible_cohort at K=2 vs K=0 differs when one root drops demos or fails matching."""
    data_dir = tmp_path / "data"
    _build_synthetic_long_study_dataset(data_dir, include_split_demos=True)
    # Replace one test demo in short_knight with a very long demo so K=2 exceeds tight max_seq_len
    long_demo = {
        "game_id": "short_knight_te_long",
        "initial_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "moves": ["e2e4"] * 120,
        "outcome": "win",
    }
    _write_jsonl(data_dir / "short_knight" / "games_demo_test.jsonl", [long_demo] * 6)

    tok = _WhitespaceMockTokenizer()
    model = _MockFrozenClassifier()
    ckpt_dir = tmp_path / "ckpt"
    _save_mock_checkpoint(ckpt_dir, model)
    frozen_model = _load_mock_checkpoint(ckpt_dir)

    # Set max_seq_len=450 so K=0 fits all 12 (~155 tok), K=2 diagonal_rook fits (~375 tok),
    # while K=2 short_knight (~590 tok) drops demos!
    manifest = {
        "frozen": True,
        "selected_checkpoint_path": str(ckpt_dir),
        "selected_checkpoint_sha256": compute_checkpoint_sha256(ckpt_dir),
        "snapshot_sha256": compute_snapshot_sha256(data_dir),
        "k_values": [0, 2],
        "num_draws": 1,
        "controls": ["token_matched_foreign"],
        "exclusions": {
            "max_seq_len": 450,
            "require_zero_dropped_demos": True,
            "excluded_variants": [],
            "excluded_root_ids": [],
        },
        "notation": "san",
    }
    man_path = tmp_path / "manifest.json"
    man_path.write_text(json.dumps(manifest), encoding="utf-8")

    res = evaluate_from_manifest(
        manifest_path=man_path,
        data_dir=data_dir,
        output_file=tmp_path / "cohort_eval.json",
        split="test",
        variants=["diagonal_rook", "short_knight"],
        n_boot=20,
        model=frozen_model,
        tokenizer=tok,
    )

    cov = res["summary"]["coverage"]
    # K=0 fits all 12 roots (6 diagonal_rook + 6 short_knight)
    assert cov["clean"]["0"]["n_feasible_roots"] == 12
    assert cov["clean"]["0"]["all_roots_feasible"] is True
    # K=2 drops demos on short_knight (all 6 roots) -> only 6 diagonal_rook roots feasible in clean
    assert cov["clean"]["2"]["n_feasible_roots"] == 6
    assert cov["clean"]["2"]["all_roots_feasible"] is False
    # token_matched_foreign matches diagonal_rook against standard/antichess (6 feasible roots)
    assert cov["token_matched_foreign"]["2"]["n_feasible_roots"] == 6
    assert cov["clean"]["0"]["n_common_roots"] == 6
    # Verify feasible vs common cohorts differ at K=0 (heldout short_knight has 6 in feasible, 0/None in common)
    k0_feas = res["summary"]["feasible_cohort"]["heldout_transfer"]["clean"]["0"]["n_roots"]
    assert k0_feas == 6
    assert res["summary"]["common_cohort"]["heldout_transfer"]["clean"]["0"] is None
    assert res["summary"]["common_cohort"]["familiar_adaptation"]["clean"]["0"]["n_roots"] == 6


# ---------------------------------------------------------------------------
# 4. Outcome-Independent R256 Subset Selection & Re-Estimation
# ---------------------------------------------------------------------------


def test_r256_subset_selection_outcome_independent_and_eval(tmp_path: Path):
    data_a = tmp_path / "data_a"
    data_b = tmp_path / "data_b"
    _build_synthetic_long_study_dataset(data_a, include_split_demos=True)
    _build_synthetic_long_study_dataset(data_b, include_split_demos=True)

    # Completely flip outcomes / R in data_b to prove outcome-independence
    for v in ("standard", "diagonal_rook", "short_knight", "antichess"):
        p_path = data_b / v / "positions_test.jsonl"
        rows = [json.loads(line) for line in p_path.read_text().splitlines() if line.strip()]
        for r in rows:
            r["label_probs"] = [0.05, 0.05, 0.90]
            r["R"] = 256
        _write_jsonl(p_path, rows)

    man_a_path = tmp_path / "r256_a.json"
    man_b_path = tmp_path / "r256_b.json"
    man_a = select_r256_subset(
        data_dir=data_a,
        output_manifest=man_a_path,
        families_per_variant=2,
        split="test",
    )
    # Test CLI entrypoint as well
    eval_long_study_main(
        [
            "select-r256-subset",
            "--data-dir",
            str(data_b),
            "--output-manifest",
            str(man_b_path),
            "--families-per-variant",
            "2",
            "--split",
            "test",
        ]
    )
    man_b = json.loads(man_b_path.read_text(encoding="utf-8"))

    assert man_a["outcome_independent"] is True
    assert man_a["selection_rule"] == "sha256(family_id)_ascending"
    assert man_a["variants"] == man_b["variants"]
    for v, v_info in man_a["variants"].items():
        assert v_info["n_selected_families"] == 2
        expected_sorted = sorted(
            [f"{v}_fam_A", f"{v}_fam_B", f"{v}_fam_C"],
            key=lambda fid: (hashlib.sha256(fid.encode("utf-8")).hexdigest(), fid),
        )[:2]
        assert v_info["selected_family_ids"] == expected_sorted

    # Test R256 re-estimation scoring inside evaluate_from_manifest
    # Write positions_test_r256.jsonl on data_a with R=256 labels
    for v in ("standard", "diagonal_rook", "short_knight", "antichess"):
        rows = [
            json.loads(line)
            for line in (data_a / v / "positions_test.jsonl").read_text().splitlines()
            if line.strip()
        ]
        for r in rows:
            r["R"] = 256
            r["label_probs"] = [0.55, 0.25, 0.20]
        _write_jsonl(data_a / v / "positions_test_r256.jsonl", rows)

    tok = _WhitespaceMockTokenizer()
    ckpt_dir = tmp_path / "ckpt_r256"
    _save_mock_checkpoint(ckpt_dir, _MockFrozenClassifier())
    frozen_model = _load_mock_checkpoint(ckpt_dir)

    eval_man_path = tmp_path / "eval_manifest.json"
    eval_man_path.write_text(
        json.dumps(
            {
                "frozen": True,
                "selected_checkpoint_path": str(ckpt_dir),
                "selected_checkpoint_sha256": compute_checkpoint_sha256(ckpt_dir),
                "snapshot_sha256": compute_snapshot_sha256(data_a),
                "k_values": [0, 1],
                "num_draws": 1,
                "controls": [],
                "exclusions": {
                    "max_seq_len": 4096,
                    "require_zero_dropped_demos": True,
                    "excluded_variants": [],
                    "excluded_root_ids": [],
                },
                "notation": "san",
            }
        )
    )
    res = evaluate_from_manifest(
        manifest_path=eval_man_path,
        data_dir=data_a,
        output_file=tmp_path / "eval_with_r256.json",
        split="test",
        n_boot=20,
        r256_subset_manifest=man_a_path,
        r256_positions_filename="positions_test_r256.jsonl",
        model=frozen_model,
        tokenizer=tok,
    )
    r256_res = res["r256_reestimation"]
    assert r256_res is not None
    assert r256_res["outcome_independent"] is True
    assert r256_res["positions_filename"] == "positions_test_r256.jsonl"
    tr_r256 = r256_res["summary"]["feasible_cohort"]["heldout_transfer"]["clean"]["1"]
    assert tr_r256["mc_noise_floor"]["per_variant"]["short_knight"]["R"] == 256


def test_categorical_vs_vector_brier_formula():
    p = np.array([[0.7, 0.2, 0.1]], dtype=np.float64)
    y = np.array([[0.6, 0.3, 0.1]], dtype=np.float64)
    # Vector Brier: (0.1)^2 + (-0.1)^2 + 0^2 = 0.02
    # Categorical Brier (argmax(y)=0 -> [1,0,0]): (-0.3)^2 + 0.2^2 + 0.1^2 = 0.09 + 0.04 + 0.01 = 0.14
    bcat = compute_categorical_brier_per_sample(p, y)
    assert np.isclose(bcat[0], 0.14)


def test_scratch_checkpoint_discovery_and_default_loader(tmp_path: Path):
    from chess_icl_v2.models.scratch_qwen2 import (
        ScratchQwen2Classifier,
        create_scratch_qwen2_config,
    )

    data_dir = tmp_path / "data_scratch"
    _build_synthetic_long_study_dataset(data_dir, include_split_demos=True)
    tok = _WhitespaceMockTokenizer()
    cfg = create_scratch_qwen2_config(
        vocab_size=5020,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
    )
    ckpts_dir = tmp_path / "scratch_candidates"
    for step in (500, 1000):
        m = ScratchQwen2Classifier(config=cfg, tokenizer=tok)
        m.save_weights(ckpts_dir / f"step_{step:05d}")

    discovered = discover_candidate_checkpoints(ckpts_dir)
    assert [step for step, _ in discovered] == [500, 1000]

    manifest_path = tmp_path / "scratch_manifest.json"
    manifest = select_checkpoint(
        data_dir=data_dir,
        output_manifest=manifest_path,
        checkpoints_dir=ckpts_dir,
        selection_regime="familiar",
        k_values=[0, 1],
        num_draws=1,
        controls=[],
        notation="san",
        max_seq_len=2048,
        seed=42,
        gpu=-1,
        tokenizer=tok,
        arch="auto",
    )
    assert manifest["arch"] == "scratch"
    assert manifest["selected_checkpoint"]["step"] in (500, 1000)

    res = evaluate_from_manifest(
        manifest_path=manifest_path,
        data_dir=data_dir,
        output_file=tmp_path / "scratch_eval.json",
        split="test",
        variants=["diagonal_rook"],
        n_boot=10,
        gpu=-1,
        tokenizer=tok,
        arch="auto",
    )
    assert res["metadata"]["selected_checkpoint_sha256"] == manifest["selected_checkpoint_sha256"]
