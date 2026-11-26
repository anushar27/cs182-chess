"""Unit tests for chess_icl_v2.eval (metrics, baselines, controls, uncertainty)."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from chess_icl_v2.eval import (
    BoardKNNBaseline,
    DemoMajorityBaseline,
    EqualMaterialSlice,
    LengthHistogramBaseline,
    Predictor,
    QueryMaterialBaseline,
    TrainingPriorBaseline,
    UniformBaseline,
    aggregate_evaluation_records,
    aggregate_seed_draw_hierarchy,
    bootstrap_by_source_game,
    compute_argmax_accuracy,
    compute_brier,
    compute_context_gain,
    compute_ece,
    compute_l1_tv,
    compute_mc_noise_floor,
    compute_nll,
    compute_side_calibration,
    compute_standard_material_diff,
    encode_fen_64x13_plus_1,
    filter_equal_material,
    macro_average_variants,
    make_corrupted_outcome_episodes,
    make_corrupted_trajectory_episodes,
    make_mismatched_context_episodes,
)


# ---------------------------------------------------------------------------
# Metrics Tests
# ---------------------------------------------------------------------------


def test_compute_nll_and_clipping():
    p = np.array([[0.7, 0.2, 0.1], [0.0, 1.0, 0.0]], dtype=np.float64)
    y = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    mean_nll, per_sample = compute_nll(p, y)

    expected_0 = -math.log(0.7)
    expected_1 = -math.log(1e-7)
    assert per_sample.shape == (2,)
    assert np.isclose(per_sample[0], expected_0)
    assert np.isclose(per_sample[1], expected_1)
    assert np.isclose(mean_nll, 0.5 * (expected_0 + expected_1))

    # Empty input
    mean_empty, arr_empty = compute_nll(np.zeros((0, 3)), np.zeros((0, 3)))
    assert mean_empty == 0.0
    assert arr_empty.shape == (0,)


def test_compute_brier_and_l1_tv():
    p = np.array([[0.8, 0.1, 0.1], [0.2, 0.5, 0.3]], dtype=np.float64)
    y = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]], dtype=np.float64)

    # Row 0: (0.2^2 + 0.1^2 + 0.1^2) = 0.06
    # Row 1: (0.2^2 + 0.0^2 + 0.2^2) = 0.08
    brier = compute_brier(p, y)
    assert np.isclose(brier, 0.07)

    # Row 0 L1: 0.2 + 0.1 + 0.1 = 0.4, TV = 0.2
    # Row 1 L1: 0.2 + 0.0 + 0.2 = 0.4, TV = 0.2
    tv_l1 = compute_l1_tv(p, y)
    assert np.isclose(tv_l1["l1"], 0.4)
    assert np.isclose(tv_l1["tv"], 0.2)


def test_compute_argmax_accuracy_and_counts():
    p = np.array(
        [
            [0.7, 0.2, 0.1],  # win (true win) -> hit
            [0.6, 0.3, 0.1],  # win (true draw) -> miss
            [0.1, 0.2, 0.7],  # loss (true loss) -> hit
            [0.2, 0.5, 0.3],  # draw (true draw) -> hit
        ]
    )
    y = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.2, 0.6, 0.2],
            [0.0, 0.2, 0.8],
            [0.1, 0.7, 0.2],
        ]
    )
    res = compute_argmax_accuracy(p, y)
    assert np.isclose(res["accuracy"], 0.75)
    assert res["n_samples"] == 4
    assert res["pred_counts"] == {"win": 2, "draw": 1, "loss": 1}
    assert res["true_counts"] == {"win": 1, "draw": 2, "loss": 1}


def test_compute_ece_and_side_calibration():
    # Perfectly calibrated confidence on White, miscalibrated on Black
    p = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.8, 0.1, 0.1],
            [0.9, 0.05, 0.05],
            [0.9, 0.05, 0.05],
        ]
    )
    y = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.8, 0.1, 0.1],
            [0.1, 0.1, 0.8],
            [0.1, 0.1, 0.8],
        ]
    )
    ece_white = compute_ece(p[:2], y[:2], n_bins=10)
    assert np.isclose(ece_white["ece"], 0.0)
    assert np.isclose(ece_white["classwise_ece"], 0.0)

    res = compute_side_calibration(p, y, sides=["w", "white", "b", "black"])
    assert res["white"]["n_samples"] == 2
    assert res["black"]["n_samples"] == 2
    assert np.isclose(res["white"]["ece"], 0.0)
    assert np.isclose(res["black"]["ece"], 0.8)
    assert np.allclose(res["white"]["mean_pred"], [0.8, 0.1, 0.1])
    assert np.allclose(res["black"]["mean_pred"], [0.9, 0.05, 0.05])
    assert res["side_nll_gap"] > 1.0


def test_compute_context_gain():
    nll_k0 = np.array([1.2, 1.0, 0.9, 1.5])
    nll_k4 = np.array([0.8, 0.9, 0.6, 1.5])
    variants = ["v1", "v1", "v2", "v2"]

    cg = compute_context_gain(nll_k0, nll_k4, variants=variants)
    # gains: [0.4, 0.1, 0.3, 0.0] -> v1 mean = 0.25, v2 mean = 0.15, macro = 0.20
    assert np.isclose(cg["mean"], 0.20)
    assert np.isclose(cg["macro_mean"], 0.20)
    assert np.isclose(cg["per_variant"]["v1"]["mean"], 0.25)
    assert np.isclose(cg["per_variant"]["v2"]["mean"], 0.15)
    assert cg["n_roots"] == 4

    # Mapping input API
    cg_map = compute_context_gain(
        {"v1": [1.2, 1.0], "v2": [0.9, 1.5]},
        {"v1": [0.8, 0.9], "v2": [0.6, 1.5]},
    )
    assert np.isclose(cg_map["macro_mean"], 0.20)


def test_compute_mc_noise_floor():
    # Pure deterministic targets -> zero entropy, zero MC variance
    y_det = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    floor_det = compute_mc_noise_floor(y_det, R=20)
    assert np.isclose(floor_det["entropy_floor"], 0.0)
    assert np.isclose(floor_det["mc_brier_variance"], 0.0)
    assert np.isclose(floor_det["mc_nll_bias"], 0.0)

    # Uniform targets -> entropy = ln(3), gini = 2/3, brier_var = (2/3)/R
    y_uni = np.array([[1 / 3, 1 / 3, 1 / 3]])
    floor_r10 = compute_mc_noise_floor(y_uni, R=10)
    floor_r40 = compute_mc_noise_floor(y_uni, R=40)
    assert np.isclose(floor_r10["entropy_floor"], math.log(3.0))
    assert np.isclose(floor_r10["mc_brier_variance"], (2.0 / 3.0) / 10.0)
    assert np.isclose(floor_r10["mc_nll_bias"], 2.0 / (2.0 * 10.0))
    assert np.isclose(floor_r10["mc_brier_variance"], 4.0 * floor_r40["mc_brier_variance"])


# ---------------------------------------------------------------------------
# Baselines Tests
# ---------------------------------------------------------------------------


def test_uniform_and_training_prior_baselines():
    uni = UniformBaseline()
    assert isinstance(uni, Predictor)
    eps = [
        {"query_fen": "8/8/8/8/8/8/8/4K2k w - - 0 1", "demos": [], "y_true": [0.7, 0.2, 0.1]},
        {"query_fen": "8/8/8/8/8/8/8/4K2k b - - 0 1", "demos": [], "y_true": [0.3, 0.4, 0.3]},
    ]
    p_uni = uni.predict_proba(eps)
    assert p_uni.shape == (2, 3)
    assert np.allclose(p_uni, 1.0 / 3.0)

    prior = TrainingPriorBaseline().fit(eps)
    p_prior = prior.predict_proba(eps)
    assert p_prior.shape == (2, 3)
    assert np.allclose(p_prior[0], [0.5, 0.3, 0.2])
    assert np.allclose(p_prior[1], [0.5, 0.3, 0.2])


def test_demo_majority_baseline_perspective_flip():
    demos = [
        {"moves": ["e2e4", "e7e5"], "outcome": "win"},
        {"moves": ["d2d4", "d7d5"], "outcome": "win"},
        {"moves": ["c2c4", "c7c5"], "outcome": "win"},
        {"moves": ["g1f3", "g8f6"], "outcome": "draw"},
    ]
    # White perspective counts: [3, 1, 0], alpha=1 -> [(3+1)/7, (1+1)/7, (0+1)/7] = [4/7, 2/7, 1/7]
    ep_white = {"query_fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1", "demos": demos}
    ep_black = {"query_fen": "4k3/8/8/8/8/8/8/4K3 b - - 0 1", "demos": demos}
    ep_k0 = {"query_fen": "4k3/8/8/8/8/8/8/4K3 b - - 0 1", "demos": []}

    model = DemoMajorityBaseline(alpha=1.0)
    probs = model.predict_proba([ep_white, ep_black, ep_k0])
    assert np.allclose(probs[0], [4 / 7, 2 / 7, 1 / 7])
    # Black to move MUST swap W and L!
    assert np.allclose(probs[1], [1 / 7, 2 / 7, 4 / 7])
    # K=0 falls back to uniform
    assert np.allclose(probs[2], [1 / 3, 1 / 3, 1 / 3])


def test_length_histogram_baseline_fit_and_predict():
    train_eps = []
    for i in range(40):
        if i % 2 == 0:
            # Short games + White wins -> query wins
            demos = [
                {"moves": ["e2e4", "e7e5"], "outcome": "win"},
                {"moves": ["d2d4", "d7d5"], "outcome": "win"},
            ]
            y = [0.9, 0.05, 0.05]
        else:
            # Long games + White loses -> query loses
            demos = [
                {"moves": ["e2e4"] * 12, "outcome": "loss"},
                {"moves": ["d2d4"] * 14, "outcome": "loss"},
            ]
            y = [0.05, 0.05, 0.9]
        train_eps.append(
            {
                "query_fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
                "demos": demos,
                "y_true": y,
            }
        )

    clf = LengthHistogramBaseline(l2_reg=1e-4).fit(train_eps)
    preds = clf.predict_proba(train_eps)
    assert preds.shape == (40, 3)
    assert np.allclose(np.sum(preds, axis=1), 1.0)
    assert preds[0, 0] > 0.8
    assert preds[1, 2] > 0.8


def test_query_material_baseline_fit_and_predict():
    # White has extra Queen -> White wins when White to move, loses when Black to move
    fen_w_up_wtm = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
    fen_w_up_btm = "4k3/8/8/8/8/8/8/3QK3 b - - 0 1"
    fen_b_up_btm = "3qk3/8/8/8/8/8/8/4K3 b - - 0 1"
    fen_b_up_wtm = "3qk3/8/8/8/8/8/8/4K3 w - - 0 1"

    train_eps = (
        [{"query_fen": fen_w_up_wtm, "demos": [], "y_true": [0.95, 0.03, 0.02]}] * 10
        + [{"query_fen": fen_w_up_btm, "demos": [], "y_true": [0.02, 0.03, 0.95]}] * 10
        + [{"query_fen": fen_b_up_btm, "demos": [], "y_true": [0.95, 0.03, 0.02]}] * 10
        + [{"query_fen": fen_b_up_wtm, "demos": [], "y_true": [0.02, 0.03, 0.95]}] * 10
    )

    clf = QueryMaterialBaseline(l2_reg=1e-4).fit(train_eps)
    test_eps = [
        {"query_fen": fen_w_up_wtm, "demos": []},
        {"query_fen": fen_w_up_btm, "demos": []},
    ]
    preds = clf.predict_proba(test_eps)
    assert preds.shape == (2, 3)
    assert np.allclose(np.sum(preds, axis=1), 1.0)
    assert preds[0, 0] > 0.85  # White up & White to move -> win
    assert preds[1, 2] > 0.85  # White up & Black to move -> loss


def test_board_knn_baseline_encoding_metrics_and_perspective():
    # Test 64*13 + 1 encoding including 'X' blocked squares
    fen_blocked = "4k3/8/8/3X4/8/8/8/4K3 w - - 0 1"
    vec = encode_fen_64x13_plus_1(fen_blocked)
    assert vec.shape == (64 * 13 + 1,)
    assert vec[-1] == 1.0  # White to move
    # d5 is rank 4 (0-indexed), file 3 -> square 4*8 + 3 = 35 -> all 13 channels 0
    d5_slice = vec[35 * 13 : 36 * 13]
    assert np.all(d5_slice == 0.0)

    # Demo 1 reaches after 1.e4 e5 (Black to move after e4, White to move after e5)
    # Include a non-standard diagonal rook move in demo 2 to verify variant move safety
    demo_win = {
        "initial_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "moves": ["e2e4", "e7e5"],
        "outcome": "win",
    }
    demo_loss = {
        "initial_fen": "4k3/8/8/8/8/8/8/R3K3 w - - 0 1",
        "moves": ["a1d4"],  # Diagonal rook move (illegal in standard chess)
        "outcome": "loss",
    }

    # Query exactly matches position after 1. e2e4 (Black to move!)
    # Nearest neighbor is the post-e2e4 state from demo_win (which had White outcome = "win").
    # Because query is Black to move, White "win" must map to high Black "loss" (index 2)!
    query_after_e4 = {
        "query_fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "demos": [demo_win, demo_loss],
    }

    for metric in ("cosine", "hamming"):
        knn = BoardKNNBaseline(top_k=1, metric=metric, state_mode="trajectories", alpha=0.5)
        probs = knn.predict_proba([query_after_e4])
        assert probs.shape == (1, 3)
        assert np.allclose(np.sum(probs, axis=1), 1.0)
        assert probs[0, 2] > probs[0, 0]  # Loss for Black > Win for Black

    # Final state mode test
    knn_final = BoardKNNBaseline(top_k=1, metric="cosine", state_mode="final", alpha=0.0)
    query_after_e5 = {
        "query_fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
        "demos": [demo_win, demo_loss],
    }
    p_final = knn_final.predict_proba([query_after_e5])
    assert np.isclose(p_final[0, 0], 1.0)


def test_equal_material_slice():
    ep_eq = {"query_fen": "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"}  # +1 pawn
    ep_eq2 = {"query_fen": "4k3/8/8/8/8/8/3PP3/4K3 w - - 0 1"}  # +2 pawns
    ep_unbal = {"query_fen": "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"}  # +9 queen

    assert compute_standard_material_diff(ep_eq["query_fen"]) == 1
    assert compute_standard_material_diff(ep_unbal["query_fen"]) == 9

    filtered = filter_equal_material([ep_eq, ep_eq2, ep_unbal], threshold=2)
    assert len(filtered) == 2
    slicer = EqualMaterialSlice(threshold=1)
    assert slicer.select_indices([ep_eq, ep_eq2, ep_unbal]) == [0]


# ---------------------------------------------------------------------------
# Controls Tests
# ---------------------------------------------------------------------------


def _sample_episodes():
    return [
        {
            "variant": "antichess",
            "query_fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "y_true": [0.7, 0.2, 0.1],
            "demos": [
                {
                    "moves": ["e2e4", "e7e5", "g1f3"],
                    "outcome": "win",
                    "fens": ["fen1", "fen2"],
                },
                {
                    "moves": ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3"],
                    "outcome": "loss",
                },
            ],
        },
        {
            "variant": "diagonal_rook",
            "query_fen": "4k3/8/8/8/8/8/8/4K3 b - - 0 1",
            "y_true": [0.2, 0.3, 0.5],
            "demos": [
                {"moves": ["a2a4", "a7a5"], "outcome": "draw"},
                {"moves": ["b2b4", "b7b5", "c2c4", "c7c5"], "outcome": "win"},
            ],
        },
    ]


def test_make_mismatched_context_episodes():
    episodes = _sample_episodes()
    orig_snapshot = copy.deepcopy(episodes)

    other_pool = {
        "antichess": [
            {"moves": ["e2e4"] * 3, "outcome": "win"},
            {"moves": ["e2e4"] * 5, "outcome": "loss"},
        ],
        "horde": [
            {"moves": ["a2a3"] * 2, "outcome": "win"},
            {"moves": ["a2a3"] * 3, "outcome": "win"},
            {"moves": ["a2a3"] * 5, "outcome": "loss"},
            {"moves": ["a2a3"] * 20, "outcome": "draw"},
        ],
    }

    mismatched = make_mismatched_context_episodes(episodes, other_pool, seed=42)
    assert episodes == orig_snapshot  # Input never mutated
    assert len(mismatched) == len(episodes)

    # First episode is antichess (lengths 3 and 5) -> must pull from horde lengths 3 and 5!
    assert mismatched[0]["control_source_variant"] == "horde"
    assert len(mismatched[0]["demos"]) == 2
    assert sorted(len(d["moves"]) for d in mismatched[0]["demos"]) == [3, 5]


def test_make_corrupted_outcome_episodes():
    episodes = _sample_episodes()
    orig_snapshot = copy.deepcopy(episodes)

    inv = make_corrupted_outcome_episodes(episodes, mode="inverted", seed=42)
    assert episodes == orig_snapshot
    assert inv[0]["demos"][0]["outcome"] == "loss"
    assert inv[0]["demos"][1]["outcome"] == "win"
    assert inv[1]["demos"][0]["outcome"] == "draw"
    assert inv[1]["demos"][1]["outcome"] == "loss"

    all_draw = make_corrupted_outcome_episodes(episodes, mode="all_draw", seed=42)
    assert episodes == orig_snapshot
    for ep in all_draw:
        assert all(d["outcome"] == "draw" for d in ep["demos"])

    scrambled = make_corrupted_outcome_episodes(episodes, mode="scrambled", seed=7)
    assert episodes == orig_snapshot
    orig_multiset = sorted(d["outcome"] for ep in episodes for d in ep["demos"])
    scram_multiset = sorted(d["outcome"] for ep in scrambled for d in ep["demos"])
    assert orig_multiset == scram_multiset

    with pytest.raises(ValueError):
        make_corrupted_outcome_episodes(episodes, mode="invalid_mode")


def test_make_corrupted_trajectory_episodes():
    episodes = _sample_episodes()
    orig_snapshot = copy.deepcopy(episodes)

    for mode in ("shuffled", "mismatched"):
        corrupted = make_corrupted_trajectory_episodes(episodes, seed=42, mode=mode)
        assert episodes == orig_snapshot
        for ep_orig, ep_corr in zip(episodes, corrupted):
            for d_orig, d_corr in zip(ep_orig["demos"], ep_corr["demos"]):
                assert d_corr["outcome"] == d_orig["outcome"]
                assert len(d_corr["moves"]) == len(d_orig["moves"])
                assert "fens" not in d_corr
        if mode == "shuffled":
            assert corrupted[0]["demos"][0]["moves"] != episodes[0]["demos"][0]["moves"]


# ---------------------------------------------------------------------------
# Uncertainty Tests
# ---------------------------------------------------------------------------


def test_bootstrap_by_source_game():
    # 4 clusters, 3 roots each -> identical values within each cluster
    # Cluster means: 0.0, 1.0, 2.0, 3.0 -> overall mean = 1.5
    per_root = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
    game_ids = ["g0", "g0", "g0", "g1", "g1", "g1", "g2", "g2", "g2", "g3", "g3", "g3"]

    res_clust = bootstrap_by_source_game(per_root, game_ids, n_boot=2000, seed=42)
    # Independent roots would have smaller std than clustered games!
    res_indep = bootstrap_by_source_game(per_root, list(range(12)), n_boot=2000, seed=42)

    assert np.isclose(res_clust["mean"], 1.5)
    assert res_clust["n_clusters"] == 4
    assert res_clust["n_roots"] == 12
    assert res_clust["ci_low"] <= res_clust["mean"] <= res_clust["ci_high"]
    assert res_clust["std"] > res_indep["std"]

    # Deterministic repeatability
    res_repeat = bootstrap_by_source_game(per_root, game_ids, n_boot=2000, seed=42)
    assert res_clust == res_repeat


def test_hierarchy_and_macro_average_variants():
    rng = np.random.default_rng(123)
    # 3 seeds, 5 draws, 6 roots (2 per game)
    arr_v1 = 1.0 + 0.1 * rng.standard_normal((3, 5, 6))
    arr_v2 = 2.0 + 0.1 * rng.standard_normal((3, 5, 6))
    gids = ["g0", "g0", "g1", "g1", "g2", "g2"]

    hier = aggregate_seed_draw_hierarchy(arr_v1, source_game_ids=gids, n_boot=500, seed=42)
    assert hier["n_seeds"] == 3
    assert hier["n_draws"] == 5
    assert hier["n_roots"] == 6
    assert len(hier["seed_means"]) == 3
    assert hier["combined_se"] >= hier["game_bootstrap"]["std"]

    macro = macro_average_variants(
        {"v1": arr_v1, "v2": arr_v2},
        source_game_ids_by_variant={"v1": gids, "v2": gids},
        n_boot=500,
        seed=42,
    )
    assert macro["n_variants"] == 2
    expected_macro = 0.5 * (np.mean(arr_v1) + np.mean(arr_v2))
    assert np.isclose(macro["macro_mean"], expected_macro)
    assert macro["ci_low"] <= macro["macro_mean"] <= macro["ci_high"]

    # Flat record aggregation API
    records = []
    for v_name, arr in [("v1", arr_v1), ("v2", arr_v2)]:
        for s in range(3):
            for d in range(5):
                for r in range(6):
                    records.append(
                        {
                            "variant": v_name,
                            "model_seed": s,
                            "demo_draw": d,
                            "root_id": f"r{r}",
                            "source_game_id": gids[r],
                            "nll": float(arr[s, d, r]),
                        }
                    )
    rec_res = aggregate_evaluation_records(records, value_key="nll", n_boot=500, seed=42)
    assert np.isclose(rec_res["macro_mean"], expected_macro)
