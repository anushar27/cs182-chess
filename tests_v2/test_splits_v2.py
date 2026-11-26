"""Unit tests for chess_icl_v2.data.splits (splitting, deduplication, leakage/balance audit)."""

from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np

from chess_icl_v2.data.splits import (
    audit_dataset_integrity,
    board_similarity_signature,
    check_prompt_leakage,
    compute_canonical_state_key,
    deduplicate_and_filter_roots,
    split_games_into_pools,
)
from chess_icl_v2.models.qwen_model import QwenPromptFormatter
from chess_icl_v2.variants import get_board


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _board_with_n_pawns(n_white_pawns: int, turn: str = "w") -> str:
    """Create a valid-looking FEN with Kings on e1/e8 and `n_white_pawns` on rank 2/3."""
    squares = ["."] * 64
    squares[4] = "K"   # e1
    squares[60] = "k"  # e8
    placed = 0
    for sq in list(range(8, 16)) + list(range(16, 24)):
        if placed >= n_white_pawns:
            break
        squares[sq] = "P"
        placed += 1
    rows = []
    for r in range(7, -1, -1):
        empty = 0
        row_s = []
        for f in range(8):
            ch = squares[r * 8 + f]
            if ch == ".":
                empty += 1
            else:
                if empty:
                    row_s.append(str(empty))
                    empty = 0
                row_s.append(ch)
        if empty:
            row_s.append(str(empty))
        rows.append("".join(row_s))
    return f"{'/'.join(rows)} {turn} - - 0 1"


# ---------------------------------------------------------------------------
# 1. Disjoint Game Pool Splitting
# ---------------------------------------------------------------------------


def test_split_games_into_pools_disjoint():
    games = []
    for g_idx in range(40):
        # Include two records sharing the same game_id to verify group-level splitting
        games.append({"game_id": f"g_{g_idx}", "moves": ["e2e4", "e7e5"], "outcome": "win"})
        games.append({"game_id": f"g_{g_idx}", "moves": ["d2d4", "d7d5"], "outcome": "loss"})

    pools = split_games_into_pools(
        games, demo_frac=0.4, train_frac=0.3, val_frac=0.15, test_frac=0.15, seed=42
    )
    pools_repeat = split_games_into_pools(
        games, demo_frac=0.4, train_frac=0.3, val_frac=0.15, test_frac=0.15, seed=42
    )
    assert pools == pools_repeat

    id_sets = {
        name: {g["game_id"] for g in lst}
        for name, lst in pools.items()
    }
    assert len(id_sets["demo_games"]) == 16
    assert len(id_sets["train_source_games"]) == 12
    assert len(id_sets["val_source_games"]) == 6
    assert len(id_sets["test_source_games"]) == 6

    all_ids = set()
    for s in id_sets.values():
        assert len(all_ids & s) == 0
        all_ids |= s
    assert len(all_ids) == 40


# ---------------------------------------------------------------------------
# 2. Board Similarity Signature
# ---------------------------------------------------------------------------


def test_board_similarity_signature_and_hamming():
    vb = get_board("blocked_squares", seed=7)
    arr, turn_bit = board_similarity_signature(vb)
    assert arr.shape == (64,)
    assert arr.dtype == np.int8
    assert turn_bit == 1  # White to move

    state_dict = vb.to_state_dict()
    arr2, turn_bit2 = board_similarity_signature(state_dict)
    assert np.array_equal(arr, arr2)
    assert turn_bit == turn_bit2


# ---------------------------------------------------------------------------
# 3. Canonical & Near-Duplicate Deduplication
# ---------------------------------------------------------------------------


def test_deduplicate_and_filter_roots():
    # Demo game starts from standard FEN and plays 1.e2e4 e7e5
    demo_games = [
        {
            "initial_fen": chess.STARTING_FEN,
            "moves": ["e2e4", "e7e5"],
            "outcome": "win",
        }
    ]

    # Exact state after 1.e2e4 e7e5 (64/64 match vs demo state -> neardup vs demos)
    fen_after_e4_e5 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"

    # A sparse base board (K+k+12 pawns = 14 pieces, 50 empty squares)
    # Changing 3 squares out of 64 -> 61/64 = 95.3% match (>= 90% -> near-dup!)
    # Changing 8 squares out of 64 -> 56/64 = 87.5% match (< 90% -> kept!)
    fen_test_base = _board_with_n_pawns(12, turn="w")
    fen_val_exact_dup = fen_test_base  # Exact canonical dup of test!
    fen_train_neardup = _board_with_n_pawns(9, turn="w")  # 3 differing squares (61/64 = 0.953)
    fen_train_distinct = _board_with_n_pawns(4, turn="w")  # 8 differing squares (56/64 = 0.875)
    fen_opening_ply = _board_with_n_pawns(2, turn="w")

    roots_by_split = {
        "test": [
            {"variant": "standard", "fen": fen_test_base, "ply": 12},
            {"variant": "standard", "fen": fen_after_e4_e5, "ply": 10},  # matches demo state
        ],
        "val": [
            {"variant": "standard", "fen": fen_val_exact_dup, "ply": 14},  # exact dup of test
            {"variant": "standard", "fen": fen_opening_ply, "ply": 4},  # ply < 8
        ],
        "train": [
            {"variant": "standard", "fen": fen_train_neardup, "ply": 16},  # 95.3% match vs test
            {"variant": "standard", "fen": fen_train_distinct, "ply": 16},  # 87.5% match -> kept
        ],
    }

    res = deduplicate_and_filter_roots(
        roots_by_split, demo_games=demo_games, max_similarity=0.90, min_ply=8
    )
    cleaned = res["roots_by_split"]
    stats = res["dedup_stats"]

    assert len(cleaned["test"]) == 1
    assert cleaned["test"][0]["fen"] == fen_test_base
    assert len(cleaned["val"]) == 0
    assert len(cleaned["train"]) == 1
    assert cleaned["train"][0]["fen"] == fen_train_distinct

    assert stats["opening_ply_removed"] == 1
    assert stats["exact_removed"] == 1
    assert stats["neardup_vs_demos_removed"] == 1
    assert stats["neardup_cross_split_removed"] == 1


# ---------------------------------------------------------------------------
# 4. Integrity, Balance & Negative Leakage Audit
# ---------------------------------------------------------------------------


def _build_clean_variant_dir(data_dir: Path, variant: str) -> None:
    v_dir = data_dir / variant
    demos = [
        {
            "source_game_id": f"{variant}_demo_{i}",
            "initial_fen": chess.STARTING_FEN,
            "moves": ["e2e4", "e7e5", "g1f3", "b8c6"],
            "outcome": "win" if i % 2 == 0 else "loss",
        }
        for i in range(4)
    ]
    train_pos = [
        {
            "source_game_id": f"{variant}_train_0",
            "root_id": f"{variant}_tr_0",
            "fen": _board_with_n_pawns(1, turn="w"),
            "y_true": [0.7, 0.2, 0.1],
        },
        {
            "source_game_id": f"{variant}_train_1",
            "root_id": f"{variant}_tr_1",
            "fen": _board_with_n_pawns(2, turn="b"),
            "y_true": [0.2, 0.3, 0.5],
        },
    ]
    val_pos = [
        {
            "source_game_id": f"{variant}_val_0",
            "root_id": f"{variant}_va_0",
            "fen": _board_with_n_pawns(3, turn="w"),
            "y_true": [0.6, 0.2, 0.2],
        },
    ]
    test_pos = [
        {
            "source_game_id": f"{variant}_test_0",
            "root_id": f"{variant}_te_0",
            "fen": _board_with_n_pawns(4, turn="b"),
            "y_true": [0.3, 0.4, 0.3],
        },
    ]
    _write_jsonl(v_dir / "games_demo.jsonl", demos)
    _write_jsonl(v_dir / "positions_train.jsonl", train_pos)
    _write_jsonl(v_dir / "positions_val.jsonl", val_pos)
    _write_jsonl(v_dir / "positions_test.jsonl", test_pos)


def test_audit_dataset_integrity_clean_and_negative_leakage(tmp_path: Path):
    data_dir = tmp_path / "clean_data"
    _build_clean_variant_dir(data_dir, "diagonal_rook")

    report = audit_dataset_integrity(data_dir, variants=["diagonal_rook"])
    assert report["passed"] is True
    assert report["n_errors"] == 0
    tr_stats = report["per_variant"]["diagonal_rook"]["split_stats"]["train"]
    assert tr_stats["n_positions"] == 2
    assert np.isclose(tr_stats["white_frac"], 0.5)
    assert np.isclose(tr_stats["black_frac"], 0.5)
    assert np.isclose(tr_stats["equal_material_frac"], 1.0)

    # Negative test 1: inject variant name into formatted prompt via leaky formatter header
    leaky_formatter = QwenPromptFormatter(
        notation="uci",
        header="Playing diagonal_rook chess variant! Predict outcome:",
    )
    leaky_report = audit_dataset_integrity(
        data_dir, variants=["diagonal_rook"], qwen_formatter=leaky_formatter
    )
    assert leaky_report["passed"] is False
    assert any("diagonal_rook" in err for err in leaky_report["errors"])

    # Direct helper check for injected variant name
    findings = check_prompt_leakage("Observe antichess moves: e2e4 e7e5")
    assert len(findings) >= 1 and any("antichess" in f for f in findings)

    # Negative test 2: source_game_id overlap between demo and test
    overlap_dir = tmp_path / "overlap_data"
    _build_clean_variant_dir(overlap_dir, "diagonal_rook")
    test_path = overlap_dir / "diagonal_rook" / "positions_test.jsonl"
    _write_jsonl(
        test_path,
        [
            {
                "source_game_id": "diagonal_rook_demo_0",  # overlaps games_demo.jsonl!
                "root_id": "te_overlap",
                "fen": _board_with_n_pawns(5, turn="w"),
                "y_true": [0.5, 0.25, 0.25],
            }
        ],
    )
    overlap_report = audit_dataset_integrity(overlap_dir, variants=["diagonal_rook"])
    assert overlap_report["passed"] is False
    assert any("source_game_id overlap" in err for err in overlap_report["errors"])

    # Negative test 3: canonical_state_key overlap between train and val
    ckey_dir = tmp_path / "ckey_overlap_data"
    _build_clean_variant_dir(ckey_dir, "diagonal_rook")
    val_path = ckey_dir / "diagonal_rook" / "positions_val.jsonl"
    _write_jsonl(
        val_path,
        [
            {
                "source_game_id": "diagonal_rook_val_unique",
                "root_id": "va_ckey_dup",
                "fen": _board_with_n_pawns(1, turn="w"),  # identical to train[0]!
                "y_true": [0.5, 0.25, 0.25],
            }
        ],
    )
    ckey_report = audit_dataset_integrity(ckey_dir, variants=["diagonal_rook"])
    assert ckey_report["passed"] is False
    assert any("canonical_state_key overlap" in err for err in ckey_report["errors"])
