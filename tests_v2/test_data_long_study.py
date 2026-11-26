"""Tests for long-study data production, censoring/budget, radius-2 audit, and snapshots."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

import chess
import pytest

from chess_icl_v2.data.splits import (
    audit_dataset_integrity,
    build_pilot_index,
    radius_two_audit,
    sample_adjacent_roots_per_family,
    split_games_into_independent_pools,
)
from chess_icl_v2.rollouts.generator import (
    RULES_VERSION,
    play_full_game,
    rollout_continuations,
    rollout_continuations_v2,
)
from chess_icl_v2.variants import get_board
from scripts_v2.build_long_study_data import (
    freeze_snapshot,
    run_initial_generation,
    run_r256_subset,
    run_train_wave,
    verify_snapshot,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sparse_fen(white_pawns_squares: list[int], turn: str = "w") -> str:
    """Create a legal-looking sparse FEN with Kings on e1/e8 and White pawns on specified squares."""
    squares = ["."] * 64
    squares[4] = "K"   # e1
    squares[60] = "k"  # e8
    for sq in white_pawns_squares:
        squares[sq] = "P"
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
# 1. Budget, Same-Stream Extension & Unresolved Censoring Tests
# ---------------------------------------------------------------------------


def test_same_stream_extension_play_full_game_and_rollouts():
    # Extending from max_plies=6 -> 18 on the same board & RNG stream must match max_plies=18 identically.
    g_ext = play_full_game(
        "standard",
        "terminal_aware_v1",
        max_plies=6,
        seed=77,
        extend_until_terminal=True,
        extend_step_plies=6,
        max_total_plies=18,
        max_extensions=2,
        unresolved_as_draw=False,
    )
    g_direct = play_full_game(
        "standard",
        "terminal_aware_v1",
        max_plies=18,
        seed=77,
        extend_until_terminal=False,
        max_extensions=0,
        unresolved_as_draw=False,
    )
    assert g_ext["rules_version"] == RULES_VERSION == "research_rules_v2_royal_conversion"
    assert g_ext["moves"] == g_direct["moves"]
    assert g_ext["move_count"] == 18
    assert g_ext["plies_budget_used"] == 18
    assert g_ext["extensions_used"] == 2
    assert g_ext["is_terminal"] is False
    assert g_ext["result"] == "unresolved"

    # Rollout continuation extension uses exact same RNG stream
    b_start = get_board("standard")
    r_ext = rollout_continuations(
        "standard",
        b_start.to_state_dict(),
        "terminal_aware_v1",
        R=2,
        max_plies=4,
        seed=11,
        unresolved_as_draw=False,
        extend_step_plies=4,
        max_total_plies=12,
        max_extensions=2,
    )
    assert r_ext["rules_version"] == RULES_VERSION
    assert r_ext["budget_used"] == 12
    assert r_ext["total_plies"] == 24
    assert r_ext["extensions_used"] == 4  # 2 extensions * 2 rollouts


def test_unresolved_continuations_never_become_draws_or_renormalized():
    b_start = get_board("standard")
    res_all_unres = rollout_continuations(
        "standard",
        b_start.to_state_dict(),
        "terminal_aware_v1",
        R=4,
        max_plies=1,
        seed=5,
        unresolved_as_draw=False,
    )
    assert res_all_unres["label_probs"] is None
    assert res_all_unres["label_probs_censored"] is None
    assert res_all_unres["n_unresolved"] == 4
    assert res_all_unres["counts"] == [0, 0, 0]
    assert res_all_unres["horizon_rate"] == 1.0
    assert res_all_unres["target_bounds"] == {
        "lower": [0.0, 0.0, 0.0],
        "upper": [1.0, 1.0, 1.0],
    }

    # Mixed resolved + unresolved: White Ra1, Ke1 vs Black Kg8, pawns f7,g7,h7.
    # Ra1-a8# is backrank mate in 1 (1 of 14 legal moves); all other moves are non-terminal at max_plies=1.
    fen_mixed = "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1"
    b_mix = get_board("standard", fen=fen_mixed)
    res_mix = rollout_continuations(
        "standard",
        b_mix.to_state_dict(),
        "uniform_legal_v1",
        R=64,
        max_plies=1,
        seed=42,
        unresolved_as_draw=False,
    )
    assert 0 < res_mix["n_unresolved"] < 64
    assert res_mix["counts"][0] > 0  # some Ra1-a8# wins
    assert res_mix["counts"][1] == 0  # unresolved MUST NOT become draws!
    assert sum(res_mix["counts"]) + res_mix["n_unresolved"] == 64
    assert res_mix["label_probs"] is None
    assert res_mix["label_probs_censored"] is None
    w, d, l = res_mix["counts"]
    u = res_mix["n_unresolved"]
    assert res_mix["target_bounds"]["lower"] == [w / 64, d / 64, l / 64]
    assert res_mix["target_bounds"]["upper"] == [(w + u) / 64, (d + u) / 64, (l + u) / 64]

    # Fully resolved case via rollout_continuations_v2 (mate in 1 for White)
    fen_mate = "4k3/3ppp2/8/8/2B5/5Q2/8/4K3 w - - 0 1"
    b_mate = get_board("standard", fen=fen_mate)
    res_v2 = rollout_continuations_v2(
        "standard",
        b_mate.to_state_dict(),
        policy_id="terminal_aware_v1",
        R=8,
        initial_budget=2560,
        seed=1,
    )
    assert res_v2["rules_version"] == RULES_VERSION
    assert res_v2["n_unresolved"] == 0
    assert res_v2["label_probs"] == [1.0, 0.0, 0.0]
    assert res_v2["target_bounds"] == {"lower": [1.0, 0.0, 0.0], "upper": [1.0, 0.0, 0.0]}


# ---------------------------------------------------------------------------
# 2. Independent Demo Pools & Max-2 Adjacent Family Sampling Tests
# ---------------------------------------------------------------------------


def test_independent_demo_pools_and_terminal_admission():
    games = []
    for i in range(20):
        games.append(
            {
                "family_id": f"fam_{i}",
                "source_game_id": f"fam_{i}",
                "is_terminal": i >= 2,  # first 2 are non-terminal
                "hit_horizon": i < 2,
                "moves": ["e2e4", "e7e5"],
            }
        )

    pools = split_games_into_independent_pools(
        games,
        demo_train_count=4,
        demo_val_count=2,
        demo_test_count=2,
        train_source_count=4,
        val_source_count=2,
        test_source_count=2,
        seed=99,
        require_terminal_demos=True,
    )

    for dk in ("demo_train_games", "demo_val_games", "demo_test_games"):
        assert all(g["is_terminal"] for g in pools[dk])

    seen = set()
    for name, lst in pools.items():
        ids = {g["family_id"] for g in lst}
        assert len(seen & ids) == 0, f"Pool overlap in {name}"
        seen |= ids


def test_sample_adjacent_roots_per_family_max_two_and_balanced():
    source_games = []
    for i in range(6):
        g = play_full_game(
            "standard",
            "uniform_legal_v1",
            max_plies=30,
            seed=100 + i,
            collect_states=True,
            min_sample_ply=8,
        )
        g["family_id"] = f"fam_{i}"
        g["source_game_id"] = f"fam_{i}"
        source_games.append(g)

    roots = sample_adjacent_roots_per_family(
        source_games,
        variant_name="standard",
        max_families=4,
        seed=42,
        min_ply=8,
        max_roots_per_family=2,
    )
    by_fid: dict[str, list[dict]] = {}
    for r in roots:
        by_fid.setdefault(r["family_id"], []).append(r)

    assert len(by_fid) <= 4
    for fid, items in by_fid.items():
        assert 1 <= len(items) <= 2
        assert all(it["ply"] >= 8 for it in items)
        if len(items) == 2:
            assert abs(items[0]["ply"] - items[1]["ply"]) == 1
            assert items[0]["turn"] != items[1]["turn"]


# ---------------------------------------------------------------------------
# 3. Radius-Two Geometry & Pilot Separation Audit Tests
# ---------------------------------------------------------------------------


def test_radius_two_audit_catches_hamming_le_2_and_pilot_leaks(tmp_path: Path):
    # Base boards with non-overlapping pawn configurations
    train_fen = _sparse_fen([8, 9, 10, 11, 12], turn="w")          # 5 pawns on a2..e2
    val_clean_fen = _sparse_fen([16, 17, 18, 19, 20], turn="w")     # a3..e3 (Hamming = 10 vs train)
    test_clean_fen = _sparse_fen([24, 25, 26, 27, 28], turn="w")    # a4..e4

    # Hamming == 3 vs train_fen (replace 1 pawn [8->13] = 2 diff squares, plus [9->14] = 4 diff?
    # Better: adding 3 pawns at [13, 14, 15] -> exact Hamming == 3!)
    ham3_fen = _sparse_fen([8, 9, 10, 11, 12, 13, 14, 15], turn="w")
    # Hamming == 2 vs train_fen (adding 2 pawns at [13, 14])
    ham2_fen = _sparse_fen([8, 9, 10, 11, 12, 13, 14], turn="w")
    # Hamming == 1 vs train_fen (adding 1 pawn at [13])
    ham1_fen = _sparse_fen([8, 9, 10, 11, 12, 13], turn="w")
    # Opposite turn Hamming == 1 (must NOT trigger turn-matched radius-2 leak!)
    ham1_opp_turn = _sparse_fen([8, 9, 10, 11, 12, 13], turn="b")

    pilot_dir = tmp_path / "pilot"
    pilot_fen = _sparse_fen([32, 33, 34, 35, 36], turn="w")
    _write_jsonl(pilot_dir / "positions_eval.jsonl", [{"fen": pilot_fen, "label": "draw"}])

    demo_pools = {
        "train": [
            {
                "family_id": "demo_tr_0",
                "initial_fen": chess.STARTING_FEN,
                "moves": ["e2e4", "e7e5", "g1f3", "b8c6"],
                "is_terminal": True,
            }
        ],
        "val": [],
        "test": [],
    }

    # Clean case: Hamming == 3 and opposite-turn Hamming == 1 both pass!
    clean_audit = radius_two_audit(
        roots_by_split={
            "train": [{"family_id": "f_tr", "fen": train_fen, "ply": 20}],
            "val": [{"family_id": "f_va", "fen": ham3_fen, "ply": 20}],
            "test": [
                {"family_id": "f_te0", "fen": test_clean_fen, "ply": 20},
                {"family_id": "f_te1", "fen": ham1_opp_turn, "ply": 21},
            ],
        },
        demo_pools=demo_pools,
        pilot_paths=[pilot_dir],
        variant_name="standard",
    )
    assert clean_audit["passed"] is True
    assert clean_audit["n_violations"] == 0

    # Leak 1: Cross-split Hamming == 2
    leak_h2 = radius_two_audit(
        roots_by_split={
            "train": [{"family_id": "f_tr", "fen": train_fen, "ply": 20}],
            "val": [{"family_id": "f_va", "fen": ham2_fen, "ply": 20}],
            "test": [{"family_id": "f_te", "fen": val_clean_fen, "ply": 20}],
        },
        demo_pools=demo_pools,
        pilot_paths=[pilot_dir],
        variant_name="standard",
    )
    assert leak_h2["passed"] is False
    assert len(leak_h2["cross_split_violations"]) == 1
    assert leak_h2["cross_split_violations"][0]["hamming_distance"] == 2

    # Leak 2: Query root Hamming == 1 against demo mid-game state (after 1. e4 e5)
    fen_after_e4_e5_h1 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 1 2"
    # Wait: after e4 e5 Nf3 it's Black to move; let's modify 1 pawn on a2->a3 from after-e4-e5 (Hamming=2!)
    fen_e4_e5_plus1 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/P7/1PPP1PPP/RNBQKBNR w KQkq - 0 2"
    leak_demo = radius_two_audit(
        roots_by_split={
            "train": [{"family_id": "f_tr", "fen": fen_e4_e5_plus1, "ply": 10}],
            "val": [],
            "test": [],
        },
        demo_pools=demo_pools,
        pilot_paths=[pilot_dir],
        variant_name="standard",
    )
    assert leak_demo["passed"] is False
    assert len(leak_demo["vs_demos_violations"]) == 1
    assert leak_demo["vs_demos_violations"][0]["hamming_distance"] == 2

    # Leak 3: Query root Hamming == 1 against prior pilot file
    pilot_h1 = _sparse_fen([32, 33, 34, 35, 36, 37], turn="w")
    leak_pilot = radius_two_audit(
        roots_by_split={
            "train": [{"family_id": "f_tr", "fen": pilot_h1, "ply": 24}],
            "val": [],
            "test": [],
        },
        demo_pools=demo_pools,
        pilot_paths=[pilot_dir],
        variant_name="standard",
    )
    assert leak_pilot["passed"] is False
    assert len(leak_pilot["vs_pilot_violations"]) == 1
    assert leak_pilot["vs_pilot_violations"][0]["hamming_distance"] == 1

    # Sanity check ham1_fen vs train_fen also triggers
    leak_h1 = radius_two_audit(
        roots_by_split={
            "train": [{"family_id": "f_tr", "fen": train_fen, "ply": 20}],
            "val": [{"family_id": "f_va", "fen": ham1_fen, "ply": 20}],
            "test": [],
        },
        demo_pools=demo_pools,
        pilot_paths=[pilot_dir],
        variant_name="standard",
    )
    assert leak_h1["passed"] is False
    assert leak_h1["cross_split_violations"][0]["hamming_distance"] == 1


# ---------------------------------------------------------------------------
# 4. End-to-End Pipeline: initial, train_wave, freeze_snapshot, r256_subset
# ---------------------------------------------------------------------------


def test_long_study_pipeline_initial_wave_snapshot_and_r256(tmp_path: Path):
    data_dir = tmp_path / "long_study"
    pilot_dir = tmp_path / "pilot_stub"
    _write_jsonl(
        pilot_dir / "sample_games.jsonl",
        [{"variant": "standard", "moves": ["h2h4", "h7h6", "h1h3", "h8h7"]}],
    )

    variants = ["standard", "diagonal_rook"]
    manifest = run_initial_generation(
        output_dir=data_dir,
        variants=variants,
        policy_id="terminal_aware_v1",
        r_train=4,
        r_eval=4,
        initial_budget=40,
        extend_step_plies=40,
        max_total_plies=160,
        max_extensions=3,
        demo_train_families=4,
        demo_val_families=3,
        demo_test_families=3,
        train_families=6,
        val_families=4,
        test_families=4,
        min_ply=8,
        workers=2,
        seed=2026112501,
        val_first=True,
        pilot_paths=[pilot_dir],
        check_pilot=True,
    )
    assert manifest["audit_passed"] is True
    assert manifest["rules_version"] == RULES_VERSION
    assert manifest["throughput"]["games_per_sec"] > 0.0
    assert manifest["throughput"]["continuations_per_sec"] > 0.0

    for var in variants:
        v_dir = data_dir / var
        assert (v_dir / "games_demo_train.jsonl").exists()
        assert (v_dir / "games_demo_val.jsonl").exists()
        assert (v_dir / "games_demo_test.jsonl").exists()
        assert (v_dir / "positions_train.jsonl").exists()
        assert (v_dir / "positions_val.jsonl").exists()
        assert (v_dir / "positions_test.jsonl").exists()

        for sp in ("train", "val", "test"):
            recs = _load_jsonl(v_dir / f"positions_{sp}.jsonl")
            assert all(r["n_unresolved"] == 0 and r["label_probs"] is not None for r in recs)
            by_fam = {}
            for r in recs:
                by_fam.setdefault(r["family_id"], []).append(r)
            assert all(len(sibs) <= 2 for sibs in by_fam.values())

    # Snapshot untouched val/test bytes before train_wave
    val_before = {
        v: (data_dir / v / "positions_val.jsonl").read_bytes() for v in variants
    }
    test_before = {
        v: (data_dir / v / "positions_test.jsonl").read_bytes() for v in variants
    }
    demo_val_before = {
        v: (data_dir / v / "games_demo_val.jsonl").read_bytes() for v in variants
    }
    demo_test_before = {
        v: (data_dir / v / "games_demo_test.jsonl").read_bytes() for v in variants
    }

    wave_manifest = run_train_wave(
        output_dir=data_dir,
        wave_id=1,
        variants=variants,
        r_train=4,
        initial_budget=40,
        extend_step_plies=40,
        max_total_plies=160,
        max_extensions=3,
        train_families=3,
        train_demo_games=0,
        min_ply=8,
        workers=2,
        seed=2026112501,
        pilot_paths=[pilot_dir],
        check_pilot=True,
    )
    assert wave_manifest["audit_passed"] is True
    for v in variants:
        assert (data_dir / v / "positions_val.jsonl").read_bytes() == val_before[v]
        assert (data_dir / v / "positions_test.jsonl").read_bytes() == test_before[v]
        assert (data_dir / v / "games_demo_val.jsonl").read_bytes() == demo_val_before[v]
        assert (data_dir / v / "games_demo_test.jsonl").read_bytes() == demo_test_before[v]

    # Freeze immutable checksummed snapshot with early val panel
    snap_dir = data_dir / "snapshots" / "2026112501"
    snap_manifest = freeze_snapshot(
        source_dir=data_dir,
        snapshot_dir=snap_dir,
        snapshot_id="2026112501",
        variants=variants,
        panel_families=2,
        pilot_paths=[pilot_dir],
        check_pilot=True,
    )
    assert snap_manifest["audit_passed"] is True
    assert (snap_dir / "snapshot_manifest.json").exists()
    assert (snap_dir / "val_panel_manifest.json").exists()

    # Verify checksums, line counts, and read-only permissions
    ver = verify_snapshot(snap_dir)
    assert ver["passed"] is True
    assert ver["n_checked"] == snap_manifest["n_files"]
    pos_mode = stat.S_IMODE((snap_dir / "standard" / "positions_test.jsonl").stat().st_mode)
    assert pos_mode == 0o444

    # Verify early val panel selected strictly by sha256(family_id) with all retained siblings
    val_panel = json.loads((snap_dir / "val_panel_manifest.json").read_text(encoding="utf-8"))
    for v in variants:
        val_recs = _load_jsonl(snap_dir / v / "positions_val.jsonl")
        fams = sorted(
            {r["family_id"] for r in val_recs},
            key=lambda fid: (hashlib.sha256(fid.encode("utf-8")).hexdigest(), fid),
        )
        expected_fams = fams[:2]
        assert [f["family_id"] for f in val_panel["variants"][v]["selected_families"]] == expected_fams

    # R=256 subset (use R=16 in unit test for speed)
    r256_out = tmp_path / "r256_out"
    r256_man = run_r256_subset(
        source_dir=snap_dir,
        out_dir=r256_out,
        variants=variants,
        subset_families=2,
        r_eval=16,
        initial_budget=40,
        extend_step_plies=40,
        max_total_plies=160,
        max_extensions=3,
        workers=2,
    )
    assert r256_man["R"] == 16
    for v in variants:
        sub_file = r256_out / v / "positions_test_r256_subset.jsonl"
        assert sub_file.exists()
        sub_recs = _load_jsonl(sub_file)
        assert all(r["R"] == 16 for r in sub_recs)

    # Snapshot remains completely intact after r256_subset
    assert verify_snapshot(snap_dir)["passed"] is True

    # Tamper detection test: make one snapshot file writable, alter 1 byte, verify checksum fails
    tampered = snap_dir / "standard" / "positions_val.jsonl"
    os.chmod(tampered, 0o644)
    tampered.write_text(tampered.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    tamper_ver = verify_snapshot(snap_dir)
    assert tamper_ver["passed"] is False
    assert any("positions_val.jsonl" in m for m in tamper_ver["mismatches"])
    os.chmod(tampered, 0o444)


def test_censoring_file_created_when_unresolved_occurs(tmp_path: Path):
    out_dir = tmp_path / "censored_run"
    manifest = run_initial_generation(
        output_dir=out_dir,
        variants=["standard"],
        policy_id="terminal_aware_v1",
        r_train=4,
        r_eval=4,
        initial_budget=120,
        extend_step_plies=120,
        max_total_plies=360,
        max_extensions=2,
        demo_train_families=2,
        demo_val_families=2,
        demo_test_families=2,
        train_families=3,
        val_families=2,
        test_families=2,
        min_ply=8,
        workers=2,
        seed=12345,
        check_pilot=False,
        rollout_initial_budget=1,
        rollout_max_total_plies=1,
        rollout_max_extensions=0,
    )
    v_dir = out_dir / "standard"
    tr_censored_file = v_dir / "positions_train_all_with_censoring.jsonl"
    assert tr_censored_file.exists()
    all_tr = _load_jsonl(tr_censored_file)
    resolved_tr = _load_jsonl(v_dir / "positions_train.jsonl")
    assert len(all_tr) > len(resolved_tr)
    assert all(r["n_unresolved"] == 0 and r["label_probs"] is not None for r in resolved_tr)
    censored_only = [r for r in all_tr if r["n_unresolved"] > 0]
    assert len(censored_only) > 0
    for rec in censored_only:
        assert rec["label_probs"] is None
        assert rec["label_probs_censored"] is None
        assert rec["label"] == "unresolved"
        w, d, l = rec["counts"]
        u = rec["n_unresolved"]
        r_tot = rec["R"]
        assert rec["target_bounds"]["lower"] == [w / r_tot, d / r_tot, l / r_tot]
        assert rec["target_bounds"]["upper"] == [(w + u) / r_tot, (d + u) / r_tot, (l + u) / r_tot]
    assert manifest["variants"]["standard"]["splits"]["train"]["n_censored"] == len(censored_only)
