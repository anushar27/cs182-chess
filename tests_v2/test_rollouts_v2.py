"""Unit tests for chess_icl_v2.rollouts policies, game generation, and continuations."""

from __future__ import annotations

import random

import chess
import pytest

from chess_icl_v2.rollouts import POLICY_IDS, choose_move, play_full_game, rollout_continuations
from chess_icl_v2.variants import ALL_VARIANT_NAMES, VariantBoard, get_board


def test_policy_ids_present():
    assert POLICY_IDS == ("uniform_legal_v1", "terminal_aware_v1", "search_d1_v1")


def test_terminal_aware_and_search_d1_find_mate_in_1():
    # Mate-in-1 (only Qxf7# or Bxf7# are checkmate)
    fen_mate1 = "3rkb2/3ppp2/8/8/2B5/5Q2/8/4K3 w - - 0 1"
    b = get_board("standard", fen=fen_mate1)
    mates = {chess.Move.from_uci("f3f7"), chess.Move.from_uci("c4f7")}
    for pid in ("terminal_aware_v1", "search_d1_v1"):
        rng = random.Random(42)
        m = choose_move(b, pid, rng)
        assert m in mates

    # Variant-specific mate-in-1: DiagonalRook (Ka8, Kb6, Rd6, Ra4 -> a4c6#)
    diag_fen = "k7/8/1K1R4/8/R7/8/8/8 w - - 0 1"
    b_diag = get_board("diagonal_rook", fen=diag_fen)
    assert not b_diag.was_into_check()
    for pid in ("terminal_aware_v1", "search_d1_v1"):
        rng = random.Random(7)
        m = choose_move(b_diag, pid, rng)
        assert m == chess.Move.from_uci("a4c6")


def test_search_d1_avoids_blundering_mate_in_1():
    # White King on h1, Black Rook on a2, Black King on e8.
    # If White plays a move leaving backrank open? Actually Black Rook on c2, White Rook on b1 protecting?
    # Better: White King on a1, White pawn on b2? No: White King on a1, White Queen on b2, Black Rook on h2, Black King on h8.
    # If White Queen moves off rank 2 (e.g. Qb2-a3), Black plays Rh2-h1#? No, Qa3 doesn't guard b1? Wait Qa1?
    # Simpler: White King on g1, pawns f2,g2,h2; Black Rook on a2, Black Queen on d5;
    # No, simplest: White King on h1, Black King on h3, White Rook on g8:
    # Actually White King on a1, Black King on c2, White Rook on b3 (if Rb3 moves away from 1st rank, Black Kc2? No, Black Rook on h2):
    # White King on a1, Black King on c1? Illegal.
    # White King on a1, Black Rook on b8, Black King on b3 -> if White plays any move except blocking b-file or moving to b1?
    # Wait: White King on h1, White Bishop on f1, Black Rook on a1? No, that's check.
    # White King on a1, Black Queen on c2, White Bishop on b1? No, a1, c2 is knight-jump away:
    # If White moves Kb1 -> Qxc2? No. If White moves Ba2 -> Qb2# (Queen on c2 moves to b2 protected by... wait no).
    # Put Black King on c3, Black Queen on c2, White King on a1, White Pawn on h2.
    # If White plays h2h3, Black plays Qc2-b2# (or Qc1#)! Wait: White has only Kh1/Ka2/etc.
    # Even simpler: White King on a1, White Knight on b1, Black Queen on d2, Black King on c3.
    # If White moves Nb1 away (allowing Qd2-b2# or Qd2-a2#? Wait b1 guards a2/b2? No, Knight on b1 doesn't guard b2!).
    # Wait: White King on a1, White Knight on c2, Black Queen on d4, Black King on b3 -> if White plays Nc2-a3 (unblocking c2), Qd4-b2# or Qd4-c1#?
    # Let's construct a 2-move position and verify directly:
    # White King on a1, White Rook on b2, Black Rook on h1? No, Black Rook on h2, Black King on c3.
    # If White Rook on b2 moves off file b/rank 2 (say Rb2-d2? No, Rb2-c2 blocks), if Rb2-b8, Black plays Rh2-h1# (checkmate!).
    # Whereas if White plays Rb2-h2 (capturing Black Rook!) or Ka1-b1, Black does NOT have mate-in-1!
    fen = "8/8/8/8/8/2k5/1R5r/K7 w - - 0 1"
    # Verify: Black Rh2 can't mate on h1 if Rb2 doesn't capture, wait: Ka1-b1 -> Rh2-h1#? No, Rb2 blocks? No!
    # Wait: if White plays Rb2xh2, Black has only Kc3 -> no mate!
    # If White plays Rb2-b3+ (check!), Black has no mate!
    # If White plays Rb2-a2 (leaving h2-h1 open? Wait: on a2, Ka1 is trapped on a1, Black Rh2-h1# is checkmate if Rb2 moved to e.g. b8!)
    # Let's check: if Rb2 moves to b8 -> Black plays Rh2-h1#? No, Rb8 doesn't block, Kc3 guards b2, Rh1 gives checkmate!
    b = get_board("standard", fen=fen)
    rng = random.Random(0)
    chosen = choose_move(b, "search_d1_v1", rng)
    assert chosen is not None
    b.push(chosen)
    # Confirm opponent has NO mate-in-1 after search_d1_v1's move!
    for opp_m in b.legal_moves:
        b.push(opp_m)
        assert not b.is_checkmate(), f"search_d1_v1 blundered mate-in-1 with {chosen.uci()} -> {opp_m.uci()}"
        b.pop()


@pytest.mark.parametrize("variant_name", ALL_VARIANT_NAMES)
def test_policies_and_san_all_variants(variant_name: str):
    for policy_id in POLICY_IDS:
        game = play_full_game(
            variant_name,
            policy_id,
            max_plies=10,
            seed=123,
            collect_san=True,
            collect_states=True,
            min_sample_ply=2,
        )
        assert game["variant"] == variant_name
        assert game["policy"] == policy_id
        assert game["result"] in ("win", "draw", "loss")
        assert len(game["moves"]) == len(game["moves_san"]) == game["move_count"]

        # Verify every recorded SAN parses cleanly on a replayed board
        replay = get_board(
            variant_name,
            fen=game["init_fen"],
            seed=123,
            blocked=game["blocked_squares"] if game["blocked_squares"] else None,
        )
        for uci, san in zip(game["moves"], game["moves_san"]):
            m_uci = chess.Move.from_uci(uci)
            assert m_uci in replay.legal_moves
            # SAN should parse back to the exact same move
            m_san = replay.parse_san(san)
            assert m_san == m_uci
            replay.push(m_uci)


def test_rollout_continuations_perspective_symmetry():
    # Mate-in-1 for White (White to move): terminal_aware_v1 wins 100% of R continuations
    fen_white_mate = "4k3/3ppp2/8/8/2B5/5Q2/8/4K3 w - - 0 1"
    b_w = get_board("standard", fen=fen_white_mate)
    res_w = rollout_continuations(
        "standard", b_w.to_state_dict(), "terminal_aware_v1", R=8, max_plies=20, seed=1
    )
    assert res_w["root_turn"] == "w"
    assert res_w["label_probs"] == [1.0, 0.0, 0.0]
    assert res_w["label_probs_censored"] == [1.0, 0.0, 0.0]
    assert res_w["horizon_rate"] == 0.0
    assert res_w["decisive_rate"] == 1.0

    # Mirror to Black to move: Black must STILL have [1.0, 0.0, 0.0] from Black's perspective!
    b_b = b_w.mirror()
    res_b = rollout_continuations(
        "standard", b_b.to_state_dict(), "terminal_aware_v1", R=8, max_plies=20, seed=1
    )
    assert res_b["root_turn"] == "b"
    assert res_b["label_probs"] == [1.0, 0.0, 0.0]

    # Horizon hit behavior (max_plies=1 on starting position hits horizon 100% of the time)
    b_start = get_board("standard")
    res_h = rollout_continuations(
        "standard", b_start.to_state_dict(), "uniform_legal_v1", R=4, max_plies=1, seed=5
    )
    assert res_h["horizon_rate"] == 1.0
    assert res_h["label_probs"] == [0.0, 1.0, 0.0]
    assert res_h["label_probs_censored"] is None
