"""Thorough unit tests, v1 bug-regression tests, differential fuzzing, and property audits for chess_icl_v2.variants."""

from __future__ import annotations

import chess
import chess.variant
import pytest

from chess_icl_v2.variants import (
    ALL_VARIANT_NAMES,
    ALL_VARIANTS,
    COMPOSITIONAL_VARIANTS,
    PIECE_VALUES,
    SINGLE_VARIANTS,
    TEST_VARIANTS,
    TRAIN_VARIANTS,
    VARIANTS,
    VariantBoard,
    VariantCategory,
    get_board,
)
from chess_icl_v2.variants.audit import run_differential_fuzz, run_reference_audit


# ── Registry & Specs ─────────────────────────────────────────────────────────


def test_registry_and_specs():
    assert len(VARIANTS) == 16  # 13 single + 3 compositional
    assert len(ALL_VARIANTS) == 17
    assert len(TRAIN_VARIANTS) == 4
    assert len(TEST_VARIANTS) == 9
    assert len(SINGLE_VARIANTS) == 13
    assert len(COMPOSITIONAL_VARIANTS) == 3
    assert set(ALL_VARIANT_NAMES) == set(ALL_VARIANTS.keys())

    assert ALL_VARIANTS["knight_army"].is_setup_only_control is True
    assert ALL_VARIANTS["knight_army"].is_visibly_identifiable is True
    assert ALL_VARIANTS["blocked_squares"].uses_blocked_squares is True
    assert ALL_VARIANTS["blocked_squares"].is_visibly_identifiable is True
    assert ALL_VARIANTS["antichess"].no_check_rules is True
    assert ALL_VARIANTS["short_knight_and_antichess"].no_check_rules is True
    assert ALL_VARIANTS["short_knight_and_antichess"].is_compositional is True
    assert ALL_VARIANTS["short_knight_and_antichess"].components == ("short_knight", "antichess")
    assert ALL_VARIANTS["diagonal_rook"].category == VariantCategory.MOVE


def test_evaluate_material_and_board_matrix():
    board = get_board("standard")
    assert board.evaluate_material(chess.WHITE) == 0
    assert board.evaluate_material(chess.BLACK) == 0
    board.remove_piece_at(chess.D8)  # remove black queen (-9)
    assert board.evaluate_material(chess.WHITE) == PIECE_VALUES[chess.QUEEN]
    assert board.evaluate_material(chess.BLACK) == -PIECE_VALUES[chess.QUEEN]

    mat = board.board_matrix_64()
    assert len(mat) == 64
    assert mat[chess.E1] == "K"
    assert mat[chess.D8] == "."
    assert board.board_fen_with_barriers() == board.board_fen()


# ── Fix 1: Variant-Aware Check, Attack, Pin, Checkmate & Stalemate ───────────


def test_diagonal_rook_check_pin_and_insufficient_material():
    # Diagonal rook gives check along diagonal, NOT along rank/file
    b = get_board("diagonal_rook", fen="4k3/8/2R5/8/8/8/8/4K3 b - - 0 1")
    assert b.is_check()
    assert bool(b.checkers_mask() & chess.BB_C6)
    assert chess.Move.from_uci("e8d7") not in b.legal_moves  # moving along diagonal ray

    # Rook on same file as enemy king does NOT give check
    b2 = get_board("diagonal_rook", fen="4k3/8/4R3/8/8/8/8/4K3 b - - 0 1")
    assert not b2.is_check()

    # Diagonal pin: Black knight on d7 is pinned to e8 by White rook on c6
    b3 = get_board("diagonal_rook", fen="4k3/3n4/2R5/8/8/8/8/4K3 b - - 0 1")
    assert not b3.is_check()
    knight_moves = [m for m in b3.legal_moves if m.from_square == chess.D7]
    assert knight_moves == []

    # Diagonal rook checkmate in corner: Black Ka8, White Kb6 (guards a7/b7), Rc6 (checks a8), Rd6 (guards b8)
    b_mate = get_board("diagonal_rook", fen="k7/8/1KRR4/8/8/8/8/8 b - - 0 1")
    assert b_mate.is_check()
    assert not any(b_mate.legal_moves)
    assert b_mate.is_checkmate()
    assert b_mate.outcome().winner == chess.WHITE

    # Stalemate with diagonal rooks (opposite colors so sufficient material):
    # Black Ka8, White Kb6 (guards a7,b7), White Rc7 (dark, guards b8), White Ra6 (light, guards b7)
    b_stale = get_board("diagonal_rook", fen="k7/2R5/RK6/8/8/8/8/8 b - - 0 1")
    assert not b_stale.is_check()
    assert b_stale.is_stalemate()
    assert b_stale.outcome().termination == chess.Termination.STALEMATE

    # K + single diagonal rook vs K is insufficient material (like K+B vs K)
    b_insuff = get_board("diagonal_rook", fen="4k3/8/8/8/3R4/8/8/4K3 w - - 0 1")
    assert b_insuff.has_insufficient_material(chess.WHITE)
    assert b_insuff.outcome().termination == chess.Termination.INSUFFICIENT_MATERIAL


def test_straight_bishop_check_and_sufficient_material():
    # Bishop checks along file, NOT along diagonal
    b = get_board("straight_bishop", fen="4k3/8/4B3/8/8/8/8/4K3 b - - 0 1")
    assert b.is_check()
    assert bool(b.checkers_mask() & chess.BB_E6)

    b_diag = get_board("straight_bishop", fen="4k3/8/2B5/8/8/8/8/4K3 b - - 0 1")
    assert not b_diag.is_check()

    # K + straight bishop vs K CAN mate (like K+R vs K), so NOT insufficient material
    b_suff = get_board("straight_bishop", fen="4k3/8/8/8/3B4/8/8/4K3 w - - 0 1")
    assert not b_suff.has_insufficient_material(chess.WHITE)
    assert b_suff.outcome() is None


def test_weakened_queen_no_diagonal_check():
    b_diag = get_board("weakened_queen", fen="4k3/8/2Q5/8/8/8/8/4K3 b - - 0 1")
    assert not b_diag.is_check()
    assert chess.Move.from_uci("c6e8") not in b_diag.legal_moves

    b_orth = get_board("weakened_queen", fen="4k3/8/4Q3/8/8/8/8/4K3 b - - 0 1")
    assert b_orth.is_check()


def test_extended_pawn_moves_and_ep():
    b = get_board("extended_pawn", fen="4k3/8/8/8/8/4P3/8/4K3 w - - 0 1")
    dests = {m.uci() for m in b.legal_moves if m.from_square == chess.E3}
    assert dests == {"e3e4", "e3e5"}

    # Pushing 2 squares from rank 3 does not set ep_square; pushing 2 from rank 2 does
    b.push(chess.Move.from_uci("e3e5"))
    assert b.ep_square is None

    b2 = get_board("extended_pawn", fen="4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    b2.push(chess.Move.from_uci("e2e4"))
    assert b2.ep_square == chess.E3

    # 2-square push onto promotion rank yields 4 promotions (keep e8 empty)
    b3 = get_board("extended_pawn", fen="k7/8/4P3/8/8/8/8/4K3 w - - 0 1")
    promos = {m.uci() for m in b3.legal_moves if m.from_square == chess.E6}
    assert promos == {"e6e7", "e6e8q", "e6e8r", "e6e8b", "e6e8n"}


def test_short_knight_ferz_dabbaba_and_checks():
    b = get_board("short_knight", fen="4k3/8/8/8/3N4/8/8/4K3 w - - 0 1")
    n_dests = {chess.SQUARE_NAMES[m.to_square] for m in b.legal_moves if m.from_square == chess.D4}
    assert n_dests == {"c5", "e5", "c3", "e3", "d6", "d2", "b4", "f4"}

    # Check via dabbaba (0, +2) and ferz (+1, +1)
    b_dab = get_board("short_knight", fen="4k3/8/4N3/8/8/8/8/4K3 b - - 0 1")
    assert b_dab.is_check()

    b_ferz = get_board("short_knight", fen="4k3/3N4/8/8/8/8/8/4K3 b - - 0 1")
    assert b_ferz.is_check()

    # Standard (2, 1) knight jump does NOT give check
    b_std = get_board("short_knight", fen="4k3/8/3N4/8/8/8/8/4K3 b - - 0 1")
    assert not b_std.is_check()


def test_mobile_king_queen_checks_and_castling_disambiguation():
    # Mobile king gives check across the board along a file/diagonal
    b = get_board("mobile_king", fen="4k3/8/8/8/8/8/8/4K3 b - - 0 1")
    assert b.is_check()

    # Cannot move onto ray of opponent's mobile king
    b2 = get_board("mobile_king", fen="4k3/4p3/8/8/8/8/8/4K3 w - - 0 1")
    assert chess.Move.from_uci("e1f2") in b2.legal_moves
    assert chess.Move.from_uci("e1a4") not in b2.legal_moves  # a4 is on diagonal from e8!

    # Castling works when rights exist; when rights lost, e1g1 and e1h1 are plain king slides
    b_castle = get_board("mobile_king", fen="4k3/p7/8/8/8/8/PPPPPPPP/R3K2R w KQ - 0 1")
    # Put Black king behind pawn on a7? No, e8->e1 blocked by e2 pawn; check g1/c1 castling
    assert b_castle.is_castling(chess.Move.from_uci("e1g1"))
    b_castle.push(chess.Move.from_uci("e1g1"))
    assert b_castle.piece_at(chess.G1) == chess.Piece(chess.KING, chess.WHITE)
    assert b_castle.piece_at(chess.F1) == chess.Piece(chess.ROOK, chess.WHITE)

    # Without castling rights and empty h1, e1h1 is a valid king slide (not castling!)
    b_slide = get_board("mobile_king", fen="k7/8/8/8/8/8/PPPPPPPP/4K3 w - - 0 1")
    m_h1 = chess.Move.from_uci("e1h1")
    assert m_h1 in b_slide.legal_moves
    assert not b_slide.is_castling(m_h1)
    b_slide.push(m_h1)
    assert b_slide.piece_at(chess.H1) == chess.Piece(chess.KING, chess.WHITE)
    assert b_slide.piece_at(chess.G1) is None
    assert b_slide.move_stack[-1] == m_h1


def test_forward_pawn_capture_and_check():
    b = get_board("forward_pawn_capture", fen="4k3/4P3/8/8/8/8/8/4K3 b - - 0 1")
    assert b.is_check()
    # White pawn on e7 attacks d8, f8 diagonally AND e8 forward; d7, f7, e7 are safe.
    legal_ucis = {m.uci() for m in b.legal_moves}
    assert legal_ucis == {"e8e7", "e8d7", "e8f7"}


def test_dark_square_immunity_non_king_immune_king_vulnerable():
    # c3 is dark square ((2+2)%2 == 0). Black knight on c3 cannot be captured,
    # and Black knight on c3 checking White king on b1 CANNOT be captured by d1/a2 etc.!
    b = get_board("dark_square_immunity", fen="4k3/8/8/8/8/2n5/3P4/1K6 w - - 0 1")
    assert b.is_check()
    # Normally d2xc3 captures the checking knight, but c3 is a dark square -> illegal!
    assert chess.Move.from_uci("d2c3") not in b.legal_moves
    # King must flee instead
    assert chess.Move.from_uci("b1a1") in b.legal_moves

    # King on dark square (e1: file 4 + rank 0 = 4, dark!) IS STILL vulnerable to check/mate
    b_mate = get_board("dark_square_immunity", fen="4k3/8/8/8/8/8/4q3/4K3 w - - 0 1")
    # Black queen on e2 (light square) supported by... wait e2 is light so Kxe2 would be legal!
    # Put Black queen on d2 (dark square: 3+1=4) and Black rook on e8? No, Black queen on d2 + rook on a1 -> wait
    b_mate2 = get_board("dark_square_immunity", fen="4k3/8/8/8/8/8/3q4/r3K3 w - - 0 1")
    assert b_mate2.is_check()
    assert b_mate2.is_checkmate()

    # En passant onto light square when captured pawn is on dark square (e5: 4+4=8 dark) is blocked!
    b_ep = get_board("dark_square_immunity", fen="4k3/8/8/3Pp3/8/8/8/4K3 w - e6 0 1")
    assert chess.Move.from_uci("d5e6") not in b_ep.legal_moves


# ── Fix 2: Knight Army Setup-Only Null Control ───────────────────────────────


def test_knight_army_only_when_fen_none():
    b = get_board("knight_army")
    w_knights = sum(1 for sq in chess.SQUARES if b.piece_at(sq) == chess.Piece(chess.KNIGHT, chess.WHITE))
    b_knights = sum(1 for sq in chess.SQUARES if b.piece_at(sq) == chess.Piece(chess.KNIGHT, chess.BLACK))
    assert w_knights == 15
    assert b_knights == 15
    assert b.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE)
    assert b.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK)
    assert b.castling_rights == chess.BB_EMPTY

    custom_fen = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
    b_custom = get_board("knight_army", fen=custom_fen)
    assert b_custom.fen() == custom_fen
    assert b_custom.piece_at(chess.E2) == chess.Piece(chess.PAWN, chess.WHITE)


# ── Fix 3: Blocked Squares Barriers, Ray Blocking & Persistence ──────────────


def test_blocked_squares_ray_blocking_and_serialization():
    b1 = get_board("blocked_squares", seed=42)
    assert len(b1.blocked) == 8
    assert all(16 <= sq <= 47 for sq in b1.blocked)

    # Midgame FEN loading with same seed must yield exact same blocked squares
    mid_fen = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
    b2 = get_board("blocked_squares", fen=mid_fen, seed=42)
    assert b2.blocked == b1.blocked

    # Explicit ray-check blocking: White rook on e1, Black king on e8, barrier on e4
    b_ray = get_board("blocked_squares", fen="4k3/8/8/8/8/8/8/K3R3 w - - 0 1", blocked=[chess.E4, chess.A3, chess.B3, chess.C3, chess.D3, chess.F3, chess.G3, chess.H3])
    b_ray.turn = chess.BLACK
    assert not b_ray.is_check()  # e4 barrier blocks e1-e8 ray!
    b_ray.turn = chess.WHITE
    assert chess.Move.from_uci("e1e4") not in b_ray.legal_moves
    assert chess.Move.from_uci("e1e5") not in b_ray.legal_moves
    assert chess.Move.from_uci("e1e3") in b_ray.legal_moves

    # Knights can jump over blocked squares
    b_knight = get_board("blocked_squares", fen="4k3/8/8/8/8/8/4N3/4K3 w - - 0 1", blocked=[chess.E3, chess.D3, chess.F3, chess.A4, chess.B4, chess.C4, chess.G4, chess.H4])
    assert chess.Move.from_uci("e2f4") in b_knight.legal_moves  # f4 is unblocked, jumps past rank 3 barriers

    # Barrier visibility & round-trip
    mat = b_ray.board_matrix_64()
    assert mat[chess.E4] == "X"
    assert "X" in b_ray.board_fen_with_barriers()
    assert "X" in b_ray.canonical_state_key()

    restored = VariantBoard.from_state_dict(b_ray.to_state_dict())
    assert restored.blocked == b_ray.blocked
    assert restored.canonical_state_key() == b_ray.canonical_state_key()
    assert restored.fen() == b_ray.fen()


# ── Fix 4 & Fix 5: Antichess & Short Knight Antichess ────────────────────────


def test_antichess_matches_python_chess():
    b = get_board("antichess")
    ref = chess.variant.AntichessBoard()
    assert {m.uci() for m in b.legal_moves} == {m.uci() for m in ref.legal_moves}

    # Stalemate = win for stalemated side
    # Black pawn on a3 blocked by White pawn on a2, no other pieces -> Black has 0 moves -> Black wins!
    stale_fen = "8/8/8/8/8/p7/P7/8 b - - 0 1"
    b_stale = get_board("antichess", fen=stale_fen)
    ref_stale = chess.variant.AntichessBoard(stale_fen)
    assert b_stale.is_stalemate()
    assert b_stale.outcome() == ref_stale.outcome()
    assert b_stale.outcome().winner == chess.BLACK

    # King promotion allowed
    promo_fen = "8/4P3/8/8/8/8/8/k7 w - - 0 1"
    b_promo = get_board("antichess", fen=promo_fen)
    assert chess.Move.from_uci("e7e8k") in b_promo.legal_moves


def test_short_knight_and_antichess_ferz_dabbaba_and_forced_capture():
    # White knight on d4, Black pawn on d6 (dabbaba capture!) and standard L-jump c6 ignored
    b = get_board("short_knight_and_antichess", fen="8/8/2ppp3/8/3N4/8/8/8 w - - 0 1")
    legal_ucis = {m.uci() for m in b.legal_moves}
    # d4 can capture d6 via dabbaba (0, +2); c6 and e6 are standard knight jumps -> NOT legal!
    assert legal_ucis == {"d4d6"}

    # Ferz (1,1) capture is also mandatory
    b2 = get_board("short_knight_and_antichess", fen="8/8/8/4p3/3N4/8/8/8 w - - 0 1")
    assert {m.uci() for m in b2.legal_moves} == {"d4e5"}

    # Color-bound 2 short knights on opposite colors -> insufficient material draw
    b_insuff = get_board("short_knight_and_antichess", fen="8/8/8/8/8/8/8/N1n5 w - - 0 1")
    # Wait: a1 (0,0) dark, c1 (0,2) dark -> both dark! Test a1 (dark) vs b1 (light):
    assert not b_insuff.has_insufficient_material(chess.WHITE)
    b_opp = get_board("short_knight_and_antichess", fen="8/8/8/8/8/8/8/Nn6 w - - 0 1")
    assert b_opp.has_insufficient_material(chess.WHITE)
    assert b_opp.has_insufficient_material(chess.BLACK)
    assert b_opp.outcome().termination == chess.Termination.INSUFFICIENT_MATERIAL


# ── Fix 6: Piece Capture State Hygiene, EP, Castling & Termination ───────────


def test_piece_capture_state_updates_and_ep():
    # Capturing corner rook converts it to capturer's color AND clears opponent castling right
    b = get_board("piece_capture", fen="r3k2r/8/8/8/8/8/6B1/4K3 w kq - 5 10")
    # Put bishop on g2? Wait g2->a8 diagonal! g2,f3,e4,d5,c6,b7,a8!
    m = chess.Move.from_uci("g2a8")
    assert m in b.legal_moves
    b.push(m)
    assert b.piece_at(chess.A8) == chess.Piece(chess.ROOK, chess.WHITE)
    assert b.piece_at(chess.G2) is None
    assert b.halfmove_clock == 0
    assert not (b.castling_rights & chess.BB_A8)
    assert b.castling_rights & chess.BB_H8

    # Pop restores original state cleanly
    b.pop()
    assert b.piece_at(chess.A8) == chess.Piece(chess.ROOK, chess.BLACK)
    assert b.piece_at(chess.G2) == chess.Piece(chess.BISHOP, chess.WHITE)
    assert b.halfmove_clock == 5
    assert b.castling_rights & chess.BB_A8

    # En passant capture converts pawn onto ep_square and clears captured pawn
    b_ep = get_board("piece_capture", fen="4k3/8/8/3Pp3/8/8/8/4K3 w - e6 2 12")
    b_ep.push(chess.Move.from_uci("d5e6"))
    assert b_ep.piece_at(chess.E6) == chess.Piece(chess.PAWN, chess.WHITE)
    assert b_ep.piece_at(chess.E5) is None
    assert b_ep.piece_at(chess.D5) is None
    assert b_ep.halfmove_clock == 0

    # Missing king terminates immediately
    b_no_king = get_board("piece_capture", fen="8/8/8/8/8/8/8/4K3 b - - 0 1")
    assert b_no_king.is_game_over()
    assert b_no_king.outcome().winner == chess.WHITE


# ── Fix 7: Long Board Opening Pawn Push ──────────────────────────────────────


def test_long_board_opening_push_and_ep():
    b = get_board("long_board", fen="4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    e2_dests = {chess.SQUARE_NAMES[m.to_square] for m in b.legal_moves if m.from_square == chess.E2}
    # Can push 1..5 squares (e3..e7), NOT onto promotion rank e8
    assert e2_dests == {"e3", "e4", "e5", "e6", "e7"}

    # Pushing 5 squares does NOT set ep_square
    b.push(chess.Move.from_uci("e2e7"))
    assert b.ep_square is None
    b.pop()

    # Pushing 2 squares DOES set ep_square
    b.push(chess.Move.from_uci("e2e4"))
    assert b.ep_square == chess.E3


# ── Compositional Variants ───────────────────────────────────────────────────


def test_compositional_variants_smoke():
    b1 = get_board("diagonal_rook_and_piece_capture", fen="4k3/8/8/8/8/2n5/8/R3K3 w - - 0 1")
    b1.push(chess.Move.from_uci("a1c3"))
    assert b1.piece_at(chess.C3) == chess.Piece(chess.KNIGHT, chess.WHITE)

    b2 = get_board("mobile_king_and_dark_immunity", fen="4k3/8/8/8/8/2n5/8/4K3 w - - 0 1")
    assert chess.Move.from_uci("e1c3") not in b2.legal_moves  # c3 dark square immune
    assert chess.Move.from_uci("e1a5") not in b2.legal_moves  # blocked by c3 knight
    assert chess.Move.from_uci("e1h4") in b2.legal_moves  # unblocked queen slide of king


# ── Differential Fuzzing & Independent Reference Audits ──────────────────────


def test_differential_fuzz_standard_and_antichess():
    diff_res = run_differential_fuzz(num_positions=300, seed=2026)
    for name, res in diff_res.items():
        assert res.passed, f"{name} differential fuzz failed: {res.failures[:5]}"
        assert res.positions_checked == 300
        assert res.near_terminal_checked == 150


def test_independent_reference_and_property_audits():
    ref_res = run_reference_audit(
        variants=list(ALL_VARIANT_NAMES),
        trajectories_per_variant=12,
        plies_per_trajectory=12,
        near_terminal_per_variant=30,
        seed=2026,
    )
    for name, res in ref_res.items():
        assert res.passed, f"{name} reference/property audit failed: {res.failures[:5]}"
        assert res.replay_roundtrips == 12
        assert res.near_terminal_checked == 30
