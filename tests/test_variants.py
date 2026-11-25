"""Unit tests for custom chess variant engine."""

import pytest
import chess
from chess_icl.variants import VARIANTS, get_board, TRAIN_VARIANTS, TEST_VARIANTS


def test_registry():
    """All 13 variants must be present in the registry."""
    assert len(VARIANTS) == 13
    assert set(TRAIN_VARIANTS + TEST_VARIANTS) == set(VARIANTS.keys())
    assert len(TRAIN_VARIANTS) == 4
    assert len(TEST_VARIANTS) == 9


def test_diagonal_rook():
    """Rook moves diagonally, not straight."""
    board = get_board("diagonal_rook")
    # In starting position, rooks are trapped behind pawns, but e2e4 opens bishop
    board.board.set_piece_at(chess.E4, chess.Piece(chess.ROOK, chess.WHITE))
    legal = board.legal_moves()
    rook_moves = [m for m in legal if m.from_square == chess.E4]

    # e4 rook should have diagonal moves (e.g., e4f5, e4d5, e4f3, e4d3)
    # and NOT orthogonal moves (e4e5, e4e3, etc.)
    diagonal_dests = {chess.F5, chess.D5, chess.F3, chess.D3, chess.G6, chess.C6}
    straight_dests = {chess.E5, chess.E6, chess.E3, chess.D4, chess.F4}

    rook_dests = {m.to_square for m in rook_moves}
    assert any(sq in rook_dests for sq in diagonal_dests)
    assert not any(sq in rook_dests for sq in straight_dests)


def test_straight_bishop():
    """Bishop moves in straight lines, not diagonally."""
    board = get_board("straight_bishop")
    board.board.set_piece_at(chess.E4, chess.Piece(chess.BISHOP, chess.WHITE))
    legal = board.legal_moves()
    bishop_moves = [m for m in legal if m.from_square == chess.E4]

    straight_dests = {chess.E5, chess.E6, chess.E3, chess.D4, chess.F4}
    diagonal_dests = {chess.F5, chess.D5, chess.F3, chess.D3}

    bishop_dests = {m.to_square for m in bishop_moves}
    assert any(sq in bishop_dests for sq in straight_dests)
    assert not any(sq in bishop_dests for sq in diagonal_dests)


def test_weakened_queen():
    """Queen moves like a rook only, loses diagonal movement."""
    board = get_board("weakened_queen")
    board.board.set_piece_at(chess.E4, chess.Piece(chess.QUEEN, chess.WHITE))
    legal = board.legal_moves()
    queen_moves = [m for m in legal if m.from_square == chess.E4]

    queen_dests = {m.to_square for m in queen_moves}
    assert chess.E5 in queen_dests  # straight move
    assert chess.F5 not in queen_dests  # diagonal move


def test_extended_pawn():
    """Pawns can move 2 squares from non-starting ranks."""
    board = get_board("extended_pawn")
    # Place a pawn on e3 (normally can only move to e4)
    board.board.set_piece_at(chess.E3, chess.Piece(chess.PAWN, chess.WHITE))
    legal = board.legal_moves()
    pawn_moves = [m for m in legal if m.from_square == chess.E3]
    pawn_dests = {m.to_square for m in pawn_moves}

    assert chess.E4 in pawn_dests
    assert chess.E5 in pawn_dests  # 2 squares forward from rank 3!


def test_piece_capture_conversion():
    """Captured piece converts to capturer's color."""
    board = get_board("piece_capture")
    # Setup white pawn on e4, black pawn on d5
    board.board.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.WHITE))
    board.board.set_piece_at(chess.D5, chess.Piece(chess.PAWN, chess.BLACK))

    move = chess.Move(chess.E4, chess.D5)
    assert move in board.legal_moves()

    board.push(move)
    # The piece on d5 should now be a WHITE pawn!
    piece_at_d5 = board.board.piece_at(chess.D5)
    assert piece_at_d5 is not None
    assert piece_at_d5.color == chess.WHITE
    assert board.board.piece_at(chess.E4) is None


def test_forward_pawn_capture():
    """Pawns can capture directly forward."""
    board = get_board("forward_pawn_capture")
    board.board.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.WHITE))
    board.board.set_piece_at(chess.E5, chess.Piece(chess.PAWN, chess.BLACK))

    move = chess.Move(chess.E4, chess.E5)
    assert move in board.legal_moves()


def test_dark_square_immunity():
    """Pieces on dark squares cannot be captured."""
    board = get_board("dark_square_immunity")
    # c3 is a dark square ((rank 2 + file 2) % 2 == 0)
    # Put a black piece on c3 and white pawn on b2
    board.board.set_piece_at(chess.B2, chess.Piece(chess.PAWN, chess.WHITE))
    board.board.set_piece_at(chess.C3, chess.Piece(chess.KNIGHT, chess.BLACK))

    # b2-c3 is a diagonal capture onto c3
    # c3 is dark square -> capture should be illegal
    move = chess.Move(chess.B2, chess.C3)
    assert move not in board.legal_moves()


def test_blocked_squares():
    """Impassable blocked squares cannot be entered or passed through."""
    board = get_board("blocked_squares", seed=42)
    assert len(board.blocked) == 8
    for sq in board.blocked:
        # No legal move should land on a blocked square
        for m in board.legal_moves():
            assert m.to_square != sq


def test_knight_army():
    """Starting board has 15 knights and 1 king per side."""
    board = get_board("knight_army")
    white_knights = sum(
        1 for sq in chess.SQUARES
        if board.board.piece_at(sq) == chess.Piece(chess.KNIGHT, chess.WHITE)
    )
    black_knights = sum(
        1 for sq in chess.SQUARES
        if board.board.piece_at(sq) == chess.Piece(chess.KNIGHT, chess.BLACK)
    )
    assert white_knights == 15
    assert black_knights == 15
    assert board.board.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE)
    assert board.board.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK)


def test_antichess_forced_capture():
    """Forced capture: if capture is available, only captures are legal."""
    board = get_board("antichess")
    # In starting position, no captures available -> pseudo-legal moves returned
    legal = board.legal_moves()
    assert len(legal) > 0

    # Put a white pawn on e4 and black pawn on d5 -> capture is available!
    board.board.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.WHITE))
    board.board.set_piece_at(chess.D5, chess.Piece(chess.PAWN, chess.BLACK))

    legal_with_capture = board.legal_moves()
    # All legal moves MUST be captures
    for m in legal_with_capture:
        assert board.board.is_capture(m)
