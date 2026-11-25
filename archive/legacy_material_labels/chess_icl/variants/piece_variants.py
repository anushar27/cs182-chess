"""Piece variants: change the starting army composition."""

import chess
from .engine import VariantBoard, VariantRule


class KnightArmyBoard(VariantBoard):
    """Each side: 15 knights + 1 king. Standard 8x8 board.
    
    White: King on e1, Knights on all other squares of rows 1-2.
    Black: King on e8, Knights on all other squares of rows 7-8.
    """

    def _setup(self):
        self.board.clear()

        # White pieces: rows 0-1 (ranks 1-2)
        for sq in range(16):  # a1-h2
            if sq == chess.E1:
                self.board.set_piece_at(sq, chess.Piece(chess.KING, chess.WHITE))
            else:
                self.board.set_piece_at(sq, chess.Piece(chess.KNIGHT, chess.WHITE))

        # Black pieces: rows 6-7 (ranks 7-8)
        for sq in range(48, 64):  # a7-h8
            if sq == chess.E8:
                self.board.set_piece_at(sq, chess.Piece(chess.KING, chess.BLACK))
            else:
                self.board.set_piece_at(sq, chess.Piece(chess.KNIGHT, chess.BLACK))

        # Reset castling rights (no rooks = no castling)
        self.board.set_castling_fen("-")
        # Reset move counters
        self.board.halfmove_clock = 0
        self.board.fullmove_number = 1
