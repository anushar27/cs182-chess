"""Objective variants: change the win condition."""

import chess
from .engine import VariantBoard, VariantRule


class AntichessBoard(VariantBoard):
    """Goal: lose all pieces. Must capture if possible.
    
    Rules:
    - If a capture is available, you MUST capture.
    - The first player to lose all their pieces wins.
    - No check/checkmate concept — the king is a regular piece.
    - Stalemate: the stalemated player wins (they can't move = good).
    """

    def legal_moves(self) -> list[chess.Move]:
        # In antichess, we ignore check/checkmate rules
        # Use pseudo-legal moves since king is not special
        all_moves = list(self.board.pseudo_legal_moves)
        captures = [m for m in all_moves if self.board.is_capture(m)]
        # Forced capture: if any capture is available, must capture
        return captures if captures else all_moves

    def is_game_over(self) -> bool:
        if self._move_count >= self._max_moves:
            return True

        # Count pieces for the side to move
        side = self.board.turn
        has_pieces = any(
            self.board.piece_at(sq) is not None
            and self.board.piece_at(sq).color == side
            for sq in chess.SQUARES
        )

        if not has_pieces:
            return True  # Current side has no pieces → they win!

        # Also check: if no legal moves (stalemate), game is over
        if not self.legal_moves():
            return True

        return False

    def result_for_white(self) -> str:
        """In antichess, losing all pieces = winning."""
        white_pieces = sum(
            1 for sq in chess.SQUARES
            if self.board.piece_at(sq) is not None
            and self.board.piece_at(sq).color == chess.WHITE
        )
        black_pieces = sum(
            1 for sq in chess.SQUARES
            if self.board.piece_at(sq) is not None
            and self.board.piece_at(sq).color == chess.BLACK
        )

        if white_pieces == 0:
            return "win"  # White lost all pieces → White wins in antichess
        if black_pieces == 0:
            return "loss"  # Black lost all pieces → Black wins

        # Stalemate: the stalemated player wins
        if not self.legal_moves():
            if self.board.turn == chess.WHITE:
                return "win"  # White can't move → White wins
            else:
                return "loss"  # Black can't move → Black wins

        if self._move_count >= self._max_moves:
            score = self.evaluate_material(chess.WHITE)
            # In antichess, having LESS material is winning
            if score <= -3:
                return "win"
            if score >= 3:
                return "loss"
            return "draw"

        return "draw"
