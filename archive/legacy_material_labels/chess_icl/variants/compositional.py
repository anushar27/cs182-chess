"""Compositional chess variants combining two orthogonal rule modifications.

Variants:
1. diagonal_rook_and_piece_capture: Rooks move diagonally AND captured pieces flip color
2. mobile_king_and_dark_immunity: Kings move like Queens AND pieces on dark squares are immune
3. short_knight_and_antichess: Knights move (1,1) ferz steps AND antichess rules (must capture, losing all pieces wins)
"""

import chess
from .engine import VariantBoard, VariantRule, VariantCategory
from .move_variants import DiagonalRookBoard, MobileKingBoard, ShortKnightBoard
from .capture_variants import PieceCaptureBoard, DarkSquareImmunityBoard
from .objective_variants import AntichessBoard


class DiagonalRookPieceCaptureBoard(DiagonalRookBoard, PieceCaptureBoard):
    """Composition of Diagonal Rook (moves) + Piece Capture (captured pieces convert)."""

    def legal_moves(self) -> list[chess.Move]:
        # Use DiagonalRookBoard move generation
        return DiagonalRookBoard.legal_moves(self)

    def _do_push(self, move: chess.Move):
        # Use PieceCaptureBoard capture conversion logic
        PieceCaptureBoard._do_push(self, move)

    def is_game_over(self) -> bool:
        return PieceCaptureBoard.is_game_over(self)


class MobileKingDarkImmunityBoard(MobileKingBoard, DarkSquareImmunityBoard):
    """Composition of Mobile King (moves like queen) + Dark Square Immunity (dark squares immune)."""

    def legal_moves(self) -> list[chess.Move]:
        # Generate queen moves for king, then filter out dark square captures
        color = self.board.turn
        moves = [
            m for m in self.board.pseudo_legal_moves
            if (p := self.board.piece_at(m.from_square)) is not None and p.piece_type != chess.KING
        ]
        queen_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for square in self.board.pieces(chess.KING, color):
            moves.extend(self._sliding_moves(square, queen_dirs, color))

        # Filter out captures on dark squares
        allowed_moves = []
        for m in moves:
            target_piece = self.board.piece_at(m.to_square)
            if target_piece is not None:
                # Dark square check: (file + rank) % 2 == 0 is dark square in 0-indexed chess
                file_idx = chess.square_file(m.to_square)
                rank_idx = chess.square_rank(m.to_square)
                if (file_idx + rank_idx) % 2 == 0:
                    continue  # Immune on dark square
            allowed_moves.append(m)

        return self._filter_legal(allowed_moves)


class ShortKnightAntichessBoard(AntichessBoard):
    """Composition of Short Knight ((1,1) diagonal step) + Antichess (mandatory capture, goal is to lose all pieces)."""

    def legal_moves(self) -> list[chess.Move]:
        color = self.board.turn
        pseudo_moves = [
            m for m in self.board.pseudo_legal_moves
            if (p := self.board.piece_at(m.from_square)) is not None and p.piece_type != chess.KNIGHT
        ]
        # Short knight (ferz 1-step diagonal)
        diag_dirs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        for square in self.board.pieces(chess.KNIGHT, color):
            file_idx = chess.square_file(square)
            rank_idx = chess.square_rank(square)
            for df, dr in diag_dirs:
                nf, nr = file_idx + df, rank_idx + dr
                if 0 <= nf < 8 and 0 <= nr < 8:
                    target_sq = chess.square(nf, nr)
                    dest_piece = self.board.piece_at(target_sq)
                    if dest_piece is None or dest_piece.color != color:
                        pseudo_moves.append(chess.Move(square, target_sq))

        # Antichess rule: captures are mandatory!
        captures = [
            m for m in pseudo_moves
            if self.board.piece_at(m.to_square) is not None or self.board.is_en_passant(m)
        ]
        if captures:
            return captures
        return pseudo_moves


COMPOSITIONAL_RULES = {
    "diagonal_rook_and_piece_capture": VariantRule(
        "diagonal_rook_and_piece_capture", VariantCategory.CAPTURE,
        "Composition: Rook moves diagonally AND captured pieces convert color",
    ),
    "mobile_king_and_dark_immunity": VariantRule(
        "mobile_king_and_dark_immunity", VariantCategory.BOARD,
        "Composition: King moves like queen AND dark squares are immune",
    ),
    "short_knight_and_antichess": VariantRule(
        "short_knight_and_antichess", VariantCategory.OBJECTIVE,
        "Composition: Short knight (1,1) moves AND antichess rules",
    ),
}

COMPOSITIONAL_BOARDS = {
    "diagonal_rook_and_piece_capture": DiagonalRookPieceCaptureBoard,
    "mobile_king_and_dark_immunity": MobileKingDarkImmunityBoard,
    "short_knight_and_antichess": ShortKnightAntichessBoard,
}


def get_compositional_board(name: str, fen: str = None, seed: int = 0):
    rule = COMPOSITIONAL_RULES[name]
    board_cls = COMPOSITIONAL_BOARDS[name]
    return board_cls(rule, fen=fen, seed=seed)

