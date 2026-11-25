"""Move variants: change how pieces move."""

import chess
from .engine import VariantBoard, VariantRule


class DiagonalRookBoard(VariantBoard):
    """Rook moves diagonally (like a bishop). All other pieces normal."""

    def legal_moves(self) -> list[chess.Move]:
        color = self.board.turn
        # All pseudo-legal moves for non-rooks
        moves = [
            m for m in self.board.pseudo_legal_moves
            if (p := self.board.piece_at(m.from_square)) is not None and p.piece_type != chess.ROOK
        ]
        # Diagonal moves for rooks
        diag_directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        for square in self.board.pieces(chess.ROOK, color):
            moves.extend(self._sliding_moves(square, diag_directions, color))

        return self._filter_legal(moves)


class StraightBishopBoard(VariantBoard):
    """Bishop moves in straight lines (like a rook). All other pieces normal."""

    def legal_moves(self) -> list[chess.Move]:
        color = self.board.turn
        moves = [
            m for m in self.board.pseudo_legal_moves
            if (p := self.board.piece_at(m.from_square)) is not None and p.piece_type != chess.BISHOP
        ]
        straight_directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for square in self.board.pieces(chess.BISHOP, color):
            moves.extend(self._sliding_moves(square, straight_directions, color))

        return self._filter_legal(moves)


class WeakenedQueenBoard(VariantBoard):
    """Queen can only move like a rook (loses diagonal movement)."""

    def legal_moves(self) -> list[chess.Move]:
        color = self.board.turn
        moves = [
            m for m in self.board.pseudo_legal_moves
            if (p := self.board.piece_at(m.from_square)) is not None and p.piece_type != chess.QUEEN
        ]
        straight_directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for square in self.board.pieces(chess.QUEEN, color):
            moves.extend(self._sliding_moves(square, straight_directions, color))

        return self._filter_legal(moves)


class ExtendedPawnBoard(VariantBoard):
    """Pawns can always move 1 or 2 squares forward (not just from starting rank)."""

    def legal_moves(self) -> list[chess.Move]:
        color = self.board.turn
        # Non-pawn moves
        moves = [
            m for m in self.board.pseudo_legal_moves
            if (p := self.board.piece_at(m.from_square)) is not None and p.piece_type != chess.PAWN
        ]

        direction = 1 if color == chess.WHITE else -1
        promo_rank = 7 if color == chess.WHITE else 0

        for square in self.board.pieces(chess.PAWN, color):
            rank = chess.square_rank(square)
            file = chess.square_file(square)

            # Single push
            target_rank = rank + direction
            if 0 <= target_rank <= 7:
                target = chess.square(file, target_rank)
                if self.board.piece_at(target) is None:
                    if target_rank == promo_rank:
                        for pt in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                            moves.append(chess.Move(square, target, promotion=pt))
                    else:
                        moves.append(chess.Move(square, target))
                        # Double push
                        target_rank_2 = rank + 2 * direction
                        if 0 <= target_rank_2 <= 7:
                            target_2 = chess.square(file, target_rank_2)
                            if self.board.piece_at(target_2) is None:
                                if target_rank_2 == promo_rank:
                                    for pt in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                                        moves.append(chess.Move(square, target_2, promotion=pt))
                                else:
                                    moves.append(chess.Move(square, target_2))

            # Diagonal captures
            for df in [-1, 1]:
                capture_file = file + df
                capture_rank = rank + direction
                if 0 <= capture_file <= 7 and 0 <= capture_rank <= 7:
                    capture_sq = chess.square(capture_file, capture_rank)
                    target_piece = self.board.piece_at(capture_sq)
                    if target_piece is not None and target_piece.color != color:
                        if capture_rank == promo_rank:
                            for pt in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                                moves.append(chess.Move(square, capture_sq, promotion=pt))
                        else:
                            moves.append(chess.Move(square, capture_sq))

            # En passant
            ep_sq = self.board.ep_square
            if ep_sq is not None:
                ep_rank = chess.square_rank(ep_sq)
                ep_file = chess.square_file(ep_sq)
                if ep_rank == rank + direction and abs(ep_file - file) == 1:
                    moves.append(chess.Move(square, ep_sq))

        return self._filter_legal(moves)


class ShortKnightBoard(VariantBoard):
    """Knight moves in a shorter L-shape: (1,1) diagonal jumps + (2,0)/(0,2) orthogonal jumps."""

    def legal_moves(self) -> list[chess.Move]:
        color = self.board.turn
        moves = [
            m for m in self.board.pseudo_legal_moves
            if (p := self.board.piece_at(m.from_square)) is not None and p.piece_type != chess.KNIGHT
        ]

        offsets = [
            (1, 1), (1, -1), (-1, 1), (-1, -1),
            (2, 0), (-2, 0), (0, 2), (0, -2),
        ]
        for square in self.board.pieces(chess.KNIGHT, color):
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            for dr, df in offsets:
                tr, tf = rank + dr, file + df
                if 0 <= tr <= 7 and 0 <= tf <= 7:
                    target = chess.square(tf, tr)
                    target_piece = self.board.piece_at(target)
                    if target_piece is None or target_piece.color != color:
                        moves.append(chess.Move(square, target))

        return self._filter_legal(moves)


class MobileKingBoard(VariantBoard):
    """King can move any number of squares in any direction (queen movement)."""

    def legal_moves(self) -> list[chess.Move]:
        color = self.board.turn
        moves = [
            m for m in self.board.pseudo_legal_moves
            if (p := self.board.piece_at(m.from_square)) is not None and p.piece_type != chess.KING
        ]

        all_directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ]
        for square in self.board.pieces(chess.KING, color):
            moves.extend(self._sliding_moves(square, all_directions, color))

        return self._filter_legal(moves)
