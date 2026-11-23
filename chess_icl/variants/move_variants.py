"""Move variants: change how pieces move."""

import chess
from .engine import VariantBoard, VariantRule


class DiagonalRookBoard(VariantBoard):
    """Rook moves diagonally (like a bishop). All other pieces normal."""

    def legal_moves(self) -> list[chess.Move]:
        moves = []
        color = self.board.turn

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is None or piece.color != color:
                continue

            if piece.piece_type == chess.ROOK:
                # Rooks move diagonally instead of straight
                diag_directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
                moves.extend(self._sliding_moves(square, diag_directions, color))
            elif piece.piece_type == chess.QUEEN:
                # Queen still uses both diagonal + straight
                # But since rooks changed, queen stays normal (uses standard)
                for m in self.board.pseudo_legal_moves:
                    if m.from_square == square:
                        moves.append(m)
            else:
                # All other pieces use standard moves
                for m in self.board.pseudo_legal_moves:
                    if m.from_square == square:
                        moves.append(m)

        return self._filter_legal(moves)


class StraightBishopBoard(VariantBoard):
    """Bishop moves in straight lines (like a rook). All other pieces normal."""

    def legal_moves(self) -> list[chess.Move]:
        moves = []
        color = self.board.turn

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is None or piece.color != color:
                continue

            if piece.piece_type == chess.BISHOP:
                # Bishops move in straight lines instead of diagonals
                straight_directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                moves.extend(self._sliding_moves(square, straight_directions, color))
            else:
                for m in self.board.pseudo_legal_moves:
                    if m.from_square == square:
                        moves.append(m)

        return self._filter_legal(moves)


class WeakenedQueenBoard(VariantBoard):
    """Queen can only move like a rook (loses diagonal movement)."""

    def legal_moves(self) -> list[chess.Move]:
        moves = []
        color = self.board.turn

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is None or piece.color != color:
                continue

            if piece.piece_type == chess.QUEEN:
                # Queen only moves in straight lines (like a rook)
                straight_directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                moves.extend(self._sliding_moves(square, straight_directions, color))
            else:
                for m in self.board.pseudo_legal_moves:
                    if m.from_square == square:
                        moves.append(m)

        return self._filter_legal(moves)


class ExtendedPawnBoard(VariantBoard):
    """Pawns can always move 1 or 2 squares forward (not just from starting rank)."""

    def legal_moves(self) -> list[chess.Move]:
        moves = []
        color = self.board.turn

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is None or piece.color != color:
                continue

            if piece.piece_type == chess.PAWN:
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                direction = 1 if color == chess.WHITE else -1
                promo_rank = 7 if color == chess.WHITE else 0

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

                            # Double push (always allowed if single push is clear)
                            target_rank_2 = rank + 2 * direction
                            if 0 <= target_rank_2 <= 7:
                                target_2 = chess.square(file, target_rank_2)
                                if self.board.piece_at(target_2) is None:
                                    if target_rank_2 == promo_rank:
                                        for pt in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                                            moves.append(chess.Move(square, target_2, promotion=pt))
                                    else:
                                        moves.append(chess.Move(square, target_2))

                # Diagonal captures (standard)
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
            else:
                for m in self.board.pseudo_legal_moves:
                    if m.from_square == square:
                        moves.append(m)

        return self._filter_legal(moves)


class ShortKnightBoard(VariantBoard):
    """Knight moves in a shorter L-shape: (1,2) and (2,1) replaced with (1,1) diagonal jumps.
    
    The 'short knight' jumps to any square exactly 2 squares away by king distance,
    i.e., the 8 squares a king would reach, minus adjacent squares. This gives
    the knight 4 possible moves: the diagonal squares at distance sqrt(2).
    Actually: a 2x2 L means the knight jumps (2,0), (0,2), (-2,0), (0,-2) OR
    (1,1), (1,-1), (-1,1), (-1,-1). We use diagonal + orthogonal-2 jumps.
    
    Simplified: short knight can jump to any of the 4 diagonal-adjacent squares
    (like a ferz in fairy chess).
    """

    def legal_moves(self) -> list[chess.Move]:
        moves = []
        color = self.board.turn

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is None or piece.color != color:
                continue

            if piece.piece_type == chess.KNIGHT:
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                # Short knight: jumps to diagonal-adjacent squares (ferz movement)
                # Plus (2,0) and (0,2) orthogonal jumps (dabbaba)
                offsets = [
                    (1, 1), (1, -1), (-1, 1), (-1, -1),  # ferz (diagonal-1)
                    (2, 0), (-2, 0), (0, 2), (0, -2),     # dabbaba (orthogonal-2)
                ]
                for dr, df in offsets:
                    tr, tf = rank + dr, file + df
                    if 0 <= tr <= 7 and 0 <= tf <= 7:
                        target = chess.square(tf, tr)
                        target_piece = self.board.piece_at(target)
                        if target_piece is None or target_piece.color != color:
                            moves.append(chess.Move(square, target))
            else:
                for m in self.board.pseudo_legal_moves:
                    if m.from_square == square:
                        moves.append(m)

        return self._filter_legal(moves)


class MobileKingBoard(VariantBoard):
    """King can move any number of squares in any direction (like a queen).
    
    The king still cannot move into check. Castling is removed since the
    king's movement is fundamentally changed.
    """

    def legal_moves(self) -> list[chess.Move]:
        moves = []
        color = self.board.turn

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is None or piece.color != color:
                continue

            if piece.piece_type == chess.KING:
                # King moves like a queen (all 8 directions, any distance)
                all_directions = [
                    (1, 0), (-1, 0), (0, 1), (0, -1),
                    (1, 1), (1, -1), (-1, 1), (-1, -1),
                ]
                moves.extend(self._sliding_moves(square, all_directions, color))
            else:
                for m in self.board.pseudo_legal_moves:
                    if m.from_square == square:
                        moves.append(m)

        return self._filter_legal(moves)
