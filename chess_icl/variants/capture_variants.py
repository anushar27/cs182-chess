"""Capture variants: change how captures work."""

import chess
from .engine import VariantBoard, VariantRule


class PieceCaptureBoard(VariantBoard):
    """Captured piece converts to your color on the capture square."""

    def _do_push(self, move: chess.Move):
        captured = self.board.piece_at(move.to_square)
        moving_piece = self.board.piece_at(move.from_square)

        if captured is not None and captured.piece_type != chess.KING:
            capturer_color = moving_piece.color
            converted = chess.Piece(captured.piece_type, capturer_color)

            self.board.remove_piece_at(move.from_square)
            self.board.set_piece_at(move.to_square, converted)

            self.board.turn = not self.board.turn
            self.board.ep_square = None
            self.board.halfmove_clock += 1
            if self.board.turn == chess.WHITE:
                self.board.fullmove_number += 1
        else:
            self.board.push(move)

    def is_game_over(self) -> bool:
        if self._move_count >= self._max_moves:
            return True
        if not self.legal_moves():
            return True
        white_king = any(
            self.board.piece_at(sq)
            for sq in chess.SQUARES
            if self.board.piece_at(sq)
            and self.board.piece_at(sq).piece_type == chess.KING
            and self.board.piece_at(sq).color == chess.WHITE
        )
        black_king = any(
            self.board.piece_at(sq)
            for sq in chess.SQUARES
            if self.board.piece_at(sq)
            and self.board.piece_at(sq).piece_type == chess.KING
            and self.board.piece_at(sq).color == chess.BLACK
        )
        if not white_king or not black_king:
            return True
        return False

    def result_for_white(self) -> str:
        if self._move_count >= self._max_moves:
            return "draw"
        white_king = any(
            self.board.piece_at(sq)
            for sq in chess.SQUARES
            if self.board.piece_at(sq)
            and self.board.piece_at(sq).piece_type == chess.KING
            and self.board.piece_at(sq).color == chess.WHITE
        )
        black_king = any(
            self.board.piece_at(sq)
            for sq in chess.SQUARES
            if self.board.piece_at(sq)
            and self.board.piece_at(sq).piece_type == chess.KING
            and self.board.piece_at(sq).color == chess.BLACK
        )
        if not black_king:
            return "win"
        if not white_king:
            return "loss"
        return "draw"


class ForwardPawnCaptureBoard(VariantBoard):
    """Pawns can capture the square directly in front (plus normal diagonals)."""

    def legal_moves(self) -> list[chess.Move]:
        color = self.board.turn
        moves = [
            m for m in self.board.pseudo_legal_moves
            if (p := self.board.piece_at(m.from_square)) is not None and p.piece_type != chess.PAWN
        ]

        direction = 1 if color == chess.WHITE else -1
        promo_rank = 7 if color == chess.WHITE else 0
        start_rank = 1 if color == chess.WHITE else 6

        for square in self.board.pieces(chess.PAWN, color):
            rank = chess.square_rank(square)
            file = chess.square_file(square)

            # Standard single push
            target_rank = rank + direction
            if 0 <= target_rank <= 7:
                target = chess.square(file, target_rank)
                target_piece = self.board.piece_at(target)

                if target_piece is None:
                    if target_rank == promo_rank:
                        for pt in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                            moves.append(chess.Move(square, target, promotion=pt))
                    else:
                        moves.append(chess.Move(square, target))
                        if rank == start_rank:
                            target2 = chess.square(file, rank + 2 * direction)
                            if self.board.piece_at(target2) is None:
                                moves.append(chess.Move(square, target2))
                elif target_piece.color != color:
                    # Forward capture allowed
                    if target_rank == promo_rank:
                        for pt in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                            moves.append(chess.Move(square, target, promotion=pt))
                    else:
                        moves.append(chess.Move(square, target))

            # Diagonal captures
            for df in [-1, 1]:
                cf = file + df
                cr = rank + direction
                if 0 <= cf <= 7 and 0 <= cr <= 7:
                    cap_sq = chess.square(cf, cr)
                    cap_piece = self.board.piece_at(cap_sq)
                    if cap_piece is not None and cap_piece.color != color:
                        if cr == promo_rank:
                            for pt in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                                moves.append(chess.Move(square, cap_sq, promotion=pt))
                        else:
                            moves.append(chess.Move(square, cap_sq))

            # En passant
            ep_sq = self.board.ep_square
            if ep_sq is not None:
                ep_rank = chess.square_rank(ep_sq)
                ep_file = chess.square_file(ep_sq)
                if ep_rank == rank + direction and abs(ep_file - file) == 1:
                    moves.append(chess.Move(square, ep_sq))

        return self._filter_legal(moves)


class DarkSquareImmunityBoard(VariantBoard):
    """Pieces on dark squares cannot be captured."""

    def legal_moves(self) -> list[chess.Move]:
        base_moves = list(self.board.legal_moves)
        filtered = []
        for move in base_moves:
            target = move.to_square
            target_piece = self.board.piece_at(target)
            if target_piece is not None and target_piece.color != self.board.turn:
                is_dark = ((chess.square_rank(target) + chess.square_file(target)) % 2 == 0)
                if is_dark:
                    continue
            filtered.append(move)
        return filtered
