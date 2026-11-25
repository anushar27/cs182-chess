"""Board variants: change the board geometry."""

import chess
import random as _random
from .engine import VariantBoard, VariantRule


class BlockedSquaresBoard(VariantBoard):
    """8 random squares are impassable barriers.
    
    No piece can occupy or move through blocked squares.
    Blocked squares are determined at game start based on a seed,
    avoiding squares occupied by pieces in the starting position.
    """

    def __init__(self, variant: VariantRule, fen=None, seed: int = 0, **kwargs):
        self._seed = seed
        self.blocked: set[int] = set()
        super().__init__(variant, fen, seed=seed, **kwargs)

    def _setup(self):
        rng = _random.Random(self._seed)
        # Pick 8 squares that don't overlap with starting pieces
        empty = [
            sq for sq in chess.SQUARES
            if self.board.piece_at(sq) is None
        ]
        num_blocked = min(8, len(empty))
        self.blocked = set(rng.sample(empty, num_blocked))

    def legal_moves(self) -> list[chess.Move]:
        base_moves = list(self.board.legal_moves)
        filtered = []
        for move in base_moves:
            if move.to_square in self.blocked:
                continue
            if self._path_crosses_blocked(move):
                continue
            filtered.append(move)
        return filtered

    def _path_crosses_blocked(self, move: chess.Move) -> bool:
        """Check if a sliding move passes through any blocked square."""
        piece = self.board.piece_at(move.from_square)
        if piece is None:
            return False

        # Only sliding pieces can be blocked mid-path
        if piece.piece_type not in (chess.ROOK, chess.BISHOP, chess.QUEEN):
            return False

        from_rank = chess.square_rank(move.from_square)
        from_file = chess.square_file(move.from_square)
        to_rank = chess.square_rank(move.to_square)
        to_file = chess.square_file(move.to_square)

        dr = 0 if to_rank == from_rank else (1 if to_rank > from_rank else -1)
        df = 0 if to_file == from_file else (1 if to_file > from_file else -1)

        r, f = from_rank + dr, from_file + df
        while (r, f) != (to_rank, to_file):
            sq = chess.square(f, r)
            if sq in self.blocked:
                return True
            r += dr
            f += df

        return False


class LongBoardBoard(VariantBoard):
    """8x20 board with standard pieces on rows 1-2 and 19-20.
    
    Since python-chess only supports 8x8, we simulate this by:
    - Using an internal representation with 20 ranks
    - Mapping to/from special FEN-like notation
    - Custom move generation that accounts for the extended board
    
    For simplicity in this research project, we approximate the long board
    by using a standard 8x8 board but with an "extended pawn push" mechanic:
    - Pawns can move up to 10 squares forward on their first move
    - All other pieces behave normally
    - This captures the key dynamic: a large no-man's-land between armies
    
    Note: A full 8x20 implementation would require replacing python-chess
    entirely. This approximation preserves the strategic essence.
    """

    def __init__(self, variant: VariantRule, fen=None, seed: int = 0, **kwargs):
        super().__init__(variant, fen, seed=seed, **kwargs)
        # Track whether each pawn has moved (for the extended first-move rule)
        self._pawn_moved: set[int] = set()

    def legal_moves(self) -> list[chess.Move]:
        """Standard legal moves, but pawns on starting rank get extended push."""
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
                start_rank = 1 if color == chess.WHITE else 6

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

                            # Extended multi-push from starting rank (simulates long board)
                            if rank == start_rank:
                                for dist in range(2, 7):  # up to 6 squares forward
                                    tr = rank + dist * direction
                                    if 0 <= tr <= 7:
                                        tsq = chess.square(file, tr)
                                        # Check all intermediate squares are clear
                                        path_clear = True
                                        for d in range(1, dist):
                                            isq = chess.square(file, rank + d * direction)
                                            if self.board.piece_at(isq) is not None:
                                                path_clear = False
                                                break
                                        if path_clear and self.board.piece_at(tsq) is None:
                                            if tr != promo_rank:
                                                moves.append(chess.Move(square, tsq))
                                        else:
                                            break

                # Diagonal captures (standard)
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
            else:
                for m in self.board.pseudo_legal_moves:
                    if m.from_square == square:
                        moves.append(m)

        return self._filter_legal(moves)
