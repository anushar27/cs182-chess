"""Custom chess variant engine built on python-chess."""

import chess
import copy
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class VariantCategory(Enum):
    MOVE = auto()
    CAPTURE = auto()
    BOARD = auto()
    PIECES = auto()
    OBJECTIVE = auto()


@dataclass(frozen=True)
class VariantRule:
    name: str
    category: VariantCategory
    description: str  # human-readable documentation (NOT given to model)


class VariantBoard:
    """Chess board with custom variant rules.

    Wraps python-chess and overrides legal_moves, is_capture,
    is_game_over, and result based on the active VariantRule.
    Subclasses override specific methods for each variant.
    """

    def __init__(self, variant: VariantRule, fen: Optional[str] = None, seed: int = 0, **kwargs):
        self.variant = variant
        self.board = chess.Board(fen) if fen else chess.Board()
        self.seed = seed
        self._move_count = 0
        self._max_moves = 300  # Safety limit to prevent infinite games
        self._setup()

    def _setup(self):
        """Apply starting-position and board modifications. Override in subclasses."""
        pass

    def legal_moves(self) -> list[chess.Move]:
        """Return legal moves under the variant's rules."""
        return list(self.board.legal_moves)

    def push(self, move: chess.Move):
        """Make a move, enforcing variant legality."""
        legal = self.legal_moves()
        if move not in legal:
            raise ValueError(f"Illegal move: {move} in variant {self.variant.name}")
        self._do_push(move)
        self._move_count += 1

    def _do_push(self, move: chess.Move):
        """Actually execute the move. Override for custom capture/conversion logic."""
        self.board.push(move)

    def is_game_over(self) -> bool:
        """Check if the game is over under variant rules."""
        if self._move_count >= self._max_moves:
            return True
        return self.board.is_game_over(claim_draw=True)

    def result_for_white(self) -> str:
        """Return 'win', 'loss', or 'draw' from White's perspective."""
        if self._move_count >= self._max_moves:
            return "draw"
        r = self.board.result(claim_draw=True)
        if r == "1-0":
            return "win"
        if r == "0-1":
            return "loss"
        return "draw"

    def fen(self) -> str:
        return self.board.fen()

    @property
    def turn(self) -> chess.Color:
        return self.board.turn

    @property
    def move_count(self) -> int:
        return self._move_count

    def copy(self) -> "VariantBoard":
        """Deep copy for search/lookahead."""
        return copy.deepcopy(self)

    def _is_diagonal(self, from_sq: int, to_sq: int) -> bool:
        """Check if move is along a diagonal."""
        from_rank, from_file = chess.square_rank(from_sq), chess.square_file(from_sq)
        to_rank, to_file = chess.square_rank(to_sq), chess.square_file(to_sq)
        dr = abs(to_rank - from_rank)
        df = abs(to_file - from_file)
        return dr == df and dr > 0

    def _is_straight(self, from_sq: int, to_sq: int) -> bool:
        """Check if move is along a rank or file."""
        from_rank, from_file = chess.square_rank(from_sq), chess.square_file(from_sq)
        to_rank, to_file = chess.square_rank(to_sq), chess.square_file(to_sq)
        return (from_rank == to_rank or from_file == to_file) and from_sq != to_sq

    def _sliding_moves(self, square: int, directions: list[tuple[int, int]],
                       color: chess.Color) -> list[chess.Move]:
        """Generate sliding moves in given directions from a square."""
        moves = []
        for dr, df in directions:
            r = chess.square_rank(square) + dr
            f = chess.square_file(square) + df
            while 0 <= r <= 7 and 0 <= f <= 7:
                target = chess.square(f, r)
                target_piece = self.board.piece_at(target)
                if target_piece is None:
                    moves.append(chess.Move(square, target))
                elif target_piece.color != color:
                    moves.append(chess.Move(square, target))
                    break
                else:
                    break
                r += dr
                f += df
        return moves

    def _leaves_king_in_check(self, move: chess.Move) -> bool:
        """Check if making a move would leave our king in check."""
        board_copy = self.board.copy()
        board_copy.push(move)
        return board_copy.is_check()

    def _filter_legal(self, moves: list[chess.Move]) -> list[chess.Move]:
        """Filter moves that would leave king in check using python-chess internals."""
        legal = []
        for move in moves:
            board_copy = self.board.copy()
            # Try the move - if it results in king being capturable, it's illegal
            board_copy.push(move)
            # After pushing, it's the opponent's turn. Check if the opponent can
            # capture our king (meaning we left ourselves in check)
            if not board_copy.was_into_check():
                legal.append(move)
        return legal

    def __repr__(self) -> str:
        return f"VariantBoard({self.variant.name}, move={self._move_count})"
