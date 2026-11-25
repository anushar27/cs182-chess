from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import cached_property
import chess
from .common import digest
from .rules import RULES, make_board


@dataclass(frozen=True)
class State:
    variant: str
    rules_version: str
    initial_fen: str
    moves: tuple[str, ...] = ()
    extras: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**{**data, "moves": tuple(data["moves"])})

    def board(self):
        if self.rules_version != RULES.get(self.variant, {}).get("version"):
            raise ValueError("Rules version mismatch")
        b = make_board(self.variant, self.initial_fen, extras=self.extras)
        for uci in self.moves:
            move = chess.Move.from_uci(uci)
            if move not in b.legal_moves:
                raise ValueError(f"Illegal replay move {uci}")
            b.push(move)
        return b

    @property
    def key(self):
        # Includes full history, counters, rule version and any variant metadata.
        return digest(self.to_dict())

    @cached_property
    def geometry(self):
        b = self.board()
        return b.turn, tuple(b.piece_at(s).symbol() if b.piece_at(s) else "." for s in chess.SQUARES)

    def public(self):
        b = self.board()
        # No variant, rules, seeds, IDs, label, or future continuation.
        return {"initial_fen": self.initial_fen, "history_uci": list(self.moves),
                "current_fen": b.fen(en_passant="fen"), "extras": self.extras}

    def mirror(self):
        if self.variant == "dark_square_immunity":
            raise ValueError("Vertical color swap exchanges square colors; not a symmetry of dark-square immunity")
        b = make_board(self.variant, self.initial_fen, extras=self.extras).mirror()
        moves = tuple(chess.Move(chess.square_mirror(m.from_square),
                                 chess.square_mirror(m.to_square),
                                 promotion=m.promotion).uci()
                      for m in map(chess.Move.from_uci, self.moves))
        return State(self.variant, self.rules_version, b.fen(en_passant="fen", promoted=True), moves,
                     {"barriers": sorted(b.barriers)} if self.variant == "blocked_squares" else {})


def capture(variant, board):
    state = State(variant, RULES[variant]["version"], board.root().fen(en_passant="fen", promoted=True),
                  tuple(m.uci() for m in board.move_stack),
                  {"barriers": sorted(board.barriers)} if variant == "blocked_squares" else {})
    state.__dict__["geometry"] = (board.turn, tuple(board.piece_at(s).symbol() if board.piece_at(s) else "." for s in chess.SQUARES))
    return state


def position_signature(state):
    # Conservative cross-pool duplicate check, independently of variant identity.
    b = state.board()
    return " ".join(b.fen(en_passant="fen").split()[:4])


def board_distance(a, b):
    return sum(x != y for x, y in zip(a.geometry[1], b.geometry[1]))


def is_near(a, b, threshold):
    if a.geometry[0] != b.geometry[0]:
        return False
    differences = 0
    for x, y in zip(a.geometry[1], b.geometry[1]):
        differences += x != y
        if differences > threshold:
            return False
    return True
