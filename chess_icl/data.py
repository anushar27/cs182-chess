"""Game records and paired centipawn evaluations."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

GameResult = Literal["win", "draw", "loss"]


@dataclass(frozen=True)
class Game:
    game_id: str
    initial_fen: str
    moves: tuple[str, ...]
    white_outcome: GameResult


@dataclass(frozen=True)
class Position:
    fen: str
    initial_fen: str | None = None
    moves: tuple[str, ...] = ()
    # For history-dependent rules, replay moves from initial_fen to reach fen.


@dataclass(frozen=True)
class ScoredPosition:
    position: Position
    score_cp: float


@dataclass(frozen=True)
class PositionPair:
    # Both views share piece placement and side to move.
    standard: ScoredPosition
    variant: ScoredPosition


def load_games(path: Path) -> list[Game]:
    """Load games from a single ruleset and data split."""
    raise NotImplementedError("TODO: Choose a game format and implement loading.")


def load_position_pairs(path: Path) -> list[PositionPair]:
    """Load paired reference scores produced during offline preparation."""
    raise NotImplementedError("TODO: Choose a format and load scored position pairs.")
