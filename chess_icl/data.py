"""Game records and paired evaluation positions."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Outcome = Literal["win", "draw", "loss"]


@dataclass(frozen=True)
class Game:
    game_id: str
    initial_fen: str
    moves: tuple[str, ...]
    white_outcome: Outcome


@dataclass(frozen=True)
class Position:
    fen: str
    history: tuple[str, ...] = ()
    # TODO: Define the history or extra state required by the chosen rules.


@dataclass(frozen=True)
class PositionPair:
    position: Position
    standard_outcome: Outcome
    variant_outcome: Outcome
    # Both outcomes use the side-to-move perspective.


def load_games(path: Path) -> list[Game]:
    """Load games from a single ruleset and data split."""
    raise NotImplementedError("TODO: Choose a game format and implement loading.")


def build_position_pairs(standard_rules: str, variant_rules: str) -> list[PositionPair]:
    """Label each shared held-out position independently under both rulesets."""
    raise NotImplementedError("TODO: Choose the variant and implement reference labeling.")
