"""Offline game generation and reference scoring interfaces."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .data import Game, Position, PositionPair


@dataclass(frozen=True)
class VariantConfig:
    name: str
    config_path: Path | None = None


def generate_games(
    variant: VariantConfig,
    *,
    game_count: int,
    seed: int,
) -> list[Game]:
    """Generate complete games using pyffish to apply the variant rules."""
    # TODO: Load the INI text with pyffish.load_variant_config for custom rules.
    # TODO: Choose a move policy, check terminal states, and record results.
    raise NotImplementedError("TODO: Implement game generation.")


def score_position(
    position: Position,
    variant: VariantConfig,
    *,
    engine_path: Path,
    depth: int,
) -> float:
    """Get a side-to-move reference score in centipawns from Fairy-Stockfish."""
    # TODO: Use UCI VariantPath for custom rules and select UCI_Variant by name.
    # TODO: Pass the required position state, search, and normalize mate scores.
    raise NotImplementedError("TODO: Implement Fairy-Stockfish reference scoring.")


def build_position_pairs(
    positions: Sequence[tuple[Position, Position]],
    variant: VariantConfig,
    *,
    engine_path: Path,
    depth: int,
) -> list[PositionPair]:
    """Score matching standard and variant positions in that order."""
    raise NotImplementedError("TODO: Validate each pair and score both rulesets.")
