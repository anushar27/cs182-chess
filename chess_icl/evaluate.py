"""Compare frozen-model predictions on a fixed set of paired positions."""

from collections.abc import Sequence

from .data import Game, PositionPair
from .model import Evaluator


def evaluate(
    model: Evaluator,
    pairs: Sequence[PositionPair],
    variant_games: Sequence[Game],
    *,
    standard_rules: str,
    variant_rules: str,
    k_values: Sequence[int],
    seed: int,
) -> None:
    """Compare standard, rules-only variant, and K-game variant predictions."""
    raise NotImplementedError("TODO: Sample demonstrations and score each evaluation condition.")
