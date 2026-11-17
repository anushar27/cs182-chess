"""Standard-chess pretraining and in-context outcome prediction."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from .data import Game, Outcome, Position


class Evaluator(Protocol):
    def predict(
        self,
        position: Position,
        *,
        rules: str,
        demonstrations: Sequence[Game] = (),
    ) -> Outcome:
        """Predict from the side-to-move perspective."""


def pretrain(games: Sequence[Game], checkpoint: Path) -> None:
    """Train a standard-chess evaluator and save its weights."""
    raise NotImplementedError("TODO: Choose the model, encoding, and training objective.")


def load_evaluator(checkpoint: Path) -> Evaluator:
    """Restore the pretrained model for frozen evaluation."""
    raise NotImplementedError("TODO: Restore weights and encode rules and demonstrations.")
