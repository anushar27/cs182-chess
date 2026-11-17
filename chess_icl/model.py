"""Standard-chess pretraining and in-context score prediction."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from .data import Game, Position, ScoredPosition


class Evaluator(Protocol):
    def predict(
        self,
        position: Position,
        *,
        rules: str,
        demonstrations: Sequence[Game] = (),
    ) -> float:
        """Predict a centipawn score from the side-to-move perspective."""



def pretrain(positions: Sequence[ScoredPosition], checkpoint: Path) -> None:
    """Train on standard-chess position scores and save the model weights."""
    raise NotImplementedError("TODO: Encode training inputs and train a regression model.")


def load_evaluator(checkpoint: Path) -> Evaluator:
    """Restore the pretrained model for frozen evaluation."""
    raise NotImplementedError("TODO: Restore weights and encode rules and demonstrations.")
