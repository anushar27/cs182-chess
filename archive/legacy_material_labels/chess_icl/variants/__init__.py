"""Custom chess variant engine for ICL experiments.

Provides a modular variant engine built on python-chess that overrides
move generation, capture rules, board geometry, and win conditions.
"""

from .engine import VariantBoard, VariantRule, VariantCategory
from .registry import VARIANTS, get_board, TRAIN_VARIANTS, TEST_VARIANTS

__all__ = [
    "VariantBoard",
    "VariantRule",
    "VariantCategory",
    "VARIANTS",
    "get_board",
    "TRAIN_VARIANTS",
    "TEST_VARIANTS",
]
