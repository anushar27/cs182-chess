"""Concrete implementations of Standard Chess, all 13 single variants, and 3 compositional variants."""

from __future__ import annotations

from typing import Any, ClassVar, Iterable, Optional

import chess
import chess.variant

from .board import _USE_STARTING_FEN, VariantBoard, register_variant_class
from .specs import (
    ALL_VARIANTS,
    COMPOSITIONAL_VARIANT_SPECS,
    SINGLE_VARIANT_SPECS,
    STANDARD_SPEC,
    VariantSpec,
)


class StandardBoard(VariantBoard):
    """Standard FIDE chess."""

    default_spec: ClassVar[VariantSpec] = STANDARD_SPEC


# ── Train Variants (4) ───────────────────────────────────────────────────────


class DiagonalRookBoard(VariantBoard):
    """Rook moves and attacks diagonally (like a bishop)."""

    default_spec: ClassVar[VariantSpec] = SINGLE_VARIANT_SPECS["diagonal_rook"]
    _rook_mode: ClassVar[str] = "diagonal"


class StraightBishopBoard(VariantBoard):
    """Bishop moves and attacks orthogonally (like a rook)."""

    default_spec: ClassVar[VariantSpec] = SINGLE_VARIANT_SPECS["straight_bishop"]
    _bishop_mode: ClassVar[str] = "straight"


class WeakenedQueenBoard(VariantBoard):
    """Queen moves and attacks orthogonally only (like a rook)."""

    default_spec: ClassVar[VariantSpec] = SINGLE_VARIANT_SPECS["weakened_queen"]
    _queen_mode: ClassVar[str] = "straight"


class ExtendedPawnBoard(VariantBoard):
    """Pawns can always advance 1 or 2 unobstructed squares forward from any rank."""

    default_spec: ClassVar[VariantSpec] = SINGLE_VARIANT_SPECS["extended_pawn"]
    _pawn_extended_push: ClassVar[bool] = True


# ── Test Variants (9) ────────────────────────────────────────────────────────


class ShortKnightBoard(VariantBoard):
    """Knight leaps via Ferz + Dabbaba offsets: (±1,±1), (±2,0), (0,±2)."""

    default_spec: ClassVar[VariantSpec] = SINGLE_VARIANT_SPECS["short_knight"]
    _knight_mode: ClassVar[str] = "short"


class MobileKingBoard(VariantBoard):
    """King moves and attacks like a queen (any distance in all 8 directions)."""

    default_spec: ClassVar[VariantSpec] = SINGLE_VARIANT_SPECS["mobile_king"]
    _king_mode: ClassVar[str] = "queen"


class PieceCaptureBoard(VariantBoard):
    """Captured non-king pieces convert to the capturer's color on the capture square."""

    default_spec: ClassVar[VariantSpec] = SINGLE_VARIANT_SPECS["piece_capture"]
    _convert_captures: ClassVar[bool] = True
    _require_both_kings: ClassVar[bool] = True


class ForwardPawnCaptureBoard(VariantBoard):
    """Pawns can capture directly 1 square forward in addition to diagonal captures."""

    default_spec: ClassVar[VariantSpec] = SINGLE_VARIANT_SPECS["forward_pawn_capture"]
    _pawn_forward_capture: ClassVar[bool] = True


class DarkSquareImmunityBoard(VariantBoard):
    """Non-king pieces on dark squares ((rank+file)%2==0) cannot be captured."""

    default_spec: ClassVar[VariantSpec] = SINGLE_VARIANT_SPECS["dark_square_immunity"]
    _dark_square_immunity: ClassVar[bool] = True


class BlockedSquaresBoard(VariantBoard):
    """8 deterministic barrier squares in ranks 3–6 (a3..h6) block moves and rays."""

    default_spec: ClassVar[VariantSpec] = SINGLE_VARIANT_SPECS["blocked_squares"]
    _uses_blocked_squares: ClassVar[bool] = True


class LongBoardBoard(VariantBoard):
    """Frozen 8x8 Extended Opening Pawn Push approximation (1..6 squares from starting rank)."""

    default_spec: ClassVar[VariantSpec] = SINGLE_VARIANT_SPECS["long_board"]
    _pawn_long_opening_push: ClassVar[bool] = True


class KnightArmyBoard(VariantBoard):
    """Setup-only null control: 15 knights + 1 king per side when initialized without custom FEN."""

    default_spec: ClassVar[VariantSpec] = SINGLE_VARIANT_SPECS["knight_army"]
    starting_fen: ClassVar[str] = "nnnnknnn/nnnnnnnn/8/8/8/8/NNNNNNNN/NNNNKNNN w - - 0 1"

    def reset(self) -> None:
        self.set_fen(type(self).starting_fen)


class AntichessBoard(chess.variant.AntichessBoard, VariantBoard):
    """Antichess delegating directly to python-chess chess.variant.AntichessBoard."""

    default_spec: ClassVar[VariantSpec] = SINGLE_VARIANT_SPECS["antichess"]
    _no_check_rules: ClassVar[bool] = True

    def __init__(
        self,
        fen: Optional[str] = _USE_STARTING_FEN,  # type: ignore[assignment]
        *,
        chess960: bool = False,
        variant: Optional[VariantSpec] = None,
        seed: int = 0,
        blocked: Optional[Iterable[int]] = None,
    ) -> None:
        actual_fen = type(self).starting_fen if fen is _USE_STARTING_FEN else fen
        chess.variant.AntichessBoard.__init__(self, actual_fen, chess960=chess960)
        self.variant = variant if variant is not None else type(self).default_spec
        self.seed = int(seed)
        self.blocked = set()
        self.blocked_mask = 0
        if blocked is not None:
            self._set_blocked_squares(blocked)

    def _has_custom_rules(self) -> bool:
        # Standard antichess delegates pseudo-legal/attack generation straight to chess.Board
        # and antichess rules straight to chess.variant.AntichessBoard.
        return self.blocked_mask != 0


# ── Compositional Variants (3) ───────────────────────────────────────────────


class DiagonalRookPieceCaptureBoard(VariantBoard):
    """Composition: Diagonal Rook + Piece Capture."""

    default_spec: ClassVar[VariantSpec] = COMPOSITIONAL_VARIANT_SPECS["diagonal_rook_and_piece_capture"]
    _rook_mode: ClassVar[str] = "diagonal"
    _convert_captures: ClassVar[bool] = True
    _require_both_kings: ClassVar[bool] = True


class MobileKingDarkImmunityBoard(VariantBoard):
    """Composition: Mobile King + Dark Square Immunity."""

    default_spec: ClassVar[VariantSpec] = COMPOSITIONAL_VARIANT_SPECS["mobile_king_and_dark_immunity"]
    _king_mode: ClassVar[str] = "queen"
    _dark_square_immunity: ClassVar[bool] = True


class ShortKnightAntichessBoard(AntichessBoard):
    """Composition: Short Knight (Ferz+Dabbaba) + Antichess."""

    default_spec: ClassVar[VariantSpec] = COMPOSITIONAL_VARIANT_SPECS["short_knight_and_antichess"]
    _knight_mode: ClassVar[str] = "short"
    _no_check_rules: ClassVar[bool] = True

    def _has_custom_rules(self) -> bool:
        return True

    def has_insufficient_material(self, color: chess.Color) -> bool:
        if not self.occupied_co[color]:
            return False
        if not self.occupied_co[not color]:
            return True
        if self.occupied == self.knights or self.occupied == self.bishops:
            # Both bishops and short knights are strictly color-bound.
            we_light = bool(self.occupied_co[color] & chess.BB_LIGHT_SQUARES)
            we_dark = bool(self.occupied_co[color] & chess.BB_DARK_SQUARES)
            they_all_dark = not (self.occupied_co[not color] & chess.BB_LIGHT_SQUARES)
            they_all_light = not (self.occupied_co[not color] & chess.BB_DARK_SQUARES)
            return (we_light and they_all_dark) or (we_dark and they_all_light)
        return chess.variant.AntichessBoard.has_insufficient_material(self, color)


# ── Registry & Factory ───────────────────────────────────────────────────────

VARIANT_CLASSES: dict[str, type[VariantBoard]] = {
    "standard": StandardBoard,
    "diagonal_rook": DiagonalRookBoard,
    "straight_bishop": StraightBishopBoard,
    "weakened_queen": WeakenedQueenBoard,
    "extended_pawn": ExtendedPawnBoard,
    "short_knight": ShortKnightBoard,
    "mobile_king": MobileKingBoard,
    "piece_capture": PieceCaptureBoard,
    "forward_pawn_capture": ForwardPawnCaptureBoard,
    "dark_square_immunity": DarkSquareImmunityBoard,
    "blocked_squares": BlockedSquaresBoard,
    "long_board": LongBoardBoard,
    "knight_army": KnightArmyBoard,
    "antichess": AntichessBoard,
    "diagonal_rook_and_piece_capture": DiagonalRookPieceCaptureBoard,
    "mobile_king_and_dark_immunity": MobileKingDarkImmunityBoard,
    "short_knight_and_antichess": ShortKnightAntichessBoard,
}

for _name, _cls in VARIANT_CLASSES.items():
    register_variant_class(_name, _cls)


def get_board(
    variant_name: str,
    fen: Optional[str] = None,
    seed: int = 0,
    **kwargs: Any,
) -> VariantBoard:
    """Create a VariantBoard instance for `variant_name`.

    Args:
        variant_name: Name in ALL_VARIANTS (standard, 13 single, 3 compositional).
        fen: Optional FEN string. When None, initializes the variant's starting
            position (e.g., 15 knights + 1 king for `knight_army`).
        seed: Deterministic seed (used by `blocked_squares`).
        **kwargs: Additional board keyword arguments (e.g., `blocked`, `chess960`).
    """
    if variant_name not in ALL_VARIANTS:
        raise ValueError(
            f"Unknown variant: {variant_name!r}. Available: {list(ALL_VARIANTS.keys())}"
        )
    spec = ALL_VARIANTS[variant_name]
    board_cls = VARIANT_CLASSES[variant_name]
    if fen is None:
        return board_cls(variant=spec, seed=seed, **kwargs)
    return board_cls(fen, variant=spec, seed=seed, **kwargs)
