"""Central registry of all 13 variants."""

from .engine import VariantBoard, VariantRule, VariantCategory
from .move_variants import (
    DiagonalRookBoard,
    StraightBishopBoard,
    WeakenedQueenBoard,
    ExtendedPawnBoard,
    ShortKnightBoard,
    MobileKingBoard,
)
from .capture_variants import (
    PieceCaptureBoard,
    ForwardPawnCaptureBoard,
    DarkSquareImmunityBoard,
)
from .board_variants import (
    BlockedSquaresBoard,
    LongBoardBoard,
)
from .piece_variants import KnightArmyBoard
from .objective_variants import AntichessBoard

from typing import Optional


# ── Variant definitions ──────────────────────────────────────────────────────

VARIANTS = {
    # Move variants
    "diagonal_rook": VariantRule(
        "diagonal_rook", VariantCategory.MOVE,
        "Rook moves diagonally like a bishop",
    ),
    "straight_bishop": VariantRule(
        "straight_bishop", VariantCategory.MOVE,
        "Bishop moves in straight lines like a rook",
    ),
    "weakened_queen": VariantRule(
        "weakened_queen", VariantCategory.MOVE,
        "Queen loses diagonal movement, moves like rook only",
    ),
    "extended_pawn": VariantRule(
        "extended_pawn", VariantCategory.MOVE,
        "Pawns can always move 1 or 2 squares forward",
    ),
    "short_knight": VariantRule(
        "short_knight", VariantCategory.MOVE,
        "Knight moves in short L-shape (ferz + dabbaba)",
    ),
    "mobile_king": VariantRule(
        "mobile_king", VariantCategory.MOVE,
        "King moves like a queen (any distance, any direction)",
    ),
    # Capture variants
    "piece_capture": VariantRule(
        "piece_capture", VariantCategory.CAPTURE,
        "Captured piece converts to capturer's color",
    ),
    "forward_pawn_capture": VariantRule(
        "forward_pawn_capture", VariantCategory.CAPTURE,
        "Pawns can also capture directly forward",
    ),
    "dark_square_immunity": VariantRule(
        "dark_square_immunity", VariantCategory.CAPTURE,
        "Pieces on dark squares cannot be captured",
    ),
    # Board variants
    "blocked_squares": VariantRule(
        "blocked_squares", VariantCategory.BOARD,
        "8 random squares are impassable barriers",
    ),
    "long_board": VariantRule(
        "long_board", VariantCategory.BOARD,
        "8x20 board simulated with extended pawn pushes",
    ),
    # Piece variants
    "knight_army": VariantRule(
        "knight_army", VariantCategory.PIECES,
        "Each side: 15 knights + 1 king",
    ),
    # Objective variants
    "antichess": VariantRule(
        "antichess", VariantCategory.OBJECTIVE,
        "Lose all pieces to win; must capture if possible",
    ),
}

# ── Train/Test split ─────────────────────────────────────────────────────────

TRAIN_VARIANTS = ["diagonal_rook", "straight_bishop", "weakened_queen", "extended_pawn"]
TEST_VARIANTS = [
    "short_knight", "mobile_king",           # In-category (Move)
    "piece_capture", "forward_pawn_capture",  # Capture
    "dark_square_immunity",                   # Capture
    "blocked_squares", "long_board",          # Board
    "knight_army",                            # Pieces
    "antichess",                              # Objective
]

# ── Board class mapping ──────────────────────────────────────────────────────

_BOARD_CLASSES = {
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
}


def get_board(variant_name: str, fen: Optional[str] = None, **kwargs) -> VariantBoard:
    """Factory: create a VariantBoard instance for the named variant.
    
    Args:
        variant_name: Name of the variant (must be in VARIANTS).
        fen: Optional starting FEN. If None, uses standard start
             (or variant-specific start for knight_army, etc.)
        **kwargs: Additional arguments (e.g., seed for blocked_squares).
    
    Returns:
        A VariantBoard subclass instance.
    
    Raises:
        ValueError: If variant_name is not recognized.
    """
    if variant_name not in VARIANTS:
        raise ValueError(
            f"Unknown variant: {variant_name}. "
            f"Available: {list(VARIANTS.keys())}"
        )

    rule = VARIANTS[variant_name]
    board_cls = _BOARD_CLASSES[variant_name]
    return board_cls(rule, fen=fen, **kwargs)


def list_variants(category: Optional[VariantCategory] = None) -> list[str]:
    """List variant names, optionally filtered by category."""
    if category is None:
        return list(VARIANTS.keys())
    return [name for name, rule in VARIANTS.items() if rule.category == category]
