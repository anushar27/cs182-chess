"""Frozen specifications for all chess variants in chess_icl_v2.

Covers:
- Standard Chess (`standard`)
- 4 Train variants (`diagonal_rook`, `straight_bishop`, `weakened_queen`, `extended_pawn`)
- 9 Test variants (`short_knight`, `mobile_king`, `piece_capture`,
  `forward_pawn_capture`, `dark_square_immunity`, `blocked_squares`,
  `long_board`, `knight_army`, `antichess`)
- 3 Compositional variants (`diagonal_rook_and_piece_capture`,
  `mobile_king_and_dark_immunity`, `short_knight_and_antichess`)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class VariantCategory(str, Enum):
    """Rule modification category for a chess variant."""

    STANDARD = "standard"
    MOVE = "move"
    CAPTURE = "capture"
    BOARD = "board"
    PIECES = "pieces"
    OBJECTIVE = "objective"
    COMPOSITIONAL = "compositional"


@dataclass(frozen=True)
class VariantSpec:
    """Immutable formal specification for a chess variant."""

    name: str
    category: VariantCategory
    split: str  # "standard" | "train" | "test" | "compositional"
    description: str
    formal_rule: str
    is_setup_only_control: bool = False
    is_visibly_identifiable: bool = False
    is_compositional: bool = False
    components: tuple[str, ...] = ()
    no_check_rules: bool = False
    uses_blocked_squares: bool = False


STANDARD_SPEC = VariantSpec(
    name="standard",
    category=VariantCategory.STANDARD,
    split="standard",
    description="Standard FIDE chess.",
    formal_rule="Standard movement, captures, castling, en passant, check/checkmate, and draws.",
)

SINGLE_VARIANT_SPECS: dict[str, VariantSpec] = {
    # ── Train (4) ────────────────────────────────────────────────────────────
    "diagonal_rook": VariantSpec(
        name="diagonal_rook",
        category=VariantCategory.MOVE,
        split="train",
        description="Rook moves and attacks diagonally (like a bishop) instead of orthogonally.",
        formal_rule=(
            "Rooks slide and give check/pin along diagonals only; orthogonal rook moves are disabled. "
            "Standard castling with unmoved king and corner rook remains enabled if path is clear and unattacked. "
            "Insufficient material treats diagonal rooks identically to bishops."
        ),
    ),
    "straight_bishop": VariantSpec(
        name="straight_bishop",
        category=VariantCategory.MOVE,
        split="train",
        description="Bishop moves and attacks orthogonally (like a rook) instead of diagonally.",
        formal_rule=(
            "Bishops slide and give check/pin along ranks and files only; diagonal bishop moves are disabled. "
            "Insufficient material treats straight bishops as sufficient mating material (like rooks)."
        ),
    ),
    "weakened_queen": VariantSpec(
        name="weakened_queen",
        category=VariantCategory.MOVE,
        split="train",
        description="Queen loses diagonal movement and moves/attacks orthogonally only (like a rook).",
        formal_rule=(
            "Queens slide and give check/pin along ranks and files only; diagonal queen moves and attacks are disabled."
        ),
    ),
    "extended_pawn": VariantSpec(
        name="extended_pawn",
        category=VariantCategory.MOVE,
        split="train",
        description="Pawns can always push 1 or 2 squares forward from any rank if unobstructed.",
        formal_rule=(
            "Pawns may advance 1 or 2 empty squares forward from any rank (promoting to Q/R/B/N if landing on "
            "the backrank). En passant target square is set only on a 2-square push from the pawn's starting "
            "rank (rank 2 for White, rank 7 for Black). Diagonal captures and check rules are standard."
        ),
    ),
    # ── Test (9) ─────────────────────────────────────────────────────────────
    "short_knight": VariantSpec(
        name="short_knight",
        category=VariantCategory.MOVE,
        split="test",
        description="Knight leaps via Ferz + Dabbaba offsets: (±1,±1), (±2,0), (0,±2).",
        formal_rule=(
            "Knights jump to the 8 squares at offsets (±1,±1), (±2,0), and (0,±2), ignoring intermediate occupancy, "
            "and deliver check/attack along those same 8 offsets. Because all offsets preserve square-color parity, "
            "short knights are color-bound (treated like bishops for insufficient material)."
        ),
    ),
    "mobile_king": VariantSpec(
        name="mobile_king",
        category=VariantCategory.MOVE,
        split="test",
        description="King moves and attacks like a queen (any distance along ranks, files, or diagonals).",
        formal_rule=(
            "Kings slide and give check/pin any unobstructed distance along all 8 orthogonal and diagonal rays. "
            "Castling remains valid when unmoved and unattacked; non-castling king slides to g1/c1/h1/a1 do not "
            "trigger castling once rights are lost or blocked."
        ),
    ),
    "piece_capture": VariantSpec(
        name="piece_capture",
        category=VariantCategory.CAPTURE,
        split="test",
        description="Captured non-king piece converts to the capturer's color on the capture square.",
        formal_rule=(
            "On any capture (including en passant), the captured piece is replaced on the destination square by a "
            "piece of the captured type belonging to the capturer's color; the capturing piece vacates its origin "
            "square. Castling rights, ep_square, halfmove_clock (zeroed on capture), fullmove_number, and "
            "repetition history update identically to chess.Board. Missing-king FENs terminate immediately."
        ),
    ),
    "forward_pawn_capture": VariantSpec(
        name="forward_pawn_capture",
        category=VariantCategory.CAPTURE,
        split="test",
        description="Pawns can capture directly 1 square forward in addition to diagonal captures.",
        formal_rule=(
            "Pawns may capture an enemy piece located 1 square directly ahead (with promotion on the backrank), "
            "and consequently attack and deliver check 1 square directly ahead in addition to standard diagonal "
            "pawn attacks. Standard 1- and 2-square non-capturing pushes and diagonal en passant are unchanged."
        ),
    ),
    "dark_square_immunity": VariantSpec(
        name="dark_square_immunity",
        category=VariantCategory.CAPTURE,
        split="test",
        description="Non-king pieces on dark squares ((rank+file)%2==0) cannot be captured.",
        formal_rule=(
            "Any non-king piece standing on a dark square ((rank + file) % 2 == 0, e.g., a1, c1, b2) is immune "
            "from capture (including en passant when the captured pawn stands on a dark square). Kings remain "
            "vulnerable to check and checkmate on all squares, and pieces on dark squares still deliver check and "
            "control squares normally."
        ),
    ),
    "blocked_squares": VariantSpec(
        name="blocked_squares",
        category=VariantCategory.BOARD,
        split="test",
        description="8 deterministic barrier squares in ranks 3–6 (a3..h6) block movement, sliding, and rays.",
        formal_rule=(
            "Eight barrier squares are sampled deterministically from `seed` over squares 16..47 (a3..h6). "
            "No piece may land on a blocked square, and sliding pieces, 2-square pawn pushes, and ray-attacks/"
            "checks/pins are blocked by barriers. Knights may leap over barriers. Barriers are rendered as 'X' in "
            "board_matrix_64/board_fen_with_barriers and preserved in state dicts and canonical keys."
        ),
        is_visibly_identifiable=True,
        uses_blocked_squares=True,
    ),
    "long_board": VariantSpec(
        name="long_board",
        category=VariantCategory.BOARD,
        split="test",
        description="Frozen 8x8 Extended Opening Pawn Push approximation of an 8x20 long board.",
        formal_rule=(
            "Pawns on their starting rank (rank 2 for White, rank 7 for Black) may push 1..6 squares straight "
            "forward provided all intermediate and destination squares are empty and the move does not land on the "
            "promotion rank (max 5 squares from rank 2->7 or 7->2). En passant target square is set only on a "
            "2-square push. All other pieces, captures, and checks are standard."
        ),
    ),
    "knight_army": VariantSpec(
        name="knight_army",
        category=VariantCategory.PIECES,
        split="test",
        description="Setup-only null control: each side starts with 15 knights and 1 king.",
        formal_rule=(
            "Setup-only null control. When initialized without a custom FEN (fen is None / default start), White "
            "starts with King on e1 and Knights on a1..h2 (15 knights), Black starts with King on e8 and Knights "
            "on a7..h8 (15 knights), with castling rights '-'. Passed custom FENs are never overwritten. All "
            "movement, capture, and check rules are standard."
        ),
        is_setup_only_control=True,
        is_visibly_identifiable=True,
    ),
    "antichess": VariantSpec(
        name="antichess",
        category=VariantCategory.OBJECTIVE,
        split="test",
        description="Lose all pieces (or get stalemated) to win; captures are mandatory.",
        formal_rule=(
            "Delegates to python-chess chess.variant.AntichessBoard: captures (including en passant) are "
            "mandatory if any exist; no check, no checkmate, no castling; pawns may promote to King; losing all "
            "pieces or having no legal moves (stalemate) wins for the side to move."
        ),
        no_check_rules=True,
    ),
}

COMPOSITIONAL_VARIANT_SPECS: dict[str, VariantSpec] = {
    "diagonal_rook_and_piece_capture": VariantSpec(
        name="diagonal_rook_and_piece_capture",
        category=VariantCategory.COMPOSITIONAL,
        split="compositional",
        description="Rooks move diagonally AND captured non-king pieces convert to capturer's color.",
        formal_rule=(
            "Composition of `diagonal_rook` and `piece_capture`: rooks slide and check diagonally, and any "
            "captured non-king piece converts to the capturer's color on the capture square."
        ),
        is_compositional=True,
        components=("diagonal_rook", "piece_capture"),
    ),
    "mobile_king_and_dark_immunity": VariantSpec(
        name="mobile_king_and_dark_immunity",
        category=VariantCategory.COMPOSITIONAL,
        split="compositional",
        description="Kings move like queens AND non-king pieces on dark squares are immune to capture.",
        formal_rule=(
            "Composition of `mobile_king` and `dark_square_immunity`: kings slide and check like queens along "
            "all 8 rays, while non-king pieces on dark squares ((rank+file)%2==0) cannot be captured; kings "
            "remain vulnerable to check and checkmate everywhere."
        ),
        is_compositional=True,
        components=("mobile_king", "dark_square_immunity"),
    ),
    "short_knight_and_antichess": VariantSpec(
        name="short_knight_and_antichess",
        category=VariantCategory.COMPOSITIONAL,
        split="compositional",
        description="Knights move via Ferz+Dabbaba ((±1,±1),(±2,0),(0,±2)) AND antichess rules apply.",
        formal_rule=(
            "Composition of `short_knight` and `antichess`: knights jump using Ferz+Dabbaba offsets "
            "(±1,±1), (±2,0), (0,±2); captures are mandatory, king promotions are allowed, no check/castling, "
            "and losing all pieces or being stalemated wins. Color-bound knight insufficient-material check applies."
        ),
        is_compositional=True,
        components=("short_knight", "antichess"),
        no_check_rules=True,
    ),
}

# Preserve backward-compatible 13 single variants dict AND unified all-variants dict
VARIANTS: dict[str, VariantSpec] = {
    **SINGLE_VARIANT_SPECS,
    **COMPOSITIONAL_VARIANT_SPECS,
}

ALL_VARIANTS: dict[str, VariantSpec] = {
    "standard": STANDARD_SPEC,
    **VARIANTS,
}

TRAIN_VARIANTS: list[str] = [
    "diagonal_rook",
    "straight_bishop",
    "weakened_queen",
    "extended_pawn",
]

TEST_VARIANTS: list[str] = [
    "short_knight",
    "mobile_king",
    "piece_capture",
    "forward_pawn_capture",
    "dark_square_immunity",
    "blocked_squares",
    "long_board",
    "knight_army",
    "antichess",
]

COMPOSITIONAL_VARIANTS: list[str] = [
    "diagonal_rook_and_piece_capture",
    "mobile_king_and_dark_immunity",
    "short_knight_and_antichess",
]

SINGLE_VARIANTS: list[str] = TRAIN_VARIANTS + TEST_VARIANTS
ALL_VARIANT_NAMES: list[str] = ["standard"] + SINGLE_VARIANTS + COMPOSITIONAL_VARIANTS
