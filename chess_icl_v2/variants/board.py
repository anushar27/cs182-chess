"""VariantBoard base class built directly on python-chess chess.Board.

Provides variant-aware bitboard attack masks, check detection, pseudo-legal
and legal move generation, terminal outcome detection, barrier rendering, and
state serialization helpers shared by all v2 chess variants.
"""

from __future__ import annotations

import copy
import random
from typing import Any, ClassVar, Iterable, Iterator, Optional, Type, TypeVar

import chess

from .specs import STANDARD_SPEC, VariantSpec

TVariantBoard = TypeVar("TVariantBoard", bound="VariantBoard")

_USE_STARTING_FEN = object()

PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def _build_short_knight_table() -> list[int]:
    """Precompute 64-square Ferz + Dabbaba attack bitboards: (±1,±1), (±2,0), (0,±2)."""
    offsets = (
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
        (2, 0),
        (-2, 0),
        (0, 2),
        (0, -2),
    )
    table: list[int] = [0] * 64
    for sq in range(64):
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        mask = 0
        for dr, df in offsets:
            nr, nf = rank + dr, file + df
            if 0 <= nr <= 7 and 0 <= nf <= 7:
                mask |= chess.BB_SQUARES[chess.square(nf, nr)]
        table[sq] = mask
    return table


def _build_pawn_step_tables() -> tuple[dict[chess.Color, list[int]], dict[chess.Color, list[int]]]:
    """Precompute 1-square forward and backward pawn step bitboards by color."""
    forward: dict[chess.Color, list[int]] = {chess.WHITE: [0] * 64, chess.BLACK: [0] * 64}
    backward: dict[chess.Color, list[int]] = {chess.WHITE: [0] * 64, chess.BLACK: [0] * 64}
    for sq in range(64):
        rank = chess.square_rank(sq)
        if rank < 7:
            forward[chess.WHITE][sq] = chess.BB_SQUARES[sq + 8]
            backward[chess.BLACK][sq] = chess.BB_SQUARES[sq + 8]
        if rank > 0:
            forward[chess.BLACK][sq] = chess.BB_SQUARES[sq - 8]
            backward[chess.WHITE][sq] = chess.BB_SQUARES[sq - 8]
    return forward, backward


BB_SHORT_KNIGHT_ATTACKS: list[int] = _build_short_knight_table()
BB_FORWARD_PAWN_STEP, BB_BACKWARD_PAWN_STEP = _build_pawn_step_tables()

_BLOCKED_POOL: tuple[int, ...] = tuple(range(16, 48))  # a3..h6 (ranks 3-6, 0-indexed ranks 2-5)

# Global class registry populated by variants_impl.py for VariantBoard.from_state_dict
_VARIANT_CLASS_MAP: dict[str, Type["VariantBoard"]] = {}


def register_variant_class(name: str, cls: Type["VariantBoard"]) -> None:
    """Register a VariantBoard subclass under its canonical variant name."""
    _VARIANT_CLASS_MAP[name] = cls


def _diag_attacks(square: chess.Square, occupied: chess.Bitboard) -> chess.Bitboard:
    return chess.BB_DIAG_ATTACKS[square][chess.BB_DIAG_MASKS[square] & occupied]


def _ortho_attacks(square: chess.Square, occupied: chess.Bitboard) -> chess.Bitboard:
    return (
        chess.BB_RANK_ATTACKS[square][chess.BB_RANK_MASKS[square] & occupied]
        | chess.BB_FILE_ATTACKS[square][chess.BB_FILE_MASKS[square] & occupied]
    )


class VariantBoard(chess.Board):
    """Base board class for all chess_icl_v2 variants, subclassing chess.Board.

    Rule knobs are declarative class attributes so single and compositional
    variants combine orthogonally without duplicating check/pin/legality logic.
    """

    default_spec: ClassVar[VariantSpec] = STANDARD_SPEC

    # Declarative rule hooks
    _rook_mode: ClassVar[str] = "straight"  # "straight" | "diagonal"
    _bishop_mode: ClassVar[str] = "diagonal"  # "diagonal" | "straight"
    _queen_mode: ClassVar[str] = "queen"  # "queen" | "straight"
    _knight_mode: ClassVar[str] = "standard"  # "standard" | "short"
    _king_mode: ClassVar[str] = "standard"  # "standard" | "queen"
    _pawn_forward_capture: ClassVar[bool] = False
    _pawn_extended_push: ClassVar[bool] = False
    _pawn_long_opening_push: ClassVar[bool] = False
    _dark_square_immunity: ClassVar[bool] = False
    _convert_captures: ClassVar[bool] = False
    _no_check_rules: ClassVar[bool] = False
    _require_both_kings: ClassVar[bool] = False
    _uses_blocked_squares: ClassVar[bool] = False

    def __init__(
        self,
        fen: Optional[str] = _USE_STARTING_FEN,  # type: ignore[assignment]
        *,
        chess960: bool = False,
        variant: Optional[VariantSpec] = None,
        seed: int = 0,
        blocked: Optional[Iterable[int]] = None,
    ) -> None:
        self.variant: VariantSpec = variant if variant is not None else type(self).default_spec
        self.seed: int = int(seed)
        self.blocked: set[int] = set()
        self.blocked_mask: int = 0

        if self._uses_blocked_squares or blocked is not None:
            if blocked is not None:
                self._set_blocked_squares(blocked)
            else:
                rng = random.Random(self.seed)
                self._set_blocked_squares(rng.sample(_BLOCKED_POOL, 8))

        if fen is _USE_STARTING_FEN:
            super().__init__(type(self).starting_fen, chess960=chess960)
        else:
            super().__init__(fen, chess960=chess960)

    def _set_blocked_squares(self, squares: Iterable[int]) -> None:
        sq_set = {int(sq) for sq in squares}
        mask = 0
        for sq in sq_set:
            if not (0 <= sq < 64):
                raise ValueError(f"Blocked square index out of range 0..63: {sq}")
            mask |= chess.BB_SQUARES[sq]
        self.blocked = sq_set
        self.blocked_mask = mask

    def _has_custom_rules(self) -> bool:
        return (
            self._rook_mode != "straight"
            or self._bishop_mode != "diagonal"
            or self._queen_mode != "queen"
            or self._knight_mode != "standard"
            or self._king_mode != "standard"
            or self._pawn_forward_capture
            or self._pawn_extended_push
            or self._pawn_long_opening_push
            or self._dark_square_immunity
            or self._convert_captures
            or self._no_check_rules
            or self._require_both_kings
            or self.blocked_mask != 0
        )

    # ── Attack & Check Detection ─────────────────────────────────────────────

    def attackers_mask(
        self,
        color: chess.Color,
        square: chess.Square,
        occupied: Optional[chess.Bitboard] = None,
    ) -> chess.Bitboard:
        if not self._has_custom_rules():
            return super().attackers_mask(color, square, occupied)

        occ = (self.occupied if occupied is None else occupied) | self.blocked_mask

        ortho_sliders = self.queens
        if self._rook_mode == "straight":
            ortho_sliders |= self.rooks
        if self._bishop_mode == "straight":
            ortho_sliders |= self.bishops
        if self._king_mode == "queen":
            ortho_sliders |= self.kings

        diag_sliders = 0
        if self._queen_mode == "queen":
            diag_sliders |= self.queens
        if self._bishop_mode == "diagonal":
            diag_sliders |= self.bishops
        if self._rook_mode == "diagonal":
            diag_sliders |= self.rooks
        if self._king_mode == "queen":
            diag_sliders |= self.kings

        knight_mask = (
            BB_SHORT_KNIGHT_ATTACKS[square]
            if self._knight_mode == "short"
            else chess.BB_KNIGHT_ATTACKS[square]
        )

        attackers = knight_mask & self.knights
        if self._king_mode == "standard":
            attackers |= chess.BB_KING_ATTACKS[square] & self.kings

        if ortho_sliders:
            attackers |= _ortho_attacks(square, occ) & ortho_sliders
        if diag_sliders:
            attackers |= _diag_attacks(square, occ) & diag_sliders

        pawn_attackers = chess.BB_PAWN_ATTACKS[not color][square] & self.pawns
        if self._pawn_forward_capture:
            pawn_attackers |= BB_BACKWARD_PAWN_STEP[color][square] & self.pawns
        attackers |= pawn_attackers

        return attackers & self.occupied_co[color]

    def attacks_mask(self, square: chess.Square) -> chess.Bitboard:
        if not self._has_custom_rules():
            return super().attacks_mask(square)

        bb_square = chess.BB_SQUARES[square]
        occ = self.occupied | self.blocked_mask
        valid_targets = ~self.blocked_mask & chess.BB_ALL

        if bb_square & self.pawns:
            color = bool(bb_square & self.occupied_co[chess.WHITE])
            att = chess.BB_PAWN_ATTACKS[color][square]
            if self._pawn_forward_capture:
                att |= BB_FORWARD_PAWN_STEP[color][square]
            return att & valid_targets
        if bb_square & self.knights:
            att = (
                BB_SHORT_KNIGHT_ATTACKS[square]
                if self._knight_mode == "short"
                else chess.BB_KNIGHT_ATTACKS[square]
            )
            return att & valid_targets
        if bb_square & self.kings:
            if self._king_mode == "queen":
                att = _ortho_attacks(square, occ) | _diag_attacks(square, occ)
            else:
                att = chess.BB_KING_ATTACKS[square]
            return att & valid_targets

        att = 0
        if bb_square & self.bishops:
            att |= (
                _diag_attacks(square, occ)
                if self._bishop_mode == "diagonal"
                else _ortho_attacks(square, occ)
            )
        if bb_square & self.rooks:
            att |= (
                _ortho_attacks(square, occ)
                if self._rook_mode == "straight"
                else _diag_attacks(square, occ)
            )
        if bb_square & self.queens:
            att |= _ortho_attacks(square, occ)
            if self._queen_mode == "queen":
                att |= _diag_attacks(square, occ)
        return att & valid_targets

    def checkers_mask(self) -> chess.Bitboard:
        if self._no_check_rules:
            return chess.BB_EMPTY
        king = self.king(self.turn)
        return chess.BB_EMPTY if king is None else self.attackers_mask(not self.turn, king)

    def is_check(self) -> bool:
        return bool(self.checkers_mask())

    def is_into_check(self, move: chess.Move) -> bool:
        if self._no_check_rules:
            return False
        if not self._has_custom_rules():
            return super().is_into_check(move)

        king_sq = self.king(self.turn)
        if king_sq is None:
            return False

        from_bb = chess.BB_SQUARES[move.from_square]
        to_bb = chess.BB_SQUARES[move.to_square]

        if self.is_castling(move):
            # Verify start, transit, and end squares are unattacked with moving pieces displaced
            c_move = self._to_chess960(move)
            a_side = chess.square_file(c_move.to_square) < chess.square_file(c_move.from_square)
            backrank = chess.BB_RANK_1 if self.turn == chess.WHITE else chess.BB_RANK_8
            king_to = chess.msb((chess.BB_FILE_C if a_side else chess.BB_FILE_G) & backrank)
            rook_to = chess.msb((chess.BB_FILE_D if a_side else chess.BB_FILE_F) & backrank)
            rook_from = c_move.to_square
            king_path = chess.between(king_sq, king_to)
            if self._attacked_for_king(king_path | from_bb, self.occupied ^ from_bb):
                return True
            occ_after = (self.occupied ^ from_bb ^ chess.BB_SQUARES[rook_from]) | chess.BB_SQUARES[king_to] | chess.BB_SQUARES[rook_to]
            return bool(self.attackers_mask(not self.turn, king_to, occupied=occ_after))

        if self.is_en_passant(move):
            cap_sq = move.to_square + (-8 if self.turn == chess.WHITE else 8)
            cap_bb = chess.BB_SQUARES[cap_sq]
            occ = (self.occupied & ~from_bb & ~cap_bb) | to_bb
            return bool(self.attackers_mask(not self.turn, king_sq, occupied=occ) & ~cap_bb)

        occ = (self.occupied & ~from_bb) | to_bb
        new_king = move.to_square if move.from_square == king_sq else king_sq
        # Exclude captured opponent piece at to_square since our moving piece replaces it
        return bool(self.attackers_mask(not self.turn, new_king, occupied=occ) & ~to_bb)

    def was_into_check(self) -> bool:
        if self._no_check_rules:
            return False
        king = self.king(not self.turn)
        return king is not None and bool(self.attackers_mask(self.turn, king))

    # ── Castling & Mobile-King Chess960 Disambiguation ───────────────────────

    def _is_capture_blocked_by_immunity(self, move: chess.Move) -> bool:
        if not self._dark_square_immunity:
            return False
        if self.is_en_passant(move):
            cap_sq = move.to_square + (-8 if self.turn == chess.WHITE else 8)
            return bool(chess.BB_SQUARES[cap_sq] & chess.BB_DARK_SQUARES)
        to_bb = chess.BB_SQUARES[move.to_square]
        return bool((self.occupied_co[not self.turn] & ~self.kings) & to_bb & chess.BB_DARK_SQUARES)

    def generate_castling_moves(
        self,
        from_mask: chess.Bitboard = chess.BB_ALL,
        to_mask: chess.Bitboard = chess.BB_ALL,
    ) -> Iterator[chess.Move]:
        if self.is_variant_end():
            return

        backrank = chess.BB_RANK_1 if self.turn == chess.WHITE else chess.BB_RANK_8
        king = self.occupied_co[self.turn] & self.kings & ~self.promoted & backrank & from_mask
        king &= -king
        if not king:
            return

        bb_c = chess.BB_FILE_C & backrank
        bb_d = chess.BB_FILE_D & backrank
        bb_f = chess.BB_FILE_F & backrank
        bb_g = chess.BB_FILE_G & backrank

        occ_with_blocked = self.occupied | self.blocked_mask

        for candidate in chess.scan_reversed(self.clean_castling_rights() & backrank & to_mask):
            rook = chess.BB_SQUARES[candidate]
            if rook & self.blocked_mask:
                continue

            a_side = rook < king
            king_to = bb_c if a_side else bb_g
            rook_to = bb_d if a_side else bb_f

            king_path = chess.between(chess.msb(king), chess.msb(king_to))
            rook_path = chess.between(candidate, chess.msb(rook_to))

            if not (
                ((occ_with_blocked ^ king ^ rook) & (king_path | rook_path | king_to | rook_to))
                or self._attacked_for_king(king_path | king, self.occupied ^ king)
                or self._attacked_for_king(king_to, self.occupied ^ king ^ rook ^ rook_to)
            ):
                yield super()._from_chess960(self.chess960, chess.msb(king), candidate)

    def _to_chess960(self, move: chess.Move) -> chess.Move:
        if self._king_mode != "queen":
            return super()._to_chess960(move)
        # In mobile_king, only treat e1g1/e1c1/e8g8/e8c8 as castling when actually a legal castling move
        if move in self.generate_castling_moves():
            return super()._to_chess960(move)
        return move

    def _from_chess960(
        self,
        chess960: bool,
        from_square: chess.Square,
        to_square: chess.Square,
        promotion: Optional[chess.PieceType] = None,
        drop: Optional[chess.PieceType] = None,
    ) -> chess.Move:
        if self._king_mode != "queen":
            return super()._from_chess960(chess960, from_square, to_square, promotion, drop)
        # Only translate E1->H1 / E1->A1 etc. back to castling UCI when H1/A1 holds our own rook
        if not chess960 and promotion is None and drop is None:
            our_rooks = self.rooks & self.occupied_co[self.turn]
            if from_square == chess.E1 and self.kings & chess.BB_E1:
                if to_square == chess.H1 and our_rooks & chess.BB_H1:
                    return chess.Move(chess.E1, chess.G1)
                if to_square == chess.A1 and our_rooks & chess.BB_A1:
                    return chess.Move(chess.E1, chess.C1)
            elif from_square == chess.E8 and self.kings & chess.BB_E8:
                if to_square == chess.H8 and our_rooks & chess.BB_H8:
                    return chess.Move(chess.E8, chess.G8)
                if to_square == chess.A8 and our_rooks & chess.BB_A8:
                    return chess.Move(chess.E8, chess.C8)
        return chess.Move(from_square, to_square, promotion, drop)

    def is_castling(self, move: chess.Move) -> bool:
        if self._king_mode != "queen":
            return super().is_castling(move)
        if not (self.kings & chess.BB_SQUARES[move.from_square]):
            return False
        if self.rooks & self.occupied_co[self.turn] & chess.BB_SQUARES[move.to_square]:
            return True
        return move in self.generate_castling_moves()

    # ── Move Generation ──────────────────────────────────────────────────────

    def _yield_pawn_move(
        self, from_sq: chess.Square, to_sq: chess.Square, promo_rank: int
    ) -> Iterator[chess.Move]:
        if chess.square_rank(to_sq) == promo_rank:
            yield chess.Move(from_sq, to_sq, chess.QUEEN)
            yield chess.Move(from_sq, to_sq, chess.ROOK)
            yield chess.Move(from_sq, to_sq, chess.BISHOP)
            yield chess.Move(from_sq, to_sq, chess.KNIGHT)
        else:
            yield chess.Move(from_sq, to_sq)

    def generate_pseudo_legal_moves(
        self,
        from_mask: chess.Bitboard = chess.BB_ALL,
        to_mask: chess.Bitboard = chess.BB_ALL,
    ) -> Iterator[chess.Move]:
        if not self._has_custom_rules():
            yield from super().generate_pseudo_legal_moves(from_mask, to_mask)
            return

        our_pieces = self.occupied_co[self.turn]
        valid_to = to_mask & ~self.blocked_mask
        immune_targets = (
            ((self.occupied_co[not self.turn] & ~self.kings) & chess.BB_DARK_SQUARES)
            if self._dark_square_immunity
            else 0
        )

        # Castling
        castling_pairs: set[tuple[int, int]] = set()
        if (from_mask & self.kings) and not self._no_check_rules:
            for cmove in self.generate_castling_moves(from_mask, valid_to):
                castling_pairs.add((cmove.from_square, cmove.to_square))
                yield cmove

        # Non-pawn piece moves
        non_pawns = our_pieces & ~self.pawns & from_mask
        for from_square in chess.scan_reversed(non_pawns):
            moves = self.attacks_mask(from_square) & ~our_pieces & ~immune_targets & valid_to
            for to_square in chess.scan_reversed(moves):
                if (from_square, to_square) in castling_pairs:
                    continue
                yield chess.Move(from_square, to_square)

        # Pawn moves
        pawns = self.pawns & our_pieces & from_mask
        if not pawns:
            return

        direction = 1 if self.turn == chess.WHITE else -1
        promo_rank = 7 if self.turn == chess.WHITE else 0
        start_rank = 1 if self.turn == chess.WHITE else 6
        occ = self.occupied | self.blocked_mask

        # Diagonal captures (and forward captures if enabled)
        for from_square in chess.scan_reversed(pawns):
            cap_targets = (
                chess.BB_PAWN_ATTACKS[self.turn][from_square]
                & self.occupied_co[not self.turn]
                & ~immune_targets
                & valid_to
            )
            if self._pawn_forward_capture:
                cap_targets |= (
                    BB_FORWARD_PAWN_STEP[self.turn][from_square]
                    & self.occupied_co[not self.turn]
                    & ~immune_targets
                    & valid_to
                )
            for to_square in chess.scan_reversed(cap_targets):
                yield from self._yield_pawn_move(from_square, to_square, promo_rank)

        # Pawn pushes
        if self._pawn_extended_push:
            for from_square in chess.scan_reversed(pawns):
                r = chess.square_rank(from_square)
                f = chess.square_file(from_square)
                r1 = r + direction
                if 0 <= r1 <= 7:
                    sq1 = chess.square(f, r1)
                    if not (occ & chess.BB_SQUARES[sq1]):
                        if chess.BB_SQUARES[sq1] & valid_to:
                            yield from self._yield_pawn_move(from_square, sq1, promo_rank)
                        if r1 != promo_rank:
                            r2 = r + 2 * direction
                            if 0 <= r2 <= 7:
                                sq2 = chess.square(f, r2)
                                if not (occ & chess.BB_SQUARES[sq2]) and (chess.BB_SQUARES[sq2] & valid_to):
                                    yield from self._yield_pawn_move(from_square, sq2, promo_rank)
        elif self._pawn_long_opening_push:
            for from_square in chess.scan_reversed(pawns):
                r = chess.square_rank(from_square)
                f = chess.square_file(from_square)
                r1 = r + direction
                if 0 <= r1 <= 7:
                    sq1 = chess.square(f, r1)
                    if not (occ & chess.BB_SQUARES[sq1]):
                        if chess.BB_SQUARES[sq1] & valid_to:
                            yield from self._yield_pawn_move(from_square, sq1, promo_rank)
                        if r == start_rank and r1 != promo_rank:
                            for dist in range(2, 7):
                                tr = r + dist * direction
                                if not (0 <= tr <= 7) or tr == promo_rank:
                                    break
                                tsq = chess.square(f, tr)
                                if occ & chess.BB_SQUARES[tsq]:
                                    break
                                if chess.BB_SQUARES[tsq] & valid_to:
                                    yield chess.Move(from_square, tsq)
        else:
            if self.turn == chess.WHITE:
                single_moves = (pawns << 8) & ~occ
                double_moves = (single_moves << 8) & ~occ & (chess.BB_RANK_3 | chess.BB_RANK_4)
            else:
                single_moves = (pawns >> 8) & ~occ
                double_moves = (single_moves >> 8) & ~occ & (chess.BB_RANK_6 | chess.BB_RANK_5)

            single_moves &= valid_to
            double_moves &= valid_to

            for to_square in chess.scan_reversed(single_moves):
                from_square = to_square + (8 if self.turn == chess.BLACK else -8)
                yield from self._yield_pawn_move(from_square, to_square, promo_rank)

            for to_square in chess.scan_reversed(double_moves):
                from_square = to_square + (16 if self.turn == chess.BLACK else -16)
                yield chess.Move(from_square, to_square)

        # En passant
        if self.ep_square:
            for ep_move in self.generate_pseudo_legal_ep(from_mask, valid_to):
                if not self._is_capture_blocked_by_immunity(ep_move):
                    yield ep_move

    def is_pseudo_legal(self, move: chess.Move) -> bool:
        if not self._has_custom_rules():
            return super().is_pseudo_legal(move)

        if not move or move.drop:
            return False

        piece = self.piece_type_at(move.from_square)
        if not piece:
            return False

        from_mask = chess.BB_SQUARES[move.from_square]
        to_mask = chess.BB_SQUARES[move.to_square]

        if not (self.occupied_co[self.turn] & from_mask):
            return False
        if to_mask & self.blocked_mask:
            return False
        if self.occupied_co[self.turn] & to_mask:
            return False
        if self._is_capture_blocked_by_immunity(move):
            return False

        if move.promotion:
            if piece != chess.PAWN:
                return False
            if self.turn == chess.WHITE and chess.square_rank(move.to_square) != 7:
                return False
            if self.turn == chess.BLACK and chess.square_rank(move.to_square) != 0:
                return False

        if piece == chess.KING and not self._no_check_rules:
            castling_candidate = (
                move
                if self._king_mode == "queen"
                else super()._from_chess960(self.chess960, move.from_square, move.to_square)
            )
            if castling_candidate in self.generate_castling_moves():
                return True

        if piece == chess.PAWN:
            return move in self.generate_pseudo_legal_moves(from_mask, to_mask)

        return bool(self.attacks_mask(move.from_square) & to_mask)

    def generate_legal_moves(
        self,
        from_mask: chess.Bitboard = chess.BB_ALL,
        to_mask: chess.Bitboard = chess.BB_ALL,
    ) -> Iterator[chess.Move]:
        if not self._has_custom_rules():
            yield from super().generate_legal_moves(from_mask, to_mask)
            return

        if self.is_variant_end():
            return

        for move in self.generate_pseudo_legal_moves(from_mask, to_mask):
            if not self.is_into_check(move):
                yield move

    @property
    def legal_moves(self) -> chess.LegalMoveGenerator:
        return chess.LegalMoveGenerator(self)

    # ── Capture Conversion Hook (piece_capture) ──────────────────────────────

    def _push_capture(
        self,
        move: chess.Move,
        capture_square: chess.Square,
        piece_type: chess.PieceType,
        was_promoted: bool,
    ) -> None:
        if not self._convert_captures:
            super()._push_capture(move, capture_square, piece_type, was_promoted)
            return
        # Convert captured non-king piece to capturer's color, unless the capturer is a King
        # (a King capturing a piece must remain a King on the board).
        if piece_type != chess.KING and self.piece_type_at(move.to_square) != chess.KING:
            self._set_piece_at(move.to_square, piece_type, self.turn, was_promoted)

    # ── Terminal & Outcome Logic ─────────────────────────────────────────────

    def is_variant_end(self) -> bool:
        if self._require_both_kings:
            w_king = bool(self.kings & self.occupied_co[chess.WHITE])
            b_king = bool(self.kings & self.occupied_co[chess.BLACK])
            if not w_king or not b_king:
                return True
        return super().is_variant_end()

    def is_variant_win(self) -> bool:
        if self._require_both_kings:
            our_king = bool(self.kings & self.occupied_co[self.turn])
            their_king = bool(self.kings & self.occupied_co[not self.turn])
            if our_king and not their_king:
                return True
        return super().is_variant_win()

    def is_variant_loss(self) -> bool:
        if self._require_both_kings:
            our_king = bool(self.kings & self.occupied_co[self.turn])
            their_king = bool(self.kings & self.occupied_co[not self.turn])
            if not our_king and their_king:
                return True
        return super().is_variant_loss()

    def is_variant_draw(self) -> bool:
        if self._require_both_kings:
            w_king = bool(self.kings & self.occupied_co[chess.WHITE])
            b_king = bool(self.kings & self.occupied_co[chess.BLACK])
            if not w_king and not b_king:
                return True
        return super().is_variant_draw()

    def is_checkmate(self) -> bool:
        if self._no_check_rules or self.is_variant_end():
            return False
        if not self.is_check():
            return False
        return not any(self.generate_legal_moves())

    def is_stalemate(self) -> bool:
        if self._no_check_rules:
            # Handled via super() / AntichessBoard (where stalemate is not check)
            return not any(self.generate_legal_moves()) and not self.is_variant_end()
        if self.is_check() or self.is_variant_end():
            return False
        return not any(self.generate_legal_moves())

    def has_insufficient_material(self, color: chess.Color) -> bool:
        if not self._has_custom_rules():
            return super().has_insufficient_material(color)

        if self._uses_blocked_squares or self.blocked_mask:
            # Barriers can assist corner mates with minor pieces; only bare K vs K is strictly insufficient.
            return self.occupied == self.kings

        # Compute effective piece bitboards under movement overrides:
        # - straight_bishop: bishops move like rooks (sufficient)
        # - diagonal_rook: rooks move like bishops (insufficient like bishops)
        # - short_knight: knights are color-bound (insufficient like bishops)
        effective_rooks_or_queens = self.queens
        if self._rook_mode == "straight":
            effective_rooks_or_queens |= self.rooks
        if self._bishop_mode == "straight":
            effective_rooks_or_queens |= self.bishops

        effective_bishops = 0
        if self._bishop_mode == "diagonal":
            effective_bishops |= self.bishops
        if self._rook_mode == "diagonal":
            effective_bishops |= self.rooks
        if self._knight_mode == "short":
            effective_bishops |= self.knights

        effective_knights = self.knights if self._knight_mode == "standard" else 0

        if self.occupied_co[color] & (self.pawns | effective_rooks_or_queens):
            return False

        if self.occupied_co[color] & effective_knights:
            return (
                chess.popcount(self.occupied_co[color]) <= 2
                and not (self.occupied_co[not color] & ~self.kings & ~self.queens)
            )

        if self.occupied_co[color] & effective_bishops:
            same_color = (
                not (effective_bishops & chess.BB_DARK_SQUARES)
                or not (effective_bishops & chess.BB_LIGHT_SQUARES)
            )
            return same_color and not self.pawns and not effective_knights

        return True

    def outcome(self, *, claim_draw: bool = False) -> Optional[chess.Outcome]:
        return super().outcome(claim_draw=claim_draw)

    def is_game_over(self, *, claim_draw: bool = False) -> bool:
        return self.outcome(claim_draw=claim_draw) is not None

    # ── Mirroring, Copying & Serialization ───────────────────────────────────

    def apply_mirror(self) -> None:
        super().apply_mirror()
        if self.blocked:
            self._set_blocked_squares(chess.square_mirror(sq) for sq in self.blocked)

    def copy(self: TVariantBoard, *, stack: bool | int = True) -> TVariantBoard:
        board = super().copy(stack=stack)
        board.variant = self.variant
        board.seed = self.seed
        board.blocked = set(self.blocked)
        board.blocked_mask = self.blocked_mask
        return board

    def evaluate_material(self, color: chess.Color) -> int:
        """Evaluate material advantage from `color`'s perspective using standard values."""
        score = 0
        for piece_type, val in PIECE_VALUES.items():
            if val == 0:
                continue
            ours = chess.popcount(self.pieces_mask(piece_type, color))
            theirs = chess.popcount(self.pieces_mask(piece_type, not color))
            score += val * (ours - theirs)
        return score

    def board_matrix_64(self) -> list[str]:
        """Return a 64-element list indexed by chess.SQUARES (a1..h8) with '.', 'X', or piece symbol."""
        out: list[str] = ["."] * 64
        for sq in range(64):
            if sq in self.blocked:
                out[sq] = "X"
            else:
                piece = self.piece_at(sq)
                if piece is not None:
                    out[sq] = piece.symbol()
        return out

    def board_fen_with_barriers(self) -> str:
        """Return FEN board placement string with 'X' marking blocked barrier squares."""
        if not self.blocked:
            return self.board_fen()
        rows: list[str] = []
        for rank in range(7, -1, -1):
            empty = 0
            row_parts: list[str] = []
            for file in range(8):
                sq = chess.square(file, rank)
                if sq in self.blocked:
                    if empty:
                        row_parts.append(str(empty))
                        empty = 0
                    row_parts.append("X")
                else:
                    piece = self.piece_at(sq)
                    if piece is None:
                        empty += 1
                    else:
                        if empty:
                            row_parts.append(str(empty))
                            empty = 0
                        row_parts.append(piece.symbol())
            if empty:
                row_parts.append(str(empty))
            rows.append("".join(row_parts))
        return "/".join(rows)

    def canonical_state_key(self) -> str:
        """Return a canonical Markov state key including variant, barriers, turn, castling, and legal ep."""
        turn_str = "w" if self.turn == chess.WHITE else "b"
        ep_sq = self.ep_square if self.has_legal_en_passant() else None
        ep_str = chess.SQUARE_NAMES[ep_sq] if ep_sq is not None else "-"
        return f"{self.variant.name}:{self.board_fen_with_barriers()} {turn_str} {self.castling_xfen()} {ep_str}"

    def to_state_dict(self) -> dict[str, Any]:
        """Serialize the full board state into a JSON-compatible dictionary."""
        return {
            "variant": self.variant.name,
            "fen": self.fen(),
            "seed": self.seed,
            "blocked": sorted(self.blocked),
            "turn": bool(self.turn),
            "halfmove_clock": int(self.halfmove_clock),
            "fullmove_number": int(self.fullmove_number),
            "ep_square": self.ep_square,
            "castling_rights": int(self.castling_rights),
            "board_fen_with_barriers": self.board_fen_with_barriers(),
        }

    @classmethod
    def from_state_dict(cls: Type[TVariantBoard], d: dict[str, Any]) -> TVariantBoard:
        """Reconstruct a VariantBoard subclass instance from a state dict."""
        variant_name = d["variant"]
        target_cls: Type[VariantBoard] = _VARIANT_CLASS_MAP.get(variant_name, cls)
        kwargs: dict[str, Any] = {"seed": int(d.get("seed", 0))}
        if "blocked" in d and d["blocked"]:
            kwargs["blocked"] = d["blocked"]
        elif target_cls._uses_blocked_squares:
            kwargs["blocked"] = d.get("blocked", ())
        board = target_cls(d["fen"], **kwargs)
        if "ep_square" in d:
            board.ep_square = d["ep_square"]
        if "castling_rights" in d:
            board.castling_rights = int(d["castling_rights"])
        board.halfmove_clock = int(d.get("halfmove_clock", board.halfmove_clock))
        board.fullmove_number = int(d.get("fullmove_number", board.fullmove_number))
        return board  # type: ignore[return-value]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(variant={self.variant.name!r}, fen={self.fen()!r})"
