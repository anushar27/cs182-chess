"""Comprehensive independent audit & differential fuzzing suite for chess_icl_v2.variants.

Includes:
1. Differential fuzzing against `chess.Board` (standard) and
   `chess.variant.AntichessBoard` (antichess) over 300+ random & near-terminal positions.
2. Independent from-scratch pure-Python reference move & attack verifier
   (using explicit (rank, file) arithmetic, completely independent of VariantBoard)
   across all 13 single variants + 3 compositional variants on random trajectories
   and near-terminal positions.
3. Property audits:
   - Replay round-trip (forward play + pop + replay from start)
   - State dict round-trip (`to_state_dict` / `from_state_dict`)
   - Perspective-swap symmetry (vertical mirror + color flip; 180-deg rotation
     + color flip for dark_square_immunity variants)
4. Throughput benchmark (ms/ply) for all 17 variants.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import chess
import chess.variant

from .board import VariantBoard
from .specs import ALL_VARIANT_NAMES, COMPOSITIONAL_VARIANTS, SINGLE_VARIANTS
from .variants_impl import get_board

ORTHO_DIRS: tuple[tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAG_DIRS: tuple[tuple[int, int], ...] = ((1, 1), (1, -1), (-1, 1), (-1, -1))
ALL_DIRS: tuple[tuple[int, int], ...] = ORTHO_DIRS + DIAG_DIRS
STD_KNIGHT_OFFSETS: tuple[tuple[int, int], ...] = (
    (2, 1),
    (2, -1),
    (-2, 1),
    (-2, -1),
    (1, 2),
    (1, -2),
    (-1, 2),
    (-1, -2),
)
SHORT_KNIGHT_OFFSETS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
    (2, 0),
    (-2, 0),
    (0, 2),
    (0, -2),
)


# ── Pure-Python From-Scratch Reference Engine ────────────────────────────────


def _on_board(r: int, f: int) -> bool:
    return 0 <= r <= 7 and 0 <= f <= 7


def _is_dark_sq(sq: int) -> bool:
    return (chess.square_rank(sq) + chess.square_file(sq)) % 2 == 0


def _ref_piece_attacks_square(
    piece_type: int,
    piece_color: bool,
    from_sq: int,
    target_sq: int,
    piece_map: dict[int, tuple[int, bool]],
    blocked: set[int],
    variant_name: str,
) -> bool:
    """Explicit (rank, file) check whether piece at `from_sq` attacks `target_sq`."""
    if from_sq == target_sq:
        return False
    fr, ff = chess.square_rank(from_sq), chess.square_file(from_sq)
    tr, tf = chess.square_rank(target_sq), chess.square_file(target_sq)
    dr, df = tr - fr, tf - ff

    if piece_type == chess.PAWN:
        step = 1 if piece_color == chess.WHITE else -1
        if dr == step and abs(df) == 1:
            return True
        if variant_name == "forward_pawn_capture" and dr == step and df == 0:
            return True
        return False

    if piece_type == chess.KNIGHT:
        offsets = (
            SHORT_KNIGHT_OFFSETS
            if variant_name in ("short_knight", "short_knight_and_antichess")
            else STD_KNIGHT_OFFSETS
        )
        return (dr, df) in offsets

    if piece_type == chess.KING:
        if variant_name in ("mobile_king", "mobile_king_and_dark_immunity"):
            dirs = ALL_DIRS
        else:
            return max(abs(dr), abs(df)) == 1
    elif piece_type == chess.ROOK:
        dirs = (
            DIAG_DIRS
            if variant_name in ("diagonal_rook", "diagonal_rook_and_piece_capture")
            else ORTHO_DIRS
        )
    elif piece_type == chess.BISHOP:
        dirs = ORTHO_DIRS if variant_name == "straight_bishop" else DIAG_DIRS
    elif piece_type == chess.QUEEN:
        dirs = ORTHO_DIRS if variant_name == "weakened_queen" else ALL_DIRS
    else:
        return False

    # Sliding ray check
    step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
    step_f = 0 if df == 0 else (1 if df > 0 else -1)
    if (step_r, step_f) not in dirs:
        return False
    if step_r != 0 and step_f != 0 and abs(dr) != abs(df):
        return False

    r, f = fr + step_r, ff + step_f
    while _on_board(r, f):
        sq = chess.square(f, r)
        if sq in blocked:
            return False
        if sq == target_sq:
            return True
        if sq in piece_map:
            return False
        r += step_r
        f += step_f
    return False


def ref_attackers(
    piece_map: dict[int, tuple[int, bool]],
    attacker_color: bool,
    target_sq: int,
    blocked: set[int],
    variant_name: str,
) -> set[int]:
    """Return set of squares of `attacker_color` attacking `target_sq`."""
    attackers: set[int] = set()
    for sq, (pt, col) in piece_map.items():
        if col == attacker_color and _ref_piece_attacks_square(
            pt, col, sq, target_sq, piece_map, blocked, variant_name
        ):
            attackers.add(sq)
    return attackers


def _ref_is_immune_capture(
    piece_map: dict[int, tuple[int, bool]],
    turn: bool,
    to_sq: int,
    is_ep: bool,
    variant_name: str,
) -> bool:
    if variant_name not in ("dark_square_immunity", "mobile_king_and_dark_immunity"):
        return False
    if is_ep:
        cap_sq = to_sq + (-8 if turn == chess.WHITE else 8)
        return _is_dark_sq(cap_sq)
    if to_sq in piece_map:
        pt, col = piece_map[to_sq]
        if col != turn and pt != chess.KING and _is_dark_sq(to_sq):
            return True
    return False


def _ref_promo_types(variant_name: str) -> tuple[int, ...]:
    if variant_name in ("antichess", "short_knight_and_antichess"):
        return (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.KING)
    return (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)


def ref_pseudo_legal_moves(
    piece_map: dict[int, tuple[int, bool]],
    turn: bool,
    castling_rights: int,
    ep_square: Optional[int],
    blocked: set[int],
    variant_name: str,
) -> list[chess.Move]:
    """Independent pure-Python pseudo-legal move generator."""
    moves: list[chess.Move] = []
    promos = _ref_promo_types(variant_name)
    is_antichess = variant_name in ("antichess", "short_knight_and_antichess")
    dark_immune = variant_name in ("dark_square_immunity", "mobile_king_and_dark_immunity")

    for sq, (pt, col) in piece_map.items():
        if col != turn:
            continue
        r, f = chess.square_rank(sq), chess.square_file(sq)

        if pt == chess.PAWN:
            direction = 1 if turn == chess.WHITE else -1
            start_rank = 1 if turn == chess.WHITE else 6
            promo_rank = 7 if turn == chess.WHITE else 0
            ep_from_rank = 4 if turn == chess.WHITE else 3

            # 1-square push
            r1 = r + direction
            if _on_board(r1, f):
                sq1 = chess.square(f, r1)
                if sq1 not in piece_map and sq1 not in blocked:
                    if r1 == promo_rank:
                        for pr in promos:
                            moves.append(chess.Move(sq, sq1, promotion=pr))
                    else:
                        moves.append(chess.Move(sq, sq1))
                        # Multi-square pushes
                        if variant_name == "extended_pawn":
                            r2 = r + 2 * direction
                            if _on_board(r2, f):
                                sq2 = chess.square(f, r2)
                                if sq2 not in piece_map and sq2 not in blocked:
                                    if r2 == promo_rank:
                                        for pr in promos:
                                            moves.append(chess.Move(sq, sq2, promotion=pr))
                                    else:
                                        moves.append(chess.Move(sq, sq2))
                        elif variant_name == "long_board" and r == start_rank:
                            for dist in range(2, 7):
                                tr = r + dist * direction
                                if not _on_board(tr, f) or tr == promo_rank:
                                    break
                                tsq = chess.square(f, tr)
                                if tsq in piece_map or tsq in blocked:
                                    break
                                moves.append(chess.Move(sq, tsq))
                        elif r == start_rank:
                            r2 = r + 2 * direction
                            if _on_board(r2, f):
                                sq2 = chess.square(f, r2)
                                if sq2 not in piece_map and sq2 not in blocked:
                                    moves.append(chess.Move(sq, sq2))
                elif (
                    variant_name == "forward_pawn_capture"
                    and sq1 in piece_map
                    and piece_map[sq1][1] != turn
                    and sq1 not in blocked
                ):
                    if r1 == promo_rank:
                        for pr in promos:
                            moves.append(chess.Move(sq, sq1, promotion=pr))
                    else:
                        moves.append(chess.Move(sq, sq1))

            # Diagonal captures
            for df in (-1, 1):
                cf = f + df
                cr = r + direction
                if _on_board(cr, cf):
                    csq = chess.square(cf, cr)
                    if csq in blocked:
                        continue
                    if csq in piece_map and piece_map[csq][1] != turn:
                        if not (dark_immune and _is_dark_sq(csq) and piece_map[csq][0] != chess.KING):
                            if cr == promo_rank:
                                for pr in promos:
                                    moves.append(chess.Move(sq, csq, promotion=pr))
                            else:
                                moves.append(chess.Move(sq, csq))

            # En passant
            if (
                ep_square is not None
                and ep_square not in piece_map
                and ep_square not in blocked
                and r == ep_from_rank
            ):
                ep_r, ep_f = chess.square_rank(ep_square), chess.square_file(ep_square)
                if ep_r == r + direction and abs(ep_f - f) == 1:
                    if not _ref_is_immune_capture(piece_map, turn, ep_square, True, variant_name):
                        moves.append(chess.Move(sq, ep_square))

        elif pt == chess.KNIGHT:
            offsets = (
                SHORT_KNIGHT_OFFSETS
                if variant_name in ("short_knight", "short_knight_and_antichess")
                else STD_KNIGHT_OFFSETS
            )
            for dr, df in offsets:
                nr, nf = r + dr, f + df
                if _on_board(nr, nf):
                    nsq = chess.square(nf, nr)
                    if nsq in blocked:
                        continue
                    if nsq in piece_map and piece_map[nsq][1] == turn:
                        continue
                    if _ref_is_immune_capture(piece_map, turn, nsq, False, variant_name):
                        continue
                    moves.append(chess.Move(sq, nsq))

        elif pt == chess.KING and variant_name not in ("mobile_king", "mobile_king_and_dark_immunity"):
            for dr, df in ALL_DIRS:
                nr, nf = r + dr, f + df
                if _on_board(nr, nf):
                    nsq = chess.square(nf, nr)
                    if nsq in blocked:
                        continue
                    if nsq in piece_map and piece_map[nsq][1] == turn:
                        continue
                    if _ref_is_immune_capture(piece_map, turn, nsq, False, variant_name):
                        continue
                    moves.append(chess.Move(sq, nsq))

        else:
            if pt == chess.KING:
                dirs = ALL_DIRS
            elif pt == chess.ROOK:
                dirs = (
                    DIAG_DIRS
                    if variant_name in ("diagonal_rook", "diagonal_rook_and_piece_capture")
                    else ORTHO_DIRS
                )
            elif pt == chess.BISHOP:
                dirs = ORTHO_DIRS if variant_name == "straight_bishop" else DIAG_DIRS
            else:
                dirs = ORTHO_DIRS if variant_name == "weakened_queen" else ALL_DIRS

            for dr, df in dirs:
                nr, nf = r + dr, f + df
                while _on_board(nr, nf):
                    nsq = chess.square(nf, nr)
                    if nsq in blocked:
                        break
                    if nsq in piece_map:
                        if piece_map[nsq][1] != turn and not _ref_is_immune_capture(
                            piece_map, turn, nsq, False, variant_name
                        ):
                            moves.append(chess.Move(sq, nsq))
                        break
                    moves.append(chess.Move(sq, nsq))
                    nr += dr
                    nf += df

    # Castling (disabled in antichess variants)
    if not is_antichess:
        backrank = 0 if turn == chess.WHITE else 7
        king_sq = chess.E1 if turn == chess.WHITE else chess.E8
        if piece_map.get(king_sq) == (chess.KING, turn):
            for rook_file, king_to_file, rook_to_file in ((7, 6, 5), (0, 2, 3)):
                rook_sq = chess.square(rook_file, backrank)
                if not (castling_rights & chess.BB_SQUARES[rook_sq]):
                    continue
                if rook_sq in blocked or piece_map.get(rook_sq) != (chess.ROOK, turn):
                    continue
                king_to = chess.square(king_to_file, backrank)
                rook_to = chess.square(rook_to_file, backrank)

                # Path between king and rook plus destinations must be clear
                lo, hi = min(chess.square_file(king_sq), rook_file), max(chess.square_file(king_sq), rook_file)
                path_clear = True
                for pf in range(lo + 1, hi):
                    psq = chess.square(pf, backrank)
                    if psq in piece_map or psq in blocked:
                        path_clear = False
                        break
                if not path_clear:
                    continue
                if king_to in blocked or rook_to in blocked:
                    continue

                # King start/transit/end must not be attacked
                pm_no_king = {k: v for k, v in piece_map.items() if k != king_sq}
                step = 1 if king_to_file > 4 else -1
                safe = True
                for kf in range(4, king_to_file + step, step):
                    ksq = chess.square(kf, backrank)
                    if ref_attackers(pm_no_king, not turn, ksq, blocked, variant_name):
                        safe = False
                        break
                if safe:
                    # Deduplicate against mobile_king normal queen slides
                    cm = chess.Move(king_sq, king_to)
                    if cm in moves:
                        moves.remove(cm)
                    moves.append(cm)

    return moves


def _ref_apply_move(
    piece_map: dict[int, tuple[int, bool]],
    turn: bool,
    ep_square: Optional[int],
    move: chess.Move,
    variant_name: str,
) -> dict[int, tuple[int, bool]]:
    """Return updated piece_map after `move` for check verification."""
    new_pm = dict(piece_map)
    pt, col = new_pm.pop(move.from_square)

    # Castling check
    if pt == chess.KING and abs(chess.square_file(move.from_square) - chess.square_file(move.to_square)) == 2:
        backrank = chess.square_rank(move.from_square)
        if chess.square_file(move.to_square) == 6:
            rook_from = chess.square(7, backrank)
            rook_to = chess.square(5, backrank)
        else:
            rook_from = chess.square(0, backrank)
            rook_to = chess.square(3, backrank)
        if rook_from in new_pm and new_pm[rook_from] == (chess.ROOK, col):
            new_pm.pop(rook_from)
            new_pm[rook_to] = (chess.ROOK, col)
        new_pm[move.to_square] = (chess.KING, col)
        return new_pm

    # En passant capture
    if (
        pt == chess.PAWN
        and ep_square is not None
        and move.to_square == ep_square
        and move.to_square not in new_pm
        and abs(chess.square_file(move.to_square) - chess.square_file(move.from_square)) == 1
    ):
        cap_sq = ep_square + (-8 if turn == chess.WHITE else 8)
        new_pm.pop(cap_sq, None)

    placed_pt = move.promotion if move.promotion else pt
    if (
        variant_name in ("piece_capture", "diagonal_rook_and_piece_capture")
        and pt != chess.KING
        and move.to_square in new_pm
        and new_pm[move.to_square][0] != chess.KING
    ):
        placed_pt = new_pm[move.to_square][0]

    new_pm[move.to_square] = (placed_pt, col)
    return new_pm


def ref_legal_moves(board: VariantBoard) -> set[str]:
    """Compute sorted UCI legal moves purely from board state using reference verifier."""
    piece_map = {sq: (p.piece_type, p.color) for sq, p in board.piece_map().items()}
    vname = board.variant.name

    # Variant end check for piece_capture (missing king) or antichess (0 pieces)
    if vname in ("piece_capture", "diagonal_rook_and_piece_capture"):
        w_k = any(pt == chess.KING and col == chess.WHITE for pt, col in piece_map.values())
        b_k = any(pt == chess.KING and col == chess.BLACK for pt, col in piece_map.values())
        if not w_k or not b_k:
            return set()

    if vname in ("antichess", "short_knight_and_antichess"):
        w_any = any(col == chess.WHITE for _, col in piece_map.values())
        b_any = any(col == chess.BLACK for _, col in piece_map.values())
        if not w_any or not b_any:
            return set()

    pseudo = ref_pseudo_legal_moves(
        piece_map,
        board.turn,
        int(board.clean_castling_rights()),
        board.ep_square,
        set(board.blocked),
        vname,
    )

    if vname in ("antichess", "short_knight_and_antichess"):
        captures = [
            m for m in pseudo
            if (m.to_square in piece_map and piece_map[m.to_square][1] != board.turn)
            or (
                board.ep_square is not None
                and m.to_square == board.ep_square
                and piece_map[m.from_square][0] == chess.PAWN
                and abs(chess.square_file(m.to_square) - chess.square_file(m.from_square)) == 1
            )
        ]
        chosen = captures if captures else pseudo
        return {m.uci() for m in chosen}

    legal: set[str] = set()
    for m in pseudo:
        # Skip castling if currently in check (already ensured by castling path check including king_sq)
        pm_after = _ref_apply_move(piece_map, board.turn, board.ep_square, m, vname)
        king_sqs = [sq for sq, (pt, col) in pm_after.items() if pt == chess.KING and col == board.turn]
        if not king_sqs:
            legal.add(m.uci())
            continue
        if not ref_attackers(pm_after, not board.turn, king_sqs[0], set(board.blocked), vname):
            legal.add(m.uci())
    return legal


def ref_is_check(board: VariantBoard) -> bool:
    vname = board.variant.name
    if vname in ("antichess", "short_knight_and_antichess"):
        return False
    piece_map = {sq: (p.piece_type, p.color) for sq, p in board.piece_map().items()}
    king_sqs = [sq for sq, (pt, col) in piece_map.items() if pt == chess.KING and col == board.turn]
    if not king_sqs:
        return False
    return bool(ref_attackers(piece_map, not board.turn, king_sqs[0], set(board.blocked), vname))


# ── Helpers for Random, Near-Terminal & Symmetric Position Generation ────────


def _sample_random_board(
    variant_name: str, rng: random.Random, max_plies: int = 40
) -> VariantBoard:
    board = get_board(variant_name, seed=rng.randint(0, 1000))
    plies = rng.randint(0, max_plies)
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves or board.is_game_over():
            break
        # Bias slightly toward captures/checks for richer tactical states
        caps = [m for m in moves if board.is_capture(m)]
        if caps and rng.random() < 0.35:
            board.push(rng.choice(caps))
        else:
            board.push(rng.choice(moves))
    return board


def _sample_near_terminal_board(variant_name: str, rng: random.Random) -> VariantBoard:
    """Construct sparse endgame / near-checkmate / promotion / EP positions."""
    board = get_board(variant_name, seed=rng.randint(0, 1000))
    blocked = set(board.blocked)
    available = [sq for sq in range(64) if sq not in blocked]
    rng.shuffle(available)

    board.clear()
    is_antichess = variant_name in ("antichess", "short_knight_and_antichess")

    # Place kings (unless antichess sparse endgame)
    if not is_antichess:
        wk = available.pop()
        bk = available.pop()
        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))

    num_pieces = rng.randint(2, 7)
    piece_pool = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
    if is_antichess:
        piece_pool.append(chess.KING)

    for _ in range(num_pieces):
        if not available:
            break
        sq = available.pop()
        pt = rng.choice(piece_pool)
        col = rng.choice((chess.WHITE, chess.BLACK))
        if pt == chess.PAWN and chess.square_rank(sq) in (0, 7):
            pt = rng.choice((chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN))
        board.set_piece_at(sq, chess.Piece(pt, col))

    board.turn = rng.choice((chess.WHITE, chess.BLACK))
    board.castling_rights = chess.BB_EMPTY
    board.ep_square = None

    # Ensure non-moving side is not already leaving its king in check in check-based variants
    if not is_antichess:
        # Probe if opponent's king is currently attacked; if so, adjust turn or clear attacker
        if board.was_into_check():
            board.turn = not board.turn
            if board.was_into_check():
                # Fallback to a clean random trajectory near endgame
                return _sample_random_board(variant_name, rng, max_plies=55)
    return board


def _rotate_180_board(board: VariantBoard) -> VariantBoard:
    """180-degree rotation (sq ^ 63) + color swap, preserving dark squares (r+f)%2==0."""
    rotated = get_board(board.variant.name, fen=None, seed=board.seed)
    rotated.clear()
    for sq, piece in board.piece_map().items():
        rotated.set_piece_at(sq ^ 63, chess.Piece(piece.piece_type, not piece.color))
    rotated.turn = not board.turn
    rotated.castling_rights = chess.BB_EMPTY
    rotated.ep_square = (board.ep_square ^ 63) if board.ep_square is not None else None
    rotated.halfmove_clock = board.halfmove_clock
    rotated.fullmove_number = board.fullmove_number
    if board.blocked:
        rotated._set_blocked_squares(sq ^ 63 for sq in board.blocked)
    return rotated


# ── Audit Result Data Structures ─────────────────────────────────────────────


@dataclass
class VariantAuditResult:
    variant: str
    positions_checked: int = 0
    near_terminal_checked: int = 0
    legal_move_matches: int = 0
    check_matches: int = 0
    terminal_matches: int = 0
    replay_roundtrips: int = 0
    state_dict_roundtrips: int = 0
    symmetry_checks: int = 0
    ms_per_ply: float = 0.0
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.failures) == 0


# ── Audit Runners ────────────────────────────────────────────────────────────


def run_differential_fuzz(
    num_positions: int = 300, seed: int = 2026
) -> dict[str, VariantAuditResult]:
    """Fuzz `standard` vs chess.Board and `antichess` vs chess.variant.AntichessBoard."""
    rng = random.Random(seed)
    results: dict[str, VariantAuditResult] = {}

    for vname, ref_cls in (
        ("standard", chess.Board),
        ("antichess", chess.variant.AntichessBoard),
    ):
        res = VariantAuditResult(variant=vname)
        for i in range(num_positions):
            is_nt = i >= (num_positions // 2)
            if is_nt:
                vb = _sample_near_terminal_board(vname, rng)
                res.near_terminal_checked += 1
            else:
                vb = _sample_random_board(vname, rng, max_plies=45)

            rb = ref_cls(vb.fen())
            res.positions_checked += 1

            v_moves = {m.uci() for m in vb.legal_moves}
            r_moves = {m.uci() for m in rb.legal_moves}
            if v_moves == r_moves:
                res.legal_move_matches += 1
            else:
                res.failures.append(f"[{i}] move mismatch fen={vb.fen()}: {v_moves ^ r_moves}")
                continue

            if (
                vb.is_check() == rb.is_check()
                and vb.checkers_mask() == rb.checkers_mask()
            ):
                res.check_matches += 1
            else:
                res.failures.append(f"[{i}] check mismatch fen={vb.fen()}")
                continue

            v_out = vb.outcome(claim_draw=True)
            r_out = rb.outcome(claim_draw=True)
            if (
                vb.is_checkmate() == rb.is_checkmate()
                and vb.is_stalemate() == rb.is_stalemate()
                and vb.has_insufficient_material(chess.WHITE) == rb.has_insufficient_material(chess.WHITE)
                and vb.has_insufficient_material(chess.BLACK) == rb.has_insufficient_material(chess.BLACK)
                and vb.is_game_over(claim_draw=True) == rb.is_game_over(claim_draw=True)
                and v_out == r_out
            ):
                res.terminal_matches += 1
            else:
                res.failures.append(f"[{i}] terminal mismatch fen={vb.fen()}: {v_out} vs {r_out}")
                continue

            if v_moves:
                m_uci = rng.choice(sorted(v_moves))
                m = chess.Move.from_uci(m_uci)
                vb.push(m)
                rb.push(m)
                if vb.fen() != rb.fen():
                    res.failures.append(f"[{i}] post-push fen mismatch on {m_uci}")
                vb.pop()
                rb.pop()

        results[vname] = res
    return results


def run_reference_audit(
    variants: Optional[list[str]] = None,
    trajectories_per_variant: int = 20,
    plies_per_trajectory: int = 12,
    near_terminal_per_variant: int = 40,
    seed: int = 2026,
) -> dict[str, VariantAuditResult]:
    """Run independent from-scratch reference audit + property tests + speed bench."""
    rng = random.Random(seed)
    target_variants = variants if variants is not None else list(ALL_VARIANT_NAMES)
    results: dict[str, VariantAuditResult] = {}

    for vname in target_variants:
        res = VariantAuditResult(variant=vname)

        # 1. Trajectory fuzzing + replay round-trip + state_dict round-trip + symmetry
        for traj_idx in range(trajectories_per_variant):
            b_seed = rng.randint(0, 10000)
            b = get_board(vname, seed=b_seed)
            init_fen = b.fen()
            init_blocked = sorted(b.blocked)
            init_key = b.canonical_state_key()
            move_history: list[chess.Move] = []

            for ply in range(plies_per_trajectory):
                res.positions_checked += 1
                v_moves = {m.uci() for m in b.legal_moves}
                r_moves = ref_legal_moves(b)
                if v_moves == r_moves:
                    res.legal_move_matches += 1
                else:
                    res.failures.append(
                        f"traj {traj_idx} ply {ply} move diff fen={b.fen()} blocked={sorted(b.blocked)}: "
                        f"extra={v_moves - r_moves}, missing={r_moves - v_moves}"
                    )

                v_chk = b.is_check()
                r_chk = ref_is_check(b)
                if v_chk == r_chk:
                    res.check_matches += 1
                else:
                    res.failures.append(f"traj {traj_idx} ply {ply} check diff: {v_chk} vs {r_chk} fen={b.fen()}")

                # Terminal consistency
                if vname in ("antichess", "short_knight_and_antichess"):
                    has_pieces = bool(b.occupied_co[b.turn])
                    r_over = (not has_pieces) or (len(r_moves) == 0)
                    v_over = b.is_variant_end() or b.is_stalemate()
                    if r_over == v_over:
                        res.terminal_matches += 1
                    else:
                        res.failures.append(f"traj {traj_idx} ply {ply} antichess term diff")
                else:
                    r_mate = r_chk and len(r_moves) == 0
                    r_stale = (not r_chk) and len(r_moves) == 0
                    if b.is_checkmate() == r_mate and b.is_stalemate() == r_stale:
                        res.terminal_matches += 1
                    else:
                        res.failures.append(f"traj {traj_idx} ply {ply} term diff fen={b.fen()}")

                # State dict round-trip
                sd = b.to_state_dict()
                b_sd = VariantBoard.from_state_dict(sd)
                if (
                    b_sd.canonical_state_key() == b.canonical_state_key()
                    and b_sd.fen() == b.fen()
                    and b_sd.blocked == b.blocked
                    and {m.uci() for m in b_sd.legal_moves} == v_moves
                ):
                    res.state_dict_roundtrips += 1
                else:
                    res.failures.append(f"traj {traj_idx} ply {ply} state_dict round-trip failed")

                # Perspective-swap symmetry
                if vname in ("dark_square_immunity", "mobile_king_and_dark_immunity"):
                    # 180-deg rotation requires no castling rights (e1->d8 breaks castling geometry)
                    b_nocastle = VariantBoard.from_state_dict({**sd, "fen": b.fen()})
                    b_nocastle.castling_rights = chess.BB_EMPTY
                    rot = _rotate_180_board(b_nocastle)
                    orig_m = {f"{chess.SQUARE_NAMES[m.from_square ^ 63]}{chess.SQUARE_NAMES[m.to_square ^ 63]}{chess.piece_symbol(m.promotion) if m.promotion else ''}" for m in b_nocastle.legal_moves}
                    rot_m = {m.uci() for m in rot.legal_moves}
                    if orig_m == rot_m and b_nocastle.is_check() == rot.is_check():
                        res.symmetry_checks += 1
                    else:
                        res.failures.append(f"traj {traj_idx} ply {ply} 180-deg symmetry failed")
                else:
                    mir = b.mirror()
                    mir_expected = {
                        chess.Move(m.from_square ^ 56, m.to_square ^ 56, m.promotion).uci()
                        for m in b.legal_moves
                    }
                    mir_actual = {m.uci() for m in mir.legal_moves}
                    if mir_expected == mir_actual and b.is_check() == mir.is_check():
                        res.symmetry_checks += 1
                    else:
                        res.failures.append(f"traj {traj_idx} ply {ply} vertical mirror symmetry failed")

                if not v_moves or b.is_game_over():
                    break
                chosen_uci = rng.choice(sorted(v_moves))
                chosen = chess.Move.from_uci(chosen_uci)
                move_history.append(chosen)
                b.push(chosen)

            # Replay round-trip check (replay + pop back to init)
            final_key = b.canonical_state_key()
            final_fen = b.fen()
            replayed = get_board(vname, seed=b_seed, blocked=init_blocked if init_blocked else None)
            for mv in move_history:
                replayed.push(mv)
            ok_replay = replayed.canonical_state_key() == final_key and replayed.fen() == final_fen
            for _ in move_history:
                replayed.pop()
            ok_pop = replayed.canonical_state_key() == init_key and replayed.fen() == init_fen
            if ok_replay and ok_pop:
                res.replay_roundtrips += 1
            else:
                res.failures.append(f"traj {traj_idx} replay/pop round-trip failed")

        # 2. Near-terminal positions audit
        for nt_idx in range(near_terminal_per_variant):
            b = _sample_near_terminal_board(vname, rng)
            res.positions_checked += 1
            res.near_terminal_checked += 1

            v_moves = {m.uci() for m in b.legal_moves}
            r_moves = ref_legal_moves(b)
            if v_moves == r_moves:
                res.legal_move_matches += 1
            else:
                res.failures.append(
                    f"near-term {nt_idx} move diff fen={b.fen()} blocked={sorted(b.blocked)}: "
                    f"extra={v_moves - r_moves}, missing={r_moves - v_moves}"
                )

            v_chk = b.is_check()
            r_chk = ref_is_check(b)
            if v_chk == r_chk:
                res.check_matches += 1
            else:
                res.failures.append(f"near-term {nt_idx} check diff fen={b.fen()}")

            if vname in ("antichess", "short_knight_and_antichess"):
                has_pieces = bool(b.occupied_co[b.turn])
                r_over = (not has_pieces) or (len(r_moves) == 0)
                v_over = b.is_variant_end() or b.is_stalemate()
                if r_over == v_over:
                    res.terminal_matches += 1
                else:
                    res.failures.append(f"near-term {nt_idx} antichess term diff fen={b.fen()}")
            else:
                r_mate = r_chk and len(r_moves) == 0
                r_stale = (not r_chk) and len(r_moves) == 0
                if b.is_checkmate() == r_mate and b.is_stalemate() == r_stale:
                    res.terminal_matches += 1
                else:
                    res.failures.append(f"near-term {nt_idx} term diff fen={b.fen()}")

        # 3. Throughput benchmark (ms/ply over 400 plies)
        bench_plies = 0
        t0 = time.perf_counter()
        while bench_plies < 400:
            b = get_board(vname, seed=bench_plies % 17)
            for _ in range(80):
                moves = list(b.legal_moves)
                if not moves or b.is_game_over():
                    break
                b.push(rng.choice(moves))
                bench_plies += 1
                if bench_plies >= 400:
                    break
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        res.ms_per_ply = elapsed_ms / max(1, bench_plies)

        results[vname] = res

    return results


def format_audit_report(
    diff_results: dict[str, VariantAuditResult],
    ref_results: dict[str, VariantAuditResult],
) -> str:
    """Format a human-readable and Markdown-compatible audit table."""
    lines: list[str] = []
    lines.append("=== Differential Fuzzing vs python-chess (300 positions each) ===")
    lines.append(
        f"{'Variant':<16} {'Pos':>5} {'NearTerm':>8} {'MoveOK':>8} {'ChkOK':>8} {'TermOK':>8} {'Status':>8}"
    )
    for name, r in diff_results.items():
        status = "PASS" if r.passed else f"FAIL({len(r.failures)})"
        lines.append(
            f"{name:<16} {r.positions_checked:>5} {r.near_terminal_checked:>8} "
            f"{r.legal_move_matches:>8} {r.check_matches:>8} {r.terminal_matches:>8} {status:>8}"
        )

    lines.append("")
    lines.append("=== Independent Pure-Python Reference & Property Audit ===")
    lines.append(
        f"{'Variant':<32} {'Pos':>5} {'NT':>4} {'MoveOK':>7} {'ChkOK':>7} "
        f"{'TermOK':>7} {'DictRT':>7} {'SymOK':>7} {'RepRT':>6} {'ms/ply':>7} {'Status':>7}"
    )
    for name, r in ref_results.items():
        status = "PASS" if r.passed else f"FAIL({len(r.failures)})"
        lines.append(
            f"{name:<32} {r.positions_checked:>5} {r.near_terminal_checked:>4} "
            f"{r.legal_move_matches:>7} {r.check_matches:>7} {r.terminal_matches:>7} "
            f"{r.state_dict_roundtrips:>7} {r.symmetry_checks:>7} {r.replay_roundtrips:>6} "
            f"{r.ms_per_ply:>7.3f} {status:>7}"
        )
        for fail in r.failures[:5]:
            lines.append(f"  ! {fail}")

    return "\n".join(lines)


def main() -> int:
    diff_res = run_differential_fuzz(num_positions=300, seed=2026)
    ref_res = run_reference_audit(
        variants=list(ALL_VARIANT_NAMES),
        trajectories_per_variant=20,
        plies_per_trajectory=15,
        near_terminal_per_variant=40,
        seed=2026,
    )
    print(format_audit_report(diff_res, ref_res))
    all_ok = all(r.passed for r in diff_res.values()) and all(r.passed for r in ref_res.values())
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
