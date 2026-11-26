"""Dataset splitting, canonical & near-duplicate deduplication, and leakage/balance auditing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence, Union

import chess
import numpy as np

from chess_icl_v2.data.tokenizer_v2 import ChessV2Tokenizer, STANDARD_STARTING_FEN
from chess_icl_v2.variants.board import VariantBoard, _VARIANT_CLASS_MAP
from chess_icl_v2.variants.variants_impl import get_board

if TYPE_CHECKING:
    from chess_icl_v2.models.qwen_model import QwenPromptFormatter


_STD_PIECE_VALUES: dict[str, int] = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9}


def parse_fen_squares(fen: str) -> tuple[list[str], str]:
    """Parse FEN placement and side-to-move into 64 squares (a1..h8) and 'w'/'b'."""
    parts = fen.strip().split()
    placement = parts[0]
    stm = parts[1].lower() if len(parts) >= 2 and parts[1].lower() in {"w", "b"} else "w"
    ranks = placement.split("/")
    if len(ranks) != 8:
        raise ValueError(f"Invalid FEN placement (expected 8 ranks): {fen!r}")
    squares = ["."] * 64
    for r_idx, rank_str in enumerate(ranks):
        board_rank = 7 - r_idx
        file_idx = 0
        for ch in rank_str:
            if ch.isdigit():
                file_idx += int(ch)
            else:
                if file_idx >= 8:
                    raise ValueError(f"Invalid FEN rank width in {fen!r}")
                squares[board_rank * 8 + file_idx] = ch
                file_idx += 1
        if file_idx != 8:
            raise ValueError(f"FEN rank did not sum to 8 squares in {fen!r}")
    return squares, stm


def compute_standard_material_diff(fen: str) -> int:
    """Compute White minus Black standard material (P=1, N=3, B=3, R=5, Q=9)."""
    placement = fen.strip().split()[0]
    diff = 0
    for ch in placement:
        val = _STD_PIECE_VALUES.get(ch.upper(), 0)
        if val:
            diff += val if ch.isupper() else -val
    return diff


def _apply_uci_on_squares(squares: list[str], stm: str, move_str: str) -> tuple[list[str], str]:
    """Apply a UCI move directly on a 64-square board list without legality checks."""
    new_sq = list(squares)
    from_idx = (int(move_str[1]) - 1) * 8 + (ord(move_str[0]) - ord("a"))
    to_idx = (int(move_str[3]) - 1) * 8 + (ord(move_str[2]) - ord("a"))
    piece = new_sq[from_idx]
    from_file, from_rank = from_idx & 7, from_idx >> 3
    to_file = to_idx & 7
    if piece in ("K", "k") and from_rank == (to_idx >> 3) and abs(to_file - from_file) == 2:
        r_from = from_rank * 8 + (7 if to_file == 6 else 0)
        r_to = from_rank * 8 + (5 if to_file == 6 else 3)
        new_sq[r_to] = new_sq[r_from]
        new_sq[r_from] = "."
    if piece in ("P", "p") and from_file != to_file and new_sq[to_idx] == ".":
        ep_sq = from_rank * 8 + to_file
        if new_sq[ep_sq] in ("P", "p"):
            new_sq[ep_sq] = "."
    if len(move_str) == 5:
        new_sq[to_idx] = move_str[4].upper() if stm == "w" else move_str[4].lower()
    else:
        new_sq[to_idx] = piece
    new_sq[from_idx] = "."
    return new_sq, ("b" if stm == "w" else "w")
from chess_icl_v2.variants.specs import (
    ALL_VARIANT_NAMES,
    COMPOSITIONAL_VARIANT_SPECS,
    SINGLE_VARIANT_SPECS,
)

SQUARE_SYMBOLS: tuple[str, ...] = (
    ".",
    "X",
    "P",
    "N",
    "B",
    "R",
    "Q",
    "K",
    "p",
    "n",
    "b",
    "r",
    "q",
    "k",
)
SQUARE_TO_CODE: dict[str, int] = {sym: idx for idx, sym in enumerate(SQUARE_SYMBOLS)}
SPLIT_PRIORITY: tuple[str, ...] = ("test", "val", "train")

_UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbnkQRBNK]?$")


def _extract_game_id(rec: Mapping[str, Any], fallback_idx: int) -> str:
    for key in ("source_game_id", "game_id", "id"):
        val = rec.get(key)
        if val is not None and str(val) != "":
            return str(val)
    return f"game_{fallback_idx}"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                records.append(json.loads(s))
    return records


def _inject_blocked_into_fen(
    fen: str,
    blocked_squares: Optional[Iterable[Union[int, str]]],
) -> str:
    """Embed blocked squares as 'X' in FEN placement if not already present."""
    if not blocked_squares:
        return fen
    parts = fen.strip().split()
    if "X" in parts[0]:
        return fen
    squares, stm = parse_fen_squares(fen)
    files = "abcdefgh"
    for sq in blocked_squares:
        if isinstance(sq, int):
            idx = sq
        elif isinstance(sq, str) and len(sq) == 2:
            idx = (int(sq[1]) - 1) * 8 + files.index(sq[0].lower())
        else:
            continue
        if 0 <= idx < 64:
            squares[idx] = "X"

    rows: list[str] = []
    for r in range(7, -1, -1):
        empty = 0
        row_parts: list[str] = []
        for f in range(8):
            ch = squares[r * 8 + f]
            if ch == ".":
                empty += 1
            else:
                if empty > 0:
                    row_parts.append(str(empty))
                    empty = 0
                row_parts.append(ch)
        if empty > 0:
            row_parts.append(str(empty))
        rows.append("".join(row_parts))

    rest = " ".join(parts[1:]) if len(parts) > 1 else f"{stm} - - 0 1"
    return f"{'/'.join(rows)} {rest}"


# ---------------------------------------------------------------------------
# 1. Disjoint Game-Pool Splitting
# ---------------------------------------------------------------------------


def split_games_into_pools(
    games: Sequence[dict[str, Any]],
    demo_frac: float = 0.4,
    train_frac: float = 0.3,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Split full games by game_id/source_game_id into strictly disjoint pools.

    Pools: 'demo_games', 'train_source_games', 'val_source_games', 'test_source_games'.
    Guarantees zero overlap of game identifiers across pools.
    """
    fracs = (demo_frac, train_frac, val_frac, test_frac)
    if any(f < 0.0 for f in fracs):
        raise ValueError(f"All split fractions must be non-negative, got {fracs}")
    if abs(sum(fracs) - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {sum(fracs):.6f}")

    by_gid: dict[str, list[dict[str, Any]]] = {}
    ordered_gids: list[str] = []
    for idx, g in enumerate(games):
        gid = _extract_game_id(g, idx)
        if gid not in by_gid:
            by_gid[gid] = []
            ordered_gids.append(gid)
        by_gid[gid].append(g)

    n_games = len(ordered_gids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_games)
    shuffled_gids = [ordered_gids[int(i)] for i in perm]

    n_demo = int(round(n_games * demo_frac))
    n_train = int(round(n_games * train_frac))
    n_val = int(round(n_games * val_frac))
    # Clamp to total and assign remainder to test
    if n_demo + n_train + n_val > n_games:
        overflow = (n_demo + n_train + n_val) - n_games
        for attr in ("val", "train", "demo"):
            if overflow <= 0:
                break
            if attr == "val" and n_val > 0:
                take = min(n_val, overflow)
                n_val -= take
                overflow -= take
            elif attr == "train" and n_train > 0:
                take = min(n_train, overflow)
                n_train -= take
                overflow -= take
            elif attr == "demo" and n_demo > 0:
                take = min(n_demo, overflow)
                n_demo -= take
                overflow -= take
    n_test = n_games - (n_demo + n_train + n_val)

    c1 = n_demo
    c2 = c1 + n_train
    c3 = c2 + n_val

    demo_gids = shuffled_gids[:c1]
    train_gids = shuffled_gids[c1:c2]
    val_gids = shuffled_gids[c2:c3]
    test_gids = shuffled_gids[c3:]

    def _collect(gids: Sequence[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for gid in gids:
            out.extend(by_gid[gid])
        return out

    return {
        "demo_games": _collect(demo_gids),
        "train_source_games": _collect(train_gids),
        "val_source_games": _collect(val_gids),
        "test_source_games": _collect(test_gids),
    }


# ---------------------------------------------------------------------------
# 2. Compact Board Signature & Canonical Key Helpers
# ---------------------------------------------------------------------------


def _normalize_state_dict(d: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize aliases in a state/position dict for VariantBoard.from_state_dict."""
    out = dict(d)
    if "variant" not in out:
        out["variant"] = "standard"
    if "fen" not in out:
        out["fen"] = out.get("query_fen") or out.get("state") or STANDARD_STARTING_FEN
    if "blocked" not in out and "blocked_squares" in out and out["blocked_squares"] is not None:
        out["blocked"] = out["blocked_squares"]
    return out


def board_similarity_signature(
    state_dict_or_fen: Union[VariantBoard, Mapping[str, Any], str],
    blocked_squares: Optional[Iterable[Union[int, str]]] = None,
) -> tuple[np.ndarray, int]:
    """Produce compact (int8[64], turn_bit) signature for fast vectorized Hamming distance.

    Uses `VariantBoard.board_matrix_64()` + `turn` when reconstructible, with direct
    FEN parsing fallback when `'X'` barriers are already embedded in FEN strings.
    """
    if isinstance(state_dict_or_fen, VariantBoard):
        mat = state_dict_or_fen.board_matrix_64()
        arr = np.array([SQUARE_TO_CODE.get(ch, 0) for ch in mat], dtype=np.int8)
        turn_bit = 1 if state_dict_or_fen.turn == chess.WHITE else 0
        return arr, turn_bit

    if isinstance(state_dict_or_fen, Mapping):
        norm = _normalize_state_dict(state_dict_or_fen)
        fen = str(norm["fen"])
        blk = blocked_squares if blocked_squares is not None else norm.get("blocked")
        if "X" not in fen.split()[0] and norm.get("variant") in _VARIANT_CLASS_MAP:
            try:
                vb = VariantBoard.from_state_dict(norm)
                if blk:
                    for sq in blk:
                        sq_i = sq if isinstance(sq, int) else chess.parse_square(str(sq).lower())
                        vb.blocked.add(sq_i)
                return board_similarity_signature(vb)
            except Exception:
                pass
        fen_with_x = _inject_blocked_into_fen(fen, blk)
        squares, stm = parse_fen_squares(fen_with_x)
        arr = np.array([SQUARE_TO_CODE.get(ch, 0) for ch in squares], dtype=np.int8)
        return arr, (1 if stm == "w" else 0)

    fen_str = _inject_blocked_into_fen(str(state_dict_or_fen), blocked_squares)
    if "X" not in fen_str.split()[0]:
        try:
            vb = VariantBoard(fen_str)
            return board_similarity_signature(vb)
        except Exception:
            pass
    squares, stm = parse_fen_squares(fen_str)
    arr = np.array([SQUARE_TO_CODE.get(ch, 0) for ch in squares], dtype=np.int8)
    return arr, (1 if stm == "w" else 0)


def compute_canonical_state_key(record: Mapping[str, Any]) -> str:
    """Compute or retrieve a record's `canonical_state_key`."""
    if record.get("canonical_state_key"):
        return str(record["canonical_state_key"])
    if record.get("canonical_key"):
        return str(record["canonical_key"])

    norm = _normalize_state_dict(record)
    fen = str(norm["fen"])
    variant = str(norm["variant"])
    blk = norm.get("blocked")
    if "X" not in fen.split()[0] and variant in _VARIANT_CLASS_MAP:
        try:
            vb = VariantBoard.from_state_dict(norm)
            if blk:
                for sq in blk:
                    sq_i = sq if isinstance(sq, int) else chess.parse_square(str(sq).lower())
                    vb.blocked.add(sq_i)
            return vb.canonical_state_key()
        except Exception:
            pass

    fen_x = _inject_blocked_into_fen(fen, blk)
    parts = fen_x.strip().split()
    placement = parts[0]
    turn = parts[1].lower() if len(parts) > 1 else "w"
    castling = parts[2] if len(parts) > 2 else "-"
    ep = parts[3] if len(parts) > 3 else "-"
    return f"{variant}:{placement} {turn} {castling} {ep}"


# ---------------------------------------------------------------------------
# 3. Canonical + Near-Duplicate Deduplication
# ---------------------------------------------------------------------------


def _extract_demo_signatures(
    demo_games: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (M, 64) int8 board matrix and (M,) turn vector from all states in `demo_games`."""
    boards: list[np.ndarray] = []
    turns: list[int] = []

    for demo in demo_games:
        blk = demo.get("blocked_squares") if demo.get("blocked_squares") is not None else demo.get("blocked")
        pre_states = demo.get("state_dicts") or demo.get("states") or demo.get("fens")
        if pre_states:
            for st in pre_states:
                b_arr, t_bit = board_similarity_signature(st, blocked_squares=blk)
                boards.append(b_arr)
                turns.append(t_bit)
            continue

        init_fen = str(
            demo.get("initial_fen")
            or demo.get("init_fen")
            or demo.get("initial_state")
            or demo.get("start_fen")
            or demo.get("fen")
            or STANDARD_STARTING_FEN
        )
        init_fen_x = _inject_blocked_into_fen(init_fen, blk)
        squares, stm = parse_fen_squares(init_fen_x)
        boards.append(np.array([SQUARE_TO_CODE.get(ch, 0) for ch in squares], dtype=np.int8))
        turns.append(1 if stm == "w" else 0)

        moves = demo.get("moves") or demo.get("uci_moves") or []
        if isinstance(moves, str):
            moves = [m for m in moves.strip().split() if m]

        std_board: Optional[chess.Board] = None
        if "X" not in init_fen_x.split()[0]:
            try:
                std_board = chess.Board(init_fen_x)
            except ValueError:
                std_board = None

        for mv in moves:
            mv_s = str(mv)
            if _UCI_RE.match(mv_s):
                squares, stm = _apply_uci_on_squares(squares, stm, mv_s)
                if std_board is not None:
                    try:
                        std_board.push(chess.Move.from_uci(mv_s.lower()))
                    except Exception:
                        std_board = None
            elif std_board is not None:
                try:
                    parsed = std_board.parse_san(mv_s)
                    uci = parsed.uci()
                    std_board.push(parsed)
                    squares, stm = _apply_uci_on_squares(squares, stm, uci)
                except Exception:
                    stm = "b" if stm == "w" else "w"
            else:
                stm = "b" if stm == "w" else "w"

            boards.append(np.array([SQUARE_TO_CODE.get(ch, 0) for ch in squares], dtype=np.int8))
            turns.append(1 if stm == "w" else 0)

    if not boards:
        return np.empty((0, 64), dtype=np.int8), np.empty((0,), dtype=np.int8)
    return np.vstack(boards), np.asarray(turns, dtype=np.int8)


def _has_near_duplicate(
    q_board: np.ndarray,
    q_turn: int,
    bank_boards: np.ndarray,
    bank_turns: np.ndarray,
    max_similarity: float,
) -> bool:
    """Return True if any bank state with same turn has square similarity >= max_similarity."""
    if len(bank_boards) == 0:
        return False
    turn_mask = bank_turns == q_turn
    if not np.any(turn_mask):
        return False
    matches = np.sum(bank_boards[turn_mask] == q_board, axis=1)
    # Use strict threshold comparison: > max_similarity, or >= max_similarity when exact
    # Note: for 64 squares and max_similarity=0.90, 57/64=0.8906 < 0.90 < 58/64=0.90625.
    sims = matches.astype(np.float64) / 64.0
    return bool(np.any(sims >= max_similarity))


def deduplicate_and_filter_roots(
    roots_by_split: Mapping[str, Sequence[dict[str, Any]]],
    demo_games: Sequence[Mapping[str, Any]],
    max_similarity: float = 0.90,
    min_ply: int = 8,
) -> dict[str, Any]:
    """Deduplicate roots across splits and filter near-duplicates vs demos & higher-priority splits.

    Priority order: test > val > train (then any additional split names sorted).
    Steps:
      0. Exclude roots explicitly annotated with ply < min_ply (if ply/move_number is present).
      1. Exact deduplication via `canonical_state_key` across all splits.
      2. Near-duplicate removal (>= `max_similarity` square identity with matching turn):
         (a) against all states in `demo_games`, and
         (b) against already-accepted roots in higher-priority splits.
    """
    ordered_splits = [s for s in SPLIT_PRIORITY if s in roots_by_split] + sorted(
        s for s in roots_by_split if s not in SPLIT_PRIORITY
    )

    demo_boards, demo_turns = _extract_demo_signatures(demo_games)
    seen_canonical_keys: set[str] = set()

    higher_boards_list: list[np.ndarray] = []
    higher_turns_list: list[int] = []

    cleaned: dict[str, list[dict[str, Any]]] = {}
    input_counts: dict[str, int] = {}
    kept_counts: dict[str, int] = {}

    opening_ply_removed = 0
    exact_removed = 0
    neardup_vs_demos_removed = 0
    neardup_cross_split_removed = 0

    for split_name in ordered_splits:
        raw_roots = list(roots_by_split[split_name])
        input_counts[split_name] = len(raw_roots)

        higher_boards = (
            np.vstack(higher_boards_list)
            if higher_boards_list
            else np.empty((0, 64), dtype=np.int8)
        )
        higher_turns = (
            np.asarray(higher_turns_list, dtype=np.int8)
            if higher_turns_list
            else np.empty((0,), dtype=np.int8)
        )

        kept_for_split: list[dict[str, Any]] = []
        split_boards: list[np.ndarray] = []
        split_turns: list[int] = []

        for root in raw_roots:
            ply_val = None
            for p_key in ("ply", "move_number", "halfmove"):
                if p_key in root and root[p_key] is not None:
                    ply_val = int(root[p_key])
                    break
            if ply_val is not None and ply_val < min_ply:
                opening_ply_removed += 1
                continue

            ckey = compute_canonical_state_key(root)
            if ckey in seen_canonical_keys:
                exact_removed += 1
                continue

            q_board, q_turn = board_similarity_signature(root)

            if _has_near_duplicate(q_board, q_turn, demo_boards, demo_turns, max_similarity):
                neardup_vs_demos_removed += 1
                continue

            if _has_near_duplicate(q_board, q_turn, higher_boards, higher_turns, max_similarity):
                neardup_cross_split_removed += 1
                continue

            seen_canonical_keys.add(ckey)
            kept_for_split.append(root)
            split_boards.append(q_board)
            split_turns.append(q_turn)

        cleaned[split_name] = kept_for_split
        kept_counts[split_name] = len(kept_for_split)
        higher_boards_list.extend(split_boards)
        higher_turns_list.extend(split_turns)

    return {
        "roots_by_split": cleaned,
        "dedup_stats": {
            "input_counts": input_counts,
            "kept_counts": kept_counts,
            "opening_ply_removed": int(opening_ply_removed),
            "exact_removed": int(exact_removed),
            "neardup_vs_demos_removed": int(neardup_vs_demos_removed),
            "neardup_cross_split_removed": int(neardup_cross_split_removed),
            "max_similarity": float(max_similarity),
            "min_ply": int(min_ply),
        },
    }


# ---------------------------------------------------------------------------
# 4. Leakage & Balance Auditing
# ---------------------------------------------------------------------------


def _build_forbidden_terms() -> list[tuple[str, re.Pattern[str]]]:
    """Build case-insensitive regex patterns for variant names and rule descriptions."""
    terms: set[str] = set()
    for name in ALL_VARIANT_NAMES:
        if name == "standard":
            continue
        terms.add(name)
        terms.add(name.replace("_", " "))

    # Also include distinctive rule-description phrases from specs
    all_specs = list(SINGLE_VARIANT_SPECS.values()) + list(COMPOSITIONAL_VARIANT_SPECS.values())
    for spec in all_specs:
        if spec.description:
            terms.add(spec.description.strip())

    patterns: list[tuple[str, re.Pattern[str]]] = []
    for term in sorted(terms):
        # Word-boundary match for identifiers/phrases
        pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(term) + r"(?![A-Za-z0-9_])", re.IGNORECASE)
        patterns.append((term, pat))
    return patterns


_FORBIDDEN_PATTERNS = _build_forbidden_terms()


def check_prompt_leakage(
    prompt_text: str,
    query_record: Optional[Mapping[str, Any]] = None,
    extra_forbidden_terms: Optional[Sequence[str]] = None,
) -> list[str]:
    """Inspect formatted Qwen prompt text for variant names, rule descriptions, future moves, or query IDs."""
    findings: list[str] = []

    for term, pat in _FORBIDDEN_PATTERNS:
        if pat.search(prompt_text):
            findings.append(f"Leaked variant/rule term {term!r} in prompt text")

    if extra_forbidden_terms:
        for term in extra_forbidden_terms:
            if term and str(term).lower() in prompt_text.lower():
                findings.append(f"Leaked forbidden term {term!r} in prompt text")

    if query_record is not None:
        # Check that future trajectory / continuation moves are absent
        for fut_key in ("future_moves", "continuation", "rollout_moves"):
            fut = query_record.get(fut_key)
            if isinstance(fut, (list, tuple)) and len(fut) >= 2:
                seq_str = " ".join(str(m) for m in fut)
                if seq_str in prompt_text:
                    findings.append(f"Leaked {fut_key} sequence in prompt text")

        # Check query identifiers (if non-trivial length >= 4)
        for id_key in ("root_id", "source_game_id", "game_id"):
            val = query_record.get(id_key)
            if val is not None and len(str(val)) >= 4 and str(val) in prompt_text:
                findings.append(f"Leaked query identifier {id_key}={val!r} in prompt text")

        # Check explicit query label keys don't leak into [Query] section
        if "[Query]" in prompt_text:
            query_section = prompt_text.split("[Query]", 1)[1]
            for lbl_key in ("y_true", "target_probs", "outcome_probs", "source_game_id"):
                if lbl_key in query_section:
                    findings.append(f"Leaked field name {lbl_key!r} in [Query] section")

    return findings


def _check_v2_tokens_leakage(
    encoded: Mapping[str, Any],
    tokenizer: ChessV2Tokenizer,
) -> list[str]:
    """Verify ChessV2Tokenizer query segment contains only [QUERY, STATE, <70 tokens>, SEP]."""
    findings: list[str] = []
    input_ids: list[int] = list(encoded["input_ids"])
    q_start = int(encoded["query_start_idx"])
    query_slice = input_ids[q_start:]

    if len(query_slice) != 73:  # [QUERY] + [STATE] + 70 state tokens + [SEP]
        findings.append(f"Unexpected query token slice length {len(query_slice)} (expected 73)")
        return findings

    if query_slice[0] != tokenizer.query_id or query_slice[1] != tokenizer.state_id or query_slice[-1] != tokenizer.sep_id:
        findings.append("Query token segment missing expected [QUERY] [STATE] ... [SEP] framing")

    forbidden_in_query = {
        tokenizer.out_w_id,
        tokenizer.out_d_id,
        tokenizer.out_l_id,
        tokenizer.moves_id,
        tokenizer.demo_start_id,
        tokenizer.demo_end_id,
    }
    if any(tid in forbidden_in_query for tid in query_slice):
        findings.append("Outcome or move token leaked into [QUERY] token segment")

    return findings


def _target_vector(pos: Mapping[str, Any]) -> np.ndarray:
    for key in ("y_true", "target", "target_probs", "outcome_probs", "probs"):
        val = pos.get(key)
        if val is not None:
            arr = np.asarray(val, dtype=np.float64)
            if arr.shape == (3,):
                s = float(np.sum(arr))
                return arr / s if s > 0 else np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    if "win_prob" in pos and "draw_prob" in pos and "loss_prob" in pos:
        arr = np.array([pos["win_prob"], pos["draw_prob"], pos["loss_prob"]], dtype=np.float64)
        s = float(np.sum(arr))
        return arr / s if s > 0 else np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    lbl = str(pos.get("label", "draw")).strip().lower()
    if lbl in ("win", "w"):
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if lbl in ("loss", "l"):
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return np.array([0.0, 1.0, 0.0], dtype=np.float64)


def _position_balance_stats(positions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(positions)
    if n == 0:
        return {
            "n_positions": 0,
            "white_frac": None,
            "black_frac": None,
            "equal_material_frac": None,
            "target_mean": None,
            "target_std": None,
        }

    white_count = 0
    eq_mat_count = 0
    targets = np.empty((n, 3), dtype=np.float64)

    for i, pos in enumerate(positions):
        fen = str(pos.get("query_fen") or pos.get("fen") or pos.get("state") or STANDARD_STARTING_FEN)
        parts = fen.split()
        stm = pos.get("side_to_move") or (parts[1].lower() if len(parts) >= 2 else "w")
        if str(stm).lower().startswith("w"):
            white_count += 1
        if abs(compute_standard_material_diff(fen)) <= 2:
            eq_mat_count += 1
        targets[i] = _target_vector(pos)

    white_frac = float(white_count / n)
    return {
        "n_positions": int(n),
        "white_frac": white_frac,
        "black_frac": float(1.0 - white_frac),
        "equal_material_frac": float(eq_mat_count / n),
        "target_mean": [float(v) for v in np.mean(targets, axis=0)],
        "target_std": [float(v) for v in np.std(targets, axis=0)],
    }


# ---------------------------------------------------------------------------
# 5. Long-Study Independent Pools, Adjacent Family Sampling & Radius-2 Audit
# ---------------------------------------------------------------------------

DEFAULT_PILOT_PATHS: tuple[Path, ...] = (
    Path("data"),
    Path("data_large"),
    Path("data_v2_tiny"),
    Path("runs_v2/stage0/sample_games.jsonl"),
)

_SQ_NAME_TO_IDX: dict[str, int] = {
    f"{'abcdefgh'[f]}{r + 1}": r * 8 + f for r in range(8) for f in range(8)
}
_CODE_EMPTY = SQUARE_TO_CODE["."]
_CODE_X = SQUARE_TO_CODE["X"]
_CODE_P = SQUARE_TO_CODE["P"]
_CODE_p = SQUARE_TO_CODE["p"]
_CODE_K = SQUARE_TO_CODE["K"]
_CODE_k = SQUARE_TO_CODE["k"]
_WHITE_PROMO_CODES = {
    "q": SQUARE_TO_CODE["Q"],
    "r": SQUARE_TO_CODE["R"],
    "b": SQUARE_TO_CODE["B"],
    "n": SQUARE_TO_CODE["N"],
    "k": SQUARE_TO_CODE["K"],
}
_BLACK_PROMO_CODES = {
    "q": SQUARE_TO_CODE["q"],
    "r": SQUARE_TO_CODE["r"],
    "b": SQUARE_TO_CODE["b"],
    "n": SQUARE_TO_CODE["n"],
    "k": SQUARE_TO_CODE["k"],
}
_WHITE_CODES = {SQUARE_TO_CODE[c] for c in ("P", "N", "B", "R", "Q", "K")}
_BLACK_CODES = {SQUARE_TO_CODE[c] for c in ("p", "n", "b", "r", "q", "k")}


def _extract_family_id(rec: Mapping[str, Any], fallback_idx: int = 0) -> str:
    for key in ("family_id", "source_game_id", "game_id", "id"):
        val = rec.get(key)
        if val is not None and str(val) != "":
            return str(val)
    return f"family_{fallback_idx}"


def _convert_piece_code_color(code: int, to_white: bool) -> int:
    """Convert a piece code (P..K or p..k) to White or Black while preserving piece type."""
    if to_white and code in _BLACK_CODES:
        return code - 6
    if (not to_white) and code in _WHITE_CODES:
        return code + 6
    return code


def split_games_into_independent_pools(
    games: Sequence[dict[str, Any]],
    demo_train_count: int,
    demo_val_count: int,
    demo_test_count: int,
    train_source_count: int,
    val_source_count: int,
    test_source_count: int,
    seed: int = 42,
    require_terminal_demos: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Split full games by family/game ID into 6 strictly disjoint pools.

    Pools:
      - ``demo_train_games``, ``demo_val_games``, ``demo_test_games``
      - ``train_source_games``, ``val_source_games``, ``test_source_games``

    When ``require_terminal_demos=True``, only games with ``is_terminal=True``
    (or ``hit_horizon=False``) are eligible for the demonstration pools.
    """
    counts = (
        demo_train_count,
        demo_val_count,
        demo_test_count,
        train_source_count,
        val_source_count,
        test_source_count,
    )
    if any(c < 0 for c in counts):
        raise ValueError(f"Pool counts must be non-negative, got {counts}")

    by_fid: dict[str, list[dict[str, Any]]] = {}
    ordered_fids: list[str] = []
    for idx, g in enumerate(games):
        fid = _extract_family_id(g, idx)
        if fid not in by_fid:
            by_fid[fid] = []
            ordered_fids.append(fid)
        by_fid[fid].append(g)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ordered_fids))
    shuffled_fids = [ordered_fids[int(i)] for i in perm]

    def _is_terminal_family(fid: str) -> bool:
        return all(
            bool(g.get("is_terminal", not bool(g.get("hit_horizon", False))))
            for g in by_fid[fid]
        )

    used: set[str] = set()
    demo_pools: dict[str, list[dict[str, Any]]] = {
        "demo_train_games": [],
        "demo_val_games": [],
        "demo_test_games": [],
    }
    demo_targets = (
        ("demo_train_games", demo_train_count),
        ("demo_val_games", demo_val_count),
        ("demo_test_games", demo_test_count),
    )
    for pool_name, target in demo_targets:
        taken = 0
        for fid in shuffled_fids:
            if taken >= target:
                break
            if fid in used:
                continue
            if require_terminal_demos and not _is_terminal_family(fid):
                continue
            used.add(fid)
            demo_pools[pool_name].extend(by_fid[fid])
            taken += 1

    source_pools: dict[str, list[dict[str, Any]]] = {
        "train_source_games": [],
        "val_source_games": [],
        "test_source_games": [],
    }
    source_targets = (
        ("train_source_games", train_source_count),
        ("val_source_games", val_source_count),
        ("test_source_games", test_source_count),
    )
    for pool_name, target in source_targets:
        taken = 0
        for fid in shuffled_fids:
            if taken >= target:
                break
            if fid in used:
                continue
            used.add(fid)
            source_pools[pool_name].extend(by_fid[fid])
            taken += 1

    return {**demo_pools, **source_pools}


def _make_root_candidate(
    variant_name: str,
    family_id: str,
    ply: int,
    sd: Mapping[str, Any],
    pair_plies: Sequence[int],
) -> dict[str, Any]:
    board = VariantBoard.from_state_dict(sd)
    return {
        "variant": variant_name,
        "family_id": family_id,
        "source_game_id": family_id,
        "game_id": family_id,
        "ply": int(ply),
        "pair_plies": [int(p) for p in pair_plies],
        "fen": sd["fen"],
        "blocked": list(sd.get("blocked", [])),
        "turn": bool(sd["turn"]),
        "state_dict": dict(sd),
        "canonical_state_key": board.canonical_state_key(),
    }


def sample_candidate_pairs_by_family(
    source_games: Sequence[Mapping[str, Any]],
    variant_name: str,
    seed: int = 42,
    min_ply: int = 8,
) -> list[tuple[str, list[tuple[dict[str, Any], ...]]]]:
    """Extract ordered candidate adjacent root pairs `(ply, ply+1)` for each family.

    Pairs are prioritized so middlegame/endgame plies (`>= 24`) precede early plies
    (`min_ply..23`), preserving opposite-turn pairing within every family.
    """
    rng = np.random.default_rng(seed)
    games_list = list(source_games)
    if len(games_list) > 1:
        perm = rng.permutation(len(games_list))
        games_list = [games_list[int(i)] for i in perm]

    out: list[tuple[str, list[tuple[dict[str, Any], ...]]]] = []
    seen_fids: set[str] = set()

    for idx, game in enumerate(games_list):
        fid = _extract_family_id(game, idx)
        if fid in seen_fids:
            continue
        seen_fids.add(fid)

        valid_by_ply: dict[int, Mapping[str, Any]] = {}
        for ply, sd in game.get("states_at_plies", []):
            p_int = int(ply)
            if p_int < min_ply:
                continue
            board = VariantBoard.from_state_dict(sd)
            if board.is_game_over(claim_draw=True):
                continue
            valid_by_ply[p_int] = sd

        if not valid_by_ply:
            continue

        mid_late_pairs: list[tuple[int, int]] = []
        early_pairs: list[tuple[int, int]] = []
        for p_int in sorted(valid_by_ply):
            if (p_int + 1) in valid_by_ply:
                if p_int >= 24:
                    mid_late_pairs.append((p_int, p_int + 1))
                else:
                    early_pairs.append((p_int, p_int + 1))

        def _shuffle_pairs(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
            if len(pairs) <= 1:
                return pairs
            p_idx = rng.permutation(len(pairs))
            return [pairs[int(i)] for i in p_idx]

        ordered_pairs = _shuffle_pairs(mid_late_pairs) + _shuffle_pairs(early_pairs)
        cand_tuples: list[tuple[dict[str, Any], ...]] = []
        for p_a, p_b in ordered_pairs:
            r_a = _make_root_candidate(variant_name, fid, p_a, valid_by_ply[p_a], (p_a, p_b))
            r_b = _make_root_candidate(variant_name, fid, p_b, valid_by_ply[p_b], (p_a, p_b))
            cand_tuples.append((r_a, r_b))

        if not cand_tuples:
            # Fallback to single valid state if no adjacent pair exists
            plies = sorted(valid_by_ply)
            mid_plies = [p for p in plies if p >= 24] or plies
            chosen_p = mid_plies[int(rng.integers(len(mid_plies)))]
            cand_tuples.append(
                (_make_root_candidate(variant_name, fid, chosen_p, valid_by_ply[chosen_p], (chosen_p,)),)
            )

        out.append((fid, cand_tuples))
    return out


def sample_adjacent_roots_per_family(
    source_games: Sequence[Mapping[str, Any]],
    variant_name: str,
    max_families: Optional[int] = None,
    seed: int = 42,
    min_ply: int = 8,
    max_roots_per_family: int = 2,
) -> list[dict[str, Any]]:
    """Sample at most `max_roots_per_family` (default 2) adjacent states per family.

    Adjacent states `(ply, ply + 1)` share `family_id`/`source_game_id` and naturally
    carry opposite sides to move (`w` and `b`), ensuring White/Black balance.
    """
    if max_roots_per_family < 1 or max_roots_per_family > 2:
        raise ValueError(f"max_roots_per_family must be 1 or 2, got {max_roots_per_family}")

    family_cands = sample_candidate_pairs_by_family(
        source_games=source_games,
        variant_name=variant_name,
        seed=seed,
        min_ply=min_ply,
    )
    if max_families is not None:
        family_cands = family_cands[: max(0, int(max_families))]

    selected: list[dict[str, Any]] = []
    singleton_prefer_white = True
    for _, pairs in family_cands:
        primary = pairs[0]
        if max_roots_per_family == 2:
            selected.extend(primary[:2])
        else:
            if len(primary) == 2:
                pick = primary[0] if (bool(primary[0]["turn"]) == singleton_prefer_white) else primary[1]
                singleton_prefer_white = not singleton_prefer_white
                selected.append(pick)
            else:
                selected.append(primary[0])
    return selected


class RadiusTwoIndex:
    """Exact canonical-key and 3-chunk pigeonhole Hamming<=2 board index.

    Partitioning the 64-square board into 3 disjoint slices (0:21, 21:42, 42:64)
    guarantees by the pigeonhole principle that any board within Hamming distance
    <= 2 of a query shares at least one exact 21/22-square slice with matching turn.
    """

    def __init__(self, deduplicate_exact_boards: bool = True) -> None:
        self.deduplicate_exact_boards = deduplicate_exact_boards
        self.canonical_keys: dict[str, set[str]] = {}
        self.boards: list[bytes] = []
        self.turns: list[int] = []
        self.family_ids: list[Optional[str]] = []
        self.metadata: list[Any] = []
        self._seen_board_turn: set[tuple[int, bytes]] = set()
        self._idx0: dict[tuple[int, bytes], list[int]] = {}
        self._idx1: dict[tuple[int, bytes], list[int]] = {}
        self._idx2: dict[tuple[int, bytes], list[int]] = {}

    def __len__(self) -> int:
        return len(self.boards)

    def add_bytes(
        self,
        raw: bytes,
        turn_bit: int,
        canonical_key: Optional[str] = None,
        family_id: Optional[str] = None,
        meta: Any = None,
    ) -> None:
        t = int(turn_bit)
        if canonical_key:
            fids = self.canonical_keys.setdefault(canonical_key, set())
            if family_id is not None:
                fids.add(family_id)

        if self.deduplicate_exact_boards:
            key = (t, raw)
            if key in self._seen_board_turn:
                return
            self._seen_board_turn.add(key)

        row_id = len(self.boards)
        self.boards.append(raw)
        self.turns.append(t)
        self.family_ids.append(family_id)
        self.metadata.append(meta)

        self._idx0.setdefault((t, raw[:21]), []).append(row_id)
        self._idx1.setdefault((t, raw[21:42]), []).append(row_id)
        self._idx2.setdefault((t, raw[42:]), []).append(row_id)

    def add_signature(
        self,
        arr: np.ndarray,
        turn_bit: int,
        canonical_key: Optional[str] = None,
        family_id: Optional[str] = None,
        meta: Any = None,
    ) -> None:
        self.add_bytes(
            raw=arr.tobytes(),
            turn_bit=turn_bit,
            canonical_key=canonical_key,
            family_id=family_id,
            meta=meta,
        )

    def find_collision(
        self,
        arr: np.ndarray,
        turn_bit: int,
        canonical_key: Optional[str] = None,
        max_hamming: int = 2,
        ignore_family_id: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        if canonical_key and canonical_key in self.canonical_keys:
            fids = self.canonical_keys[canonical_key]
            if ignore_family_id is None or not fids or any(f != ignore_family_id for f in fids):
                return {
                    "reason": "canonical",
                    "hamming_distance": 0,
                    "canonical_state_key": canonical_key,
                }

        t = int(turn_bit)
        raw = arr.tobytes()
        candidates: set[int] = set()
        candidates.update(self._idx0.get((t, raw[:21]), ()))
        candidates.update(self._idx1.get((t, raw[21:42]), ()))
        candidates.update(self._idx2.get((t, raw[42:]), ()))

        for rid in candidates:
            if ignore_family_id is not None and self.family_ids[rid] == ignore_family_id:
                continue
            cand_arr = np.frombuffer(self.boards[rid], dtype=np.int8)
            dist = int(np.count_nonzero(cand_arr != arr))
            if dist <= max_hamming:
                return {
                    "reason": "radius_two",
                    "hamming_distance": dist,
                    "meta": self.metadata[rid],
                    "matched_family_id": self.family_ids[rid],
                }
        return None


def _fast_extract_game_states(
    rec: Mapping[str, Any],
    variant_hint: Optional[str] = None,
) -> list[tuple[int, bytes]]:
    """Fast bytearray(64) state trajectory extractor for games in pilot/demo JSONLs."""
    variant = str(rec.get("variant") or variant_hint or "standard")
    blk = rec.get("blocked_squares") if rec.get("blocked_squares") is not None else rec.get("blocked")

    pre_states = rec.get("state_dicts") or rec.get("states") or rec.get("fens")
    if pre_states:
        out: list[tuple[int, bytes]] = []
        for st in pre_states:
            b_arr, t_bit = board_similarity_signature(st, blocked_squares=blk)
            out.append((t_bit, b_arr.tobytes()))
        return out

    init_fen = (
        rec.get("initial_fen")
        or rec.get("init_fen")
        or rec.get("initial_state")
        or rec.get("start_fen")
        or rec.get("fen")
    )
    if not init_fen:
        if variant == "knight_army":
            init_fen = get_board("knight_army").fen()
        else:
            init_fen = STANDARD_STARTING_FEN

    init_fen_x = _inject_blocked_into_fen(str(init_fen), blk)
    squares, stm = parse_fen_squares(init_fen_x)
    buf = bytearray(SQUARE_TO_CODE.get(ch, _CODE_EMPTY) for ch in squares)
    turn_bit = 1 if stm == "w" else 0
    states: list[tuple[int, bytes]] = [(turn_bit, bytes(buf))]

    moves = rec.get("moves") or rec.get("uci_moves") or []
    if isinstance(moves, str):
        moves = [m for m in moves.strip().split() if m]

    is_piece_capture = "piece_capture" in variant
    std_board: Optional[chess.Board] = None
    has_non_uci = any(not _UCI_RE.match(str(mv)) for mv in moves)
    if has_non_uci and "X" not in init_fen_x.split()[0]:
        try:
            std_board = chess.Board(init_fen_x)
        except ValueError:
            std_board = None

    sq_map = _SQ_NAME_TO_IDX
    for mv in moves:
        mv_s = str(mv)
        if len(mv_s) >= 4 and mv_s[:2] in sq_map and mv_s[2:4] in sq_map and _UCI_RE.match(mv_s):
            f_idx = sq_map[mv_s[:2]]
            t_idx = sq_map[mv_s[2:4]]
            piece = buf[f_idx]
            from_file, from_rank = f_idx & 7, f_idx >> 3
            to_file, to_rank = t_idx & 7, t_idx >> 3

            if (piece == _CODE_K or piece == _CODE_k) and from_rank == to_rank and abs(to_file - from_file) == 2:
                if to_file == 6:
                    r_from, r_to = from_rank * 8 + 7, from_rank * 8 + 5
                else:
                    r_from, r_to = from_rank * 8, from_rank * 8 + 3
                buf[r_to] = buf[r_from]
                buf[r_from] = _CODE_EMPTY

            ep_captured = False
            if (piece == _CODE_P or piece == _CODE_p) and from_file != to_file and buf[t_idx] == _CODE_EMPTY:
                ep_sq = from_rank * 8 + to_file
                if buf[ep_sq] == _CODE_P or buf[ep_sq] == _CODE_p:
                    ep_captured = True
                    if is_piece_capture:
                        buf[t_idx] = _CODE_P if turn_bit == 1 else _CODE_p
                    buf[ep_sq] = _CODE_EMPTY

            if len(mv_s) == 5:
                promo_map = _WHITE_PROMO_CODES if turn_bit == 1 else _BLACK_PROMO_CODES
                buf[t_idx] = promo_map.get(mv_s[4].lower(), piece)
            elif not ep_captured:
                victim = buf[t_idx]
                if (
                    is_piece_capture
                    and victim != _CODE_EMPTY
                    and victim != _CODE_X
                    and victim != _CODE_K
                    and victim != _CODE_k
                    and piece != _CODE_K
                    and piece != _CODE_k
                ):
                    buf[t_idx] = _convert_piece_code_color(victim, to_white=(turn_bit == 1))
                else:
                    buf[t_idx] = piece

            buf[f_idx] = _CODE_EMPTY
            if std_board is not None:
                try:
                    std_board.push(chess.Move.from_uci(mv_s.lower()))
                except Exception:
                    std_board = None
        elif std_board is not None:
            try:
                parsed = std_board.parse_san(mv_s)
                uci = parsed.uci()
                std_board.push(parsed)
                f_idx = sq_map[uci[:2]]
                t_idx = sq_map[uci[2:4]]
                sq_strs = [SQUARE_SYMBOLS[c] for c in buf]
                new_sq, _ = _apply_uci_on_squares(sq_strs, "w" if turn_bit == 1 else "b", uci)
                buf = bytearray(SQUARE_TO_CODE.get(ch, _CODE_EMPTY) for ch in new_sq)
            except Exception:
                pass

        turn_bit ^= 1
        states.append((turn_bit, bytes(buf)))

    return states


def _add_games_to_index(
    index: RadiusTwoIndex,
    games: Sequence[Mapping[str, Any]],
    source_tag: str,
) -> int:
    added = 0
    for g in games:
        for t_bit, b_bytes in _fast_extract_game_states(g):
            index.add_bytes(b_bytes, t_bit, meta=source_tag)
            added += 1
    return added


def build_pilot_index(
    pilot_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> tuple[RadiusTwoIndex, dict[str, Any]]:
    """Load canonical keys & 64-square signatures from prior pilot JSONL files."""
    paths = list(DEFAULT_PILOT_PATHS if pilot_paths is None else pilot_paths)
    index = RadiusTwoIndex(deduplicate_exact_boards=True)

    jsonl_files: list[Path] = []
    checked_paths: list[str] = []
    missing_paths: list[str] = []

    for p in paths:
        path_obj = Path(p)
        checked_paths.append(str(path_obj))
        if not path_obj.exists():
            missing_paths.append(str(path_obj))
            continue
        if path_obj.is_file():
            if path_obj.suffix == ".jsonl":
                jsonl_files.append(path_obj)
        else:
            jsonl_files.extend(sorted(path_obj.rglob("*.jsonl")))

    records_scanned = 0
    for jf in jsonl_files:
        rel_str = str(jf)
        with jf.open("r", encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                rec = json.loads(s)
                records_scanned += 1
                if "moves" in rec or "uci_moves" in rec or "state_dicts" in rec:
                    for t_bit, b_bytes in _fast_extract_game_states(rec):
                        index.add_bytes(b_bytes, t_bit, meta=rel_str)
                else:
                    fen_val = rec.get("query_fen") or rec.get("fen") or rec.get("state")
                    if fen_val is None:
                        continue
                    blk = rec.get("blocked_squares") if rec.get("blocked_squares") is not None else rec.get("blocked")
                    fen_x = _inject_blocked_into_fen(str(fen_val), blk)
                    squares, stm = parse_fen_squares(fen_x)
                    raw_b = bytes(SQUARE_TO_CODE.get(ch, _CODE_EMPTY) for ch in squares)
                    t_bit = 1 if stm == "w" else 0
                    ckey = rec.get("canonical_state_key") or rec.get("canonical_key")
                    if ckey is None:
                        parts = fen_x.strip().split()
                        variant_s = str(rec.get("variant") or "standard")
                        ep_s = parts[3] if len(parts) > 3 else "-"
                        if ep_s == "-":
                            cast_s = parts[2] if len(parts) > 2 else "-"
                            ckey = f"{variant_s}:{parts[0]} {stm} {cast_s} -"
                        else:
                            ckey = compute_canonical_state_key(rec)
                    index.add_bytes(raw_b, t_bit, canonical_key=str(ckey), meta=rel_str)

    stats = {
        "checked_paths": checked_paths,
        "missing_paths": missing_paths,
        "files_scanned": len(jsonl_files),
        "records_scanned": records_scanned,
        "unique_states_indexed": len(index),
        "canonical_keys_indexed": len(index.canonical_keys),
    }
    return index, stats


def radius_two_audit(
    roots_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    demo_pools: Union[Sequence[Mapping[str, Any]], Mapping[str, Sequence[Mapping[str, Any]]]],
    pilot_paths: Optional[Sequence[Union[str, Path]]] = None,
    pilot_index: Optional[RadiusTwoIndex] = None,
    variant_name: Optional[str] = None,
    max_hamming: int = 2,
    check_within_split_cross_family: bool = True,
    check_pilot: bool = True,
) -> dict[str, Any]:
    """Audit exact canonical and Hamming<=2 geometry separation across splits, demos, and pilots.

    Checks:
      (a) Across train/val/test splits and across distinct families within a split.
      (b) Query roots against all states in all demonstration pools (`train`, `val`, `test`).
      (c) Query roots against prior pilot query/demo inputs (when `check_pilot=True`).
    """
    demo_index = RadiusTwoIndex(deduplicate_exact_boards=True)
    if isinstance(demo_pools, Mapping):
        for pool_name, games in demo_pools.items():
            _add_games_to_index(demo_index, games, source_tag=f"demo:{pool_name}")
    else:
        _add_games_to_index(demo_index, demo_pools, source_tag="demo")

    eff_pilot_index: Optional[RadiusTwoIndex] = None
    pilot_stats: Optional[dict[str, Any]] = None
    if check_pilot:
        if pilot_index is not None:
            eff_pilot_index = pilot_index
        else:
            eff_pilot_index, pilot_stats = build_pilot_index(pilot_paths=pilot_paths)

    query_index = RadiusTwoIndex(deduplicate_exact_boards=False)
    cross_split_violations: list[dict[str, Any]] = []
    cross_family_violations: list[dict[str, Any]] = []
    vs_demos_violations: list[dict[str, Any]] = []
    vs_pilot_violations: list[dict[str, Any]] = []

    ordered_splits = [s for s in SPLIT_PRIORITY if s in roots_by_split] + sorted(
        s for s in roots_by_split if s not in SPLIT_PRIORITY
    )

    for sp in ordered_splits:
        for idx, root in enumerate(roots_by_split[sp]):
            rec = {"variant": variant_name, **root} if variant_name and "variant" not in root else root
            fid = _extract_family_id(rec, idx)
            rid = str(rec.get("root_id") or f"{sp}_{idx}")
            arr, t_bit = board_similarity_signature(rec)
            ckey = compute_canonical_state_key(rec)

            demo_hit = demo_index.find_collision(
                arr, t_bit, canonical_key=ckey, max_hamming=max_hamming
            )
            if demo_hit is not None:
                vs_demos_violations.append({"split": sp, "root_id": rid, "family_id": fid, **demo_hit})

            if eff_pilot_index is not None:
                pilot_hit = eff_pilot_index.find_collision(
                    arr, t_bit, canonical_key=ckey, max_hamming=max_hamming
                )
                if pilot_hit is not None:
                    vs_pilot_violations.append({"split": sp, "root_id": rid, "family_id": fid, **pilot_hit})

            q_hit = query_index.find_collision(
                arr,
                t_bit,
                canonical_key=ckey,
                max_hamming=max_hamming,
                ignore_family_id=fid,
            )
            if q_hit is not None:
                other_meta = q_hit.get("meta")
                other_sp = other_meta[0] if isinstance(other_meta, tuple) else None
                entry = {
                    "split": sp,
                    "other_split": other_sp,
                    "root_id": rid,
                    "family_id": fid,
                    **q_hit,
                }
                if other_sp is not None and other_sp != sp:
                    cross_split_violations.append(entry)
                elif check_within_split_cross_family:
                    cross_family_violations.append(entry)

            query_index.add_signature(
                arr, t_bit, canonical_key=ckey, family_id=fid, meta=(sp, rid)
            )

    all_violations = (
        cross_split_violations
        + cross_family_violations
        + vs_demos_violations
        + vs_pilot_violations
    )
    return {
        "passed": len(all_violations) == 0,
        "n_violations": len(all_violations),
        "max_hamming": int(max_hamming),
        "cross_split_violations": cross_split_violations,
        "cross_family_violations": cross_family_violations,
        "vs_demos_violations": vs_demos_violations,
        "vs_pilot_violations": vs_pilot_violations,
        "pilot_stats": pilot_stats,
    }


def filter_roots_radius_two(
    candidate_pairs_by_split: Mapping[str, Sequence[tuple[str, Sequence[tuple[dict[str, Any], ...]]]]],
    demo_pools: Union[Sequence[Mapping[str, Any]], Mapping[str, Sequence[Mapping[str, Any]]]],
    pilot_index: Optional[RadiusTwoIndex] = None,
    pilot_paths: Optional[Sequence[Union[str, Path]]] = None,
    existing_roots_by_split: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
    max_hamming: int = 2,
    min_ply: int = 8,
    check_pilot: bool = True,
) -> dict[str, Any]:
    """Filter per-family candidate pairs against demos, prior pilots, and higher-priority splits.

    Selects at most ONE clean pair (<= 2 adjacent states) per family. When both siblings in
    the primary pair pass, both are kept; if one collides, alternate pairs in the same family
    are tried before falling back to a single clean sibling.
    """
    demo_index = RadiusTwoIndex(deduplicate_exact_boards=True)
    if isinstance(demo_pools, Mapping):
        for pool_name, games in demo_pools.items():
            _add_games_to_index(demo_index, games, source_tag=f"demo:{pool_name}")
    else:
        _add_games_to_index(demo_index, demo_pools, source_tag="demo")

    eff_pilot = pilot_index
    if check_pilot and eff_pilot is None:
        eff_pilot, _ = build_pilot_index(pilot_paths=pilot_paths)

    accepted_index = RadiusTwoIndex(deduplicate_exact_boards=False)
    if existing_roots_by_split:
        for sp, roots in existing_roots_by_split.items():
            for idx, r in enumerate(roots):
                fid = _extract_family_id(r, idx)
                arr, t_bit = board_similarity_signature(r)
                ckey = compute_canonical_state_key(r)
                accepted_index.add_signature(
                    arr, t_bit, canonical_key=ckey, family_id=fid, meta=(sp, r.get("root_id"))
                )

    ordered_splits = [s for s in SPLIT_PRIORITY if s in candidate_pairs_by_split] + sorted(
        s for s in candidate_pairs_by_split if s not in SPLIT_PRIORITY
    )

    kept: dict[str, list[dict[str, Any]]] = {}
    stats: dict[str, Any] = {
        "families_input": {},
        "families_retained": {},
        "roots_retained": {},
        "rejected_opening_ply": 0,
        "rejected_vs_demos": 0,
        "rejected_vs_pilot": 0,
        "rejected_cross_family_or_split": 0,
    }

    def _check_single(root: Mapping[str, Any], fid: str) -> tuple[bool, Optional[np.ndarray], int, str]:
        ply_val = root.get("ply")
        if ply_val is not None and int(ply_val) < min_ply:
            stats["rejected_opening_ply"] += 1
            return False, None, 0, ""
        arr, t_bit = board_similarity_signature(root)
        ckey = compute_canonical_state_key(root)
        if demo_index.find_collision(arr, t_bit, canonical_key=ckey, max_hamming=max_hamming) is not None:
            stats["rejected_vs_demos"] += 1
            return False, None, 0, ""
        if eff_pilot is not None and eff_pilot.find_collision(arr, t_bit, canonical_key=ckey, max_hamming=max_hamming) is not None:
            stats["rejected_vs_pilot"] += 1
            return False, None, 0, ""
        if accepted_index.find_collision(
            arr, t_bit, canonical_key=ckey, max_hamming=max_hamming, ignore_family_id=fid
        ) is not None:
            stats["rejected_cross_family_or_split"] += 1
            return False, None, 0, ""
        return True, arr, t_bit, ckey

    for sp in ordered_splits:
        fam_list = candidate_pairs_by_split[sp]
        stats["families_input"][sp] = len(fam_list)
        kept_sp: list[dict[str, Any]] = []
        retained_fams = 0

        for fid, pairs in fam_list:
            chosen_roots: list[tuple[dict[str, Any], np.ndarray, int, str]] = []
            best_partial: list[tuple[dict[str, Any], np.ndarray, int, str]] = []

            for pair in pairs:
                ok_items: list[tuple[dict[str, Any], np.ndarray, int, str]] = []
                for root in pair:
                    ok, arr, t_bit, ckey = _check_single(root, fid)
                    if ok and arr is not None:
                        ok_items.append((dict(root), arr, t_bit, ckey))
                if len(ok_items) == len(pair):
                    chosen_roots = ok_items
                    break
                if len(ok_items) > len(best_partial):
                    best_partial = ok_items

            final_items = chosen_roots if chosen_roots else best_partial
            if final_items:
                retained_fams += 1
                for r_dict, arr, t_bit, ckey in final_items:
                    accepted_index.add_signature(
                        arr, t_bit, canonical_key=ckey, family_id=fid, meta=(sp, fid)
                    )
                    kept_sp.append(r_dict)

        kept[sp] = kept_sp
        stats["families_retained"][sp] = retained_fams
        stats["roots_retained"][sp] = len(kept_sp)

    return {"roots_by_split": kept, "filter_stats": stats}


def audit_dataset_integrity(
    data_dir: Union[str, Path],
    variants: Sequence[str],
    qwen_formatter: Optional[QwenPromptFormatter] = None,
    tokenizer: Optional[ChessV2Tokenizer] = None,
    max_leakage_samples: int = 20,
    independent_demos: Union[bool, str] = "auto",
    run_radius_two_audit: bool = False,
    pilot_paths: Optional[Sequence[Union[str, Path]]] = None,
    pilot_index: Optional[RadiusTwoIndex] = None,
    check_pilot: bool = True,
    max_roots_per_family: Optional[int] = None,
    require_terminal_demos: bool = False,
) -> dict[str, Any]:
    """Audit JSONL splits under `data_dir` for game/state leakage, prompt leakage, and balance.

    Supports both legacy single-pool ``games_demo.jsonl`` and long-study independent
    demo pools ``games_demo_{train,val,test}.jsonl`` (auto-detected when
    ``independent_demos="auto"``). Optionally runs radius-2 geometry + pilot audit
    and per-family root quota/adjacency verification.
    """
    root_dir = Path(data_dir)
    if qwen_formatter is None:
        from chess_icl_v2.models.qwen_model import QwenPromptFormatter

        formatter = QwenPromptFormatter(notation="uci")
    else:
        formatter = qwen_formatter
    tok = tokenizer or ChessV2Tokenizer()

    errors: list[str] = []
    per_variant: dict[str, Any] = {}

    eff_pilot_index = pilot_index
    if run_radius_two_audit and check_pilot and eff_pilot_index is None:
        eff_pilot_index, _ = build_pilot_index(pilot_paths=pilot_paths)

    for variant in variants:
        v_dir = root_dir / variant
        use_independent = (
            bool(independent_demos)
            if isinstance(independent_demos, bool)
            else (v_dir / "games_demo_train.jsonl").exists()
        )

        if use_independent:
            split_files = {
                "demo_train": "games_demo_train.jsonl",
                "demo_val": "games_demo_val.jsonl",
                "demo_test": "games_demo_test.jsonl",
                "train": "positions_train.jsonl",
                "val": "positions_val.jsonl",
                "test": "positions_test.jsonl",
            }
        else:
            split_files = {
                "demo": "games_demo.jsonl",
                "train": "positions_train.jsonl",
                "val": "positions_val.jsonl",
                "test": "positions_test.jsonl",
            }

        loaded: dict[str, list[dict[str, Any]]] = {}
        missing: list[str] = []
        for sp, fname in split_files.items():
            fpath = v_dir / fname
            if fpath.exists():
                loaded[sp] = _load_jsonl(fpath)
            else:
                loaded[sp] = []
                missing.append(fname)

        if missing:
            errors.append(f"[{variant}] Missing JSONL files: {missing}")

        # 1. family/source_game_id disjointness across all loaded files
        gids_by_split: dict[str, set[str]] = {}
        for sp, recs in loaded.items():
            gids_by_split[sp] = {
                str(r.get("family_id") or r.get("source_game_id") or r.get("game_id"))
                for r in recs
                if (r.get("family_id") or r.get("source_game_id") or r.get("game_id")) is not None
            }

        gid_overlaps: dict[str, list[str]] = {}
        sp_names = list(split_files.keys())
        for i in range(len(sp_names)):
            for j in range(i + 1, len(sp_names)):
                s1, s2 = sp_names[i], sp_names[j]
                overlap = sorted(gids_by_split[s1] & gids_by_split[s2])
                if overlap:
                    gid_overlaps[f"{s1}_vs_{s2}"] = overlap
                    errors.append(
                        f"[{variant}] source_game_id overlap ({s1} vs {s2}): {len(overlap)} shared IDs (e.g. {overlap[:3]})"
                    )

        # Optional: verify terminal demos only
        if require_terminal_demos:
            demo_keys = ("demo_train", "demo_val", "demo_test") if use_independent else ("demo",)
            for dk in demo_keys:
                non_term = [
                    d for d in loaded[dk]
                    if not bool(d.get("is_terminal", not bool(d.get("hit_horizon", False))))
                ]
                if non_term:
                    errors.append(f"[{variant}/{dk}] Found {len(non_term)} non-terminal demo games")

        # Optional: verify <= max_roots_per_family and adjacency
        if max_roots_per_family is not None:
            for sp in ("train", "val", "test"):
                by_fid: dict[str, list[dict[str, Any]]] = {}
                for idx, pos in enumerate(loaded[sp]):
                    fid = _extract_family_id(pos, idx)
                    by_fid.setdefault(fid, []).append(pos)
                for fid, items in by_fid.items():
                    if len(items) > max_roots_per_family:
                        errors.append(
                            f"[{variant}/{sp}] Family {fid!r} has {len(items)} > {max_roots_per_family} roots"
                        )
                    elif len(items) == 2 and items[0].get("ply") is not None and items[1].get("ply") is not None:
                        if abs(int(items[0]["ply"]) - int(items[1]["ply"])) != 1:
                            errors.append(
                                f"[{variant}/{sp}] Family {fid!r} pair plies {items[0]['ply']}, {items[1]['ply']} are not adjacent"
                            )

        # 2. canonical_state_key disjointness across train/val/test
        ckeys_by_split: dict[str, set[str]] = {}
        for sp in ("train", "val", "test"):
            recs = loaded[sp]
            ckeys_by_split[sp] = {
                compute_canonical_state_key({"variant": variant, **r})
                for r in recs
            }

        ckey_overlaps: dict[str, int] = {}
        pos_splits = ("train", "val", "test")
        for i in range(len(pos_splits)):
            for j in range(i + 1, len(pos_splits)):
                s1, s2 = pos_splits[i], pos_splits[j]
                overlap = ckeys_by_split[s1] & ckeys_by_split[s2]
                if overlap:
                    ckey_overlaps[f"{s1}_vs_{s2}"] = len(overlap)
                    errors.append(
                        f"[{variant}] canonical_state_key overlap ({s1} vs {s2}): {len(overlap)} shared keys"
                    )

        # Optional: Radius-2 geometry & pilot audit
        r2_report: Optional[dict[str, Any]] = None
        if run_radius_two_audit:
            demo_arg: Any = (
                {k: loaded[k] for k in ("demo_train", "demo_val", "demo_test")}
                if use_independent
                else loaded["demo"]
            )
            r2_report = radius_two_audit(
                roots_by_split={sp: loaded[sp] for sp in ("train", "val", "test")},
                demo_pools=demo_arg,
                pilot_paths=pilot_paths,
                pilot_index=eff_pilot_index,
                variant_name=variant,
                check_pilot=check_pilot,
            )
            if not r2_report["passed"]:
                errors.append(
                    f"[{variant}] radius_two_audit failed with {r2_report['n_violations']} violations"
                )

        # 3. Formatting leakage audit via QwenPromptFormatter & ChessV2Tokenizer
        leakage_findings: list[str] = []
        for sp in ("train", "val", "test"):
            demos_sp = loaded[f"demo_{sp}"] if use_independent else loaded["demo"]
            demo_sample = demos_sp[: min(4, len(demos_sp))]
            for pos in loaded[sp][:max_leakage_samples]:
                query_dict = {
                    "fen": pos.get("query_fen") or pos.get("fen") or pos.get("state"),
                    "blocked_squares": pos.get("blocked_squares") if pos.get("blocked_squares") is not None else pos.get("blocked"),
                }
                if query_dict["fen"] is None:
                    continue
                try:
                    formatted = formatter.format_prompt_with_info(demo_sample, query_dict)
                    findings = check_prompt_leakage(
                        formatted.text,
                        query_record=pos,
                        extra_forbidden_terms=[variant],
                    )
                    leakage_findings.extend(
                        f"[{variant}/{sp}] {msg}" for msg in findings
                    )
                except Exception as exc:
                    leakage_findings.append(f"[{variant}/{sp}] QwenPromptFormatter error: {exc}")

                try:
                    encoded = tok.encode_episode(
                        demo_sample,
                        query_dict["fen"],
                        query_blocked_squares=query_dict["blocked_squares"],
                    )
                    tok_findings = _check_v2_tokens_leakage(encoded, tok)
                    leakage_findings.extend(
                        f"[{variant}/{sp}] {msg}" for msg in tok_findings
                    )
                except Exception as exc:
                    leakage_findings.append(f"[{variant}/{sp}] ChessV2Tokenizer error: {exc}")

        if leakage_findings:
            unique_findings = list(dict.fromkeys(leakage_findings))
            errors.extend(unique_findings)
        else:
            unique_findings = []

        # 4. Balance & distribution stats
        if use_independent:
            split_stats: dict[str, Any] = {
                "demo_train": {"n_demos": len(loaded["demo_train"])},
                "demo_val": {"n_demos": len(loaded["demo_val"])},
                "demo_test": {"n_demos": len(loaded["demo_test"])},
            }
        else:
            split_stats = {
                "demo": {"n_demos": len(loaded["demo"])},
            }
        for sp in ("train", "val", "test"):
            split_stats[sp] = _position_balance_stats(loaded[sp])

        v_report: dict[str, Any] = {
            "independent_demos": use_independent,
            "missing_files": missing,
            "source_game_id_overlaps": gid_overlaps,
            "canonical_key_overlaps": ckey_overlaps,
            "leakage_findings": unique_findings,
            "split_stats": split_stats,
        }
        if r2_report is not None:
            v_report["radius_two_audit"] = {
                "passed": r2_report["passed"],
                "n_violations": r2_report["n_violations"],
                "cross_split": len(r2_report["cross_split_violations"]),
                "cross_family": len(r2_report["cross_family_violations"]),
                "vs_demos": len(r2_report["vs_demos_violations"]),
                "vs_pilot": len(r2_report["vs_pilot_violations"]),
            }
        per_variant[variant] = v_report

    return {
        "passed": len(errors) == 0,
        "n_errors": len(errors),
        "errors": errors,
        "per_variant": per_variant,
    }
