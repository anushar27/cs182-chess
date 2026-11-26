"""Compact fixed-state, 1-token-per-move tokenizer for chess_icl_v2 Track 2.

Design highlights:
- Fixed 70-token board state representation:
  - 64 fixed square tokens in FEN order (a8..h8, ..., a1..h1) from
    {'.', 'X', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k'}
    where 'X' explicitly marks impassable blocked-square barriers.
  - 1 side-to-move token ('SIDE_W' or 'SIDE_B').
  - 4 fixed castling-flag slots ('CASTLE_K'/'CASTLE_NONE', 'CASTLE_Q'/'CASTLE_NONE',
    'CASTLE_k'/'CASTLE_NONE', 'CASTLE_q'/'CASTLE_NONE').
  - 1 en-passant target token ('EP_NONE' or 'EP_<sq>').
- 1-token-per-move UCI encoding:
  - All 64 * 64 = 4096 (from_sq, to_sq) pairs.
  - All 8 * 8 * 4 * 5 = 1280 pawn promotions on ranks 7->8, 2->1, 6->8, and 3->1
    for {q, r, b, n, k} (covering standard, extended_pawn 2-square promotions,
    antichess king promotions, and compositions).
  - Total 5376 move tokens, cutting episode length ~3-4x relative to BPE.
- Episode serialization with segment IDs for causal or segment-bidirectional attention:
  - [CLS] in segment 0.
  - Each demo's [DEMO_START] [STATE] <70> [MOVES] <moves> in one segment.
  - Each demo's [OUT_*] [DEMO_END] in its own isolated subsequent segment.
  - [QUERY] [STATE] <70> [SEP] in the final segment (never dropped or truncated).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence, Union

STANDARD_STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

FILES = ("a", "b", "c", "d", "e", "f", "g", "h")
RANKS = ("1", "2", "3", "4", "5", "6", "7", "8")
SQUARES_A1_H8 = tuple(f"{f}{r}" for r in RANKS for f in FILES)

SQUARE_PIECES = (".", "X", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k")
SIDE_TOKENS = ("SIDE_W", "SIDE_B")
CASTLE_TOKENS = ("CASTLE_NONE", "CASTLE_K", "CASTLE_Q", "CASTLE_k", "CASTLE_q")
EP_TOKENS = ("EP_NONE",) + tuple(f"EP_{sq}" for sq in SQUARES_A1_H8)
PROMO_PIECES = ("q", "r", "b", "n", "k")
PROMO_RANK_PAIRS = (("7", "8"), ("2", "1"), ("6", "8"), ("3", "1"))

STATE_TOKEN_LEN = 70  # 64 squares + 1 side + 4 castling + 1 en-passant


def _square_to_index(sq: Union[int, str]) -> int:
    """Convert python-chess int (0..63, a1=0..h8=63) or algebraic name to 0..63."""
    if isinstance(sq, int):
        if not 0 <= sq < 64:
            raise ValueError(f"Square index out of range 0..63: {sq}")
        return sq
    if isinstance(sq, str) and len(sq) == 2:
        f_ch, r_ch = sq[0].lower(), sq[1]
        if f_ch in FILES and r_ch in RANKS:
            return RANKS.index(r_ch) * 8 + FILES.index(f_ch)
    raise ValueError(f"Invalid square specification: {sq!r}")


def _chess_sq_to_fen_idx(sq: int) -> int:
    """Map python-chess square index (a1=0, h8=63) to FEN traversal index (a8=0, h1=63)."""
    rank = sq // 8
    file = sq % 8
    return (7 - rank) * 8 + file


def _fen_idx_to_chess_sq(fen_idx: int) -> int:
    """Map FEN traversal index (a8=0, h1=63) to python-chess square index (a1=0, h8=63)."""
    rank = 7 - (fen_idx // 8)
    file = fen_idx % 8
    return rank * 8 + file


class ChessV2Tokenizer:
    """Deterministic compact tokenizer for Track 2 Chess ICL v2."""

    PAD_TOKEN = "[PAD]"
    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"
    DEMO_START_TOKEN = "[DEMO_START]"
    DEMO_END_TOKEN = "[DEMO_END]"
    STATE_TOKEN = "[STATE]"
    MOVES_TOKEN = "[MOVES]"
    OUT_W_TOKEN = "[OUT_W]"
    OUT_D_TOKEN = "[OUT_D]"
    OUT_L_TOKEN = "[OUT_L]"
    QUERY_TOKEN = "[QUERY]"

    SPECIAL_TOKENS = (
        PAD_TOKEN,
        CLS_TOKEN,
        SEP_TOKEN,
        DEMO_START_TOKEN,
        DEMO_END_TOKEN,
        STATE_TOKEN,
        MOVES_TOKEN,
        OUT_W_TOKEN,
        OUT_D_TOKEN,
        OUT_L_TOKEN,
        QUERY_TOKEN,
    )

    def __init__(self) -> None:
        self.vocab: list[str] = self._build_vocab()
        self.token_to_id: dict[str, int] = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token: dict[int, str] = {i: t for i, t in enumerate(self.vocab)}
        self.vocab_size: int = len(self.vocab)

        self.pad_id: int = self.token_to_id[self.PAD_TOKEN]
        self.cls_id: int = self.token_to_id[self.CLS_TOKEN]
        self.sep_id: int = self.token_to_id[self.SEP_TOKEN]
        self.demo_start_id: int = self.token_to_id[self.DEMO_START_TOKEN]
        self.demo_end_id: int = self.token_to_id[self.DEMO_END_TOKEN]
        self.state_id: int = self.token_to_id[self.STATE_TOKEN]
        self.moves_id: int = self.token_to_id[self.MOVES_TOKEN]
        self.out_w_id: int = self.token_to_id[self.OUT_W_TOKEN]
        self.out_d_id: int = self.token_to_id[self.OUT_D_TOKEN]
        self.out_l_id: int = self.token_to_id[self.OUT_L_TOKEN]
        self.query_id: int = self.token_to_id[self.QUERY_TOKEN]

        self._outcome_map: dict[str, int] = {
            "win": self.out_w_id,
            "w": self.out_w_id,
            "draw": self.out_d_id,
            "d": self.out_d_id,
            "loss": self.out_l_id,
            "l": self.out_l_id,
        }

        self._move_to_id: dict[str, int] = {}
        self._id_to_move: dict[int, str] = {}
        for tok_id, tok in enumerate(self.vocab):
            if tok.startswith("MOVE_"):
                uci = tok[5:]
                self._move_to_id[uci] = tok_id
                self._id_to_move[tok_id] = uci

    @classmethod
    def _build_vocab(cls) -> list[str]:
        tokens: list[str] = list(cls.SPECIAL_TOKENS)
        tokens.extend(SQUARE_PIECES)
        tokens.extend(SIDE_TOKENS)
        tokens.extend(CASTLE_TOKENS)
        tokens.extend(EP_TOKENS)

        # 64 * 64 = 4096 base (from_sq, to_sq) moves
        for from_sq in SQUARES_A1_H8:
            for to_sq in SQUARES_A1_H8:
                tokens.append(f"MOVE_{from_sq}{to_sq}")

        # 8 * 8 * 4 * 5 = 1280 pawn promotion moves (ranks 7->8, 2->1, 6->8, 3->1; q/r/b/n/k)
        for from_r, to_r in PROMO_RANK_PAIRS:
            for from_f in FILES:
                for to_f in FILES:
                    for promo in PROMO_PIECES:
                        tokens.append(f"MOVE_{from_f}{from_r}{to_f}{to_r}{promo}")

        return tokens

    def _extract_fen_and_blocked(
        self,
        state: Union[str, Mapping[str, Any], Any],
        blocked_squares: Optional[Iterable[Union[int, str]]] = None,
    ) -> tuple[str, list[int]]:
        """Normalize str / dict / board-like state into (fen_str, blocked_sq_indices)."""
        extracted_blocked: list[int] = []

        if isinstance(state, str):
            fen = state
        elif isinstance(state, Mapping):
            fen = (
                state.get("fen")
                or state.get("board_fen")
                or state.get("state")
                or STANDARD_STARTING_FEN
            )
            raw_blocked = state.get("blocked_squares")
            if raw_blocked is None:
                raw_blocked = state.get("blocked")
            if raw_blocked is not None:
                extracted_blocked.extend(_square_to_index(s) for s in raw_blocked)
        else:
            # Duck-typed support for python-chess / VariantBoard objects
            fen_attr = getattr(state, "fen", None)
            if callable(fen_attr):
                fen = fen_attr()
            elif isinstance(fen_attr, str):
                fen = fen_attr
            elif hasattr(state, "board") and callable(getattr(state.board, "fen", None)):
                fen = state.board.fen()
            else:
                raise TypeError(f"Unsupported state type: {type(state)!r}")

            raw_blocked = getattr(state, "blocked_squares", None)
            if raw_blocked is None:
                raw_blocked = getattr(state, "blocked", None)
            if raw_blocked is not None:
                extracted_blocked.extend(_square_to_index(s) for s in raw_blocked)

        if blocked_squares is not None:
            extracted_blocked.extend(_square_to_index(s) for s in blocked_squares)

        return fen, extracted_blocked

    def encode_state(
        self,
        state: Union[str, Mapping[str, Any], Any],
        blocked_squares: Optional[Iterable[Union[int, str]]] = None,
    ) -> list[int]:
        """Encode a board state into exactly 70 fixed-length token IDs.

        Layout (70 tokens):
            [0..63]:  64 squares in FEN order (a8..h8, ..., a1..h1) from
                      {'.', 'X', P, N, B, R, Q, K, p, n, b, r, q, k}
            [64]:     side-to-move ('SIDE_W' or 'SIDE_B')
            [65..68]: 4 castling slots ('CASTLE_K'/NONE, 'CASTLE_Q'/NONE,
                      'CASTLE_k'/NONE, 'CASTLE_q'/NONE)
            [69]:     en-passant square ('EP_NONE' or 'EP_<sq>')
        """
        fen, blocked_list = self._extract_fen_and_blocked(state, blocked_squares)
        parts = fen.strip().split()
        if not parts:
            raise ValueError(f"Empty FEN string: {fen!r}")

        placement = parts[0]
        side = parts[1].lower() if len(parts) > 1 else "w"
        castling = parts[2] if len(parts) > 2 else "-"
        ep_field = parts[3].lower() if len(parts) > 3 else "-"

        squares: list[str] = []
        for ch in placement:
            if ch == "/":
                continue
            if ch.isdigit():
                squares.extend(["."] * int(ch))
            elif ch in SQUARE_PIECES:
                squares.append(ch)
            elif ch == "x":
                squares.append("X")
            else:
                raise ValueError(f"Unsupported character {ch!r} in FEN placement: {placement!r}")

        if len(squares) != 64:
            raise ValueError(
                f"FEN placement decoded to {len(squares)} squares (expected 64): {placement!r}"
            )

        for sq in blocked_list:
            squares[_chess_sq_to_fen_idx(sq)] = "X"

        tokens = [self.token_to_id[s] for s in squares]

        # Side to move
        if side == "w":
            tokens.append(self.token_to_id["SIDE_W"])
        elif side == "b":
            tokens.append(self.token_to_id["SIDE_B"])
        else:
            raise ValueError(f"Invalid side-to-move {side!r} in FEN: {fen!r}")

        # 4 fixed castling flag tokens
        for flag in ("K", "Q", "k", "q"):
            tok = f"CASTLE_{flag}" if flag in castling else "CASTLE_NONE"
            tokens.append(self.token_to_id[tok])

        # En-passant target token
        if ep_field == "-":
            tokens.append(self.token_to_id["EP_NONE"])
        else:
            ep_tok = f"EP_{ep_field}"
            if ep_tok not in self.token_to_id:
                raise ValueError(f"Invalid en-passant square {ep_field!r} in FEN: {fen!r}")
            tokens.append(self.token_to_id[ep_tok])

        assert len(tokens) == STATE_TOKEN_LEN
        return tokens

    def decode_state(self, token_ids: Sequence[int]) -> str:
        """Decode 70 state token IDs back into a 4-field FEN string (preserving 'X')."""
        filtered = [
            tid for tid in token_ids
            if tid not in (self.pad_id, self.state_id, self.query_id, self.sep_id, self.cls_id)
        ]
        if len(filtered) != STATE_TOKEN_LEN:
            raise ValueError(
                f"Expected {STATE_TOKEN_LEN} state tokens, got {len(filtered)}"
            )

        sq_tokens = [self.id_to_token[t] for t in filtered[:64]]
        rows: list[str] = []
        for r in range(8):
            row_chars = sq_tokens[r * 8 : (r + 1) * 8]
            compacted = []
            empty_run = 0
            for ch in row_chars:
                if ch == ".":
                    empty_run += 1
                else:
                    if empty_run > 0:
                        compacted.append(str(empty_run))
                        empty_run = 0
                    compacted.append(ch)
            if empty_run > 0:
                compacted.append(str(empty_run))
            rows.append("".join(compacted))
        placement = "/".join(rows)

        side_tok = self.id_to_token[filtered[64]]
        if side_tok == "SIDE_W":
            side = "w"
        elif side_tok == "SIDE_B":
            side = "b"
        else:
            raise ValueError(f"Invalid side token at offset 64: {side_tok!r}")

        castle_chars = []
        for idx, flag in enumerate(("K", "Q", "k", "q")):
            c_tok = self.id_to_token[filtered[65 + idx]]
            if c_tok == f"CASTLE_{flag}":
                castle_chars.append(flag)
            elif c_tok != "CASTLE_NONE":
                raise ValueError(f"Invalid castling token at offset {65 + idx}: {c_tok!r}")
        castling = "".join(castle_chars) if castle_chars else "-"

        ep_tok = self.id_to_token[filtered[69]]
        if ep_tok == "EP_NONE":
            ep = "-"
        elif ep_tok.startswith("EP_"):
            ep = ep_tok[3:]
        else:
            raise ValueError(f"Invalid en-passant token at offset 69: {ep_tok!r}")

        return f"{placement} {side} {castling} {ep}"

    def extract_blocked_squares(self, token_ids: Sequence[int]) -> list[int]:
        """Return sorted python-chess square indices (0..63) marked as barrier 'X'."""
        filtered = [
            tid for tid in token_ids
            if tid not in (self.pad_id, self.state_id, self.query_id, self.sep_id, self.cls_id)
        ]
        if len(filtered) < 64:
            raise ValueError(f"Expected at least 64 square tokens, got {len(filtered)}")
        x_id = self.token_to_id["X"]
        blocked = [_fen_idx_to_chess_sq(i) for i, tid in enumerate(filtered[:64]) if tid == x_id]
        return sorted(blocked)

    def encode_move(self, uci_move: str) -> int:
        """Encode a single UCI move string into 1 token ID."""
        uci = uci_move.strip().lower()
        tok_id = self._move_to_id.get(uci)
        if tok_id is None:
            raise ValueError(f"Unsupported or invalid UCI move: {uci_move!r}")
        return tok_id

    def decode_move(self, token_id: int) -> str:
        """Decode a single move token ID back into a UCI move string."""
        uci = self._id_to_move.get(token_id)
        if uci is None:
            raise ValueError(f"Token ID {token_id} is not a move token")
        return uci

    def encode_moves(self, moves: Sequence[str]) -> list[int]:
        """Encode a sequence of UCI moves (1 token per move)."""
        return [self.encode_move(m) for m in moves]

    def decode_moves(self, token_ids: Sequence[int]) -> list[str]:
        """Decode a sequence of move token IDs."""
        return [self.decode_move(tid) for tid in token_ids]

    def _encode_outcome(self, outcome: str) -> int:
        key = str(outcome).strip().lower()
        tok_id = self._outcome_map.get(key)
        if tok_id is None:
            raise ValueError(f"Unknown demo outcome {outcome!r}; expected win/draw/loss or W/D/L")
        return tok_id

    def encode_demo(
        self,
        demo: Union[Mapping[str, Any], Any],
    ) -> tuple[list[int], list[int]]:
        """Encode one demonstration game into (body_tokens, outcome_tokens).

        Returns:
            body_tokens: [DEMO_START, STATE, <70 state tokens>, MOVES, <move_1..move_L>]
            outcome_tokens: [OUT_*, DEMO_END]
        """
        if isinstance(demo, Mapping):
            init_state = (
                demo.get("initial_state")
                or demo.get("initial_fen")
                or demo.get("start_fen")
                or demo.get("state")
                or demo.get("fen")
                or STANDARD_STARTING_FEN
            )
            blocked = demo.get("blocked_squares")
            if blocked is None:
                blocked = demo.get("blocked")
            moves = demo.get("moves", ())
            outcome = demo.get("result")
            if outcome is None:
                outcome = demo.get("outcome", "draw")
        else:
            init_state = (
                getattr(demo, "initial_state", None)
                or getattr(demo, "initial_fen", None)
                or getattr(demo, "start_fen", None)
                or getattr(demo, "state", None)
                or getattr(demo, "fen", None)
                or STANDARD_STARTING_FEN
            )
            blocked = getattr(demo, "blocked_squares", None)
            if blocked is None:
                blocked = getattr(demo, "blocked", None)
            moves = getattr(demo, "moves", ())
            outcome = getattr(demo, "result", None)
            if outcome is None:
                outcome = getattr(demo, "outcome", "draw")

        state_tokens = self.encode_state(init_state, blocked_squares=blocked)
        move_tokens = self.encode_moves(moves)
        outcome_id = self._encode_outcome(outcome)

        body_tokens = (
            [self.demo_start_id, self.state_id]
            + state_tokens
            + [self.moves_id]
            + move_tokens
        )
        outcome_tokens = [outcome_id, self.demo_end_id]
        return body_tokens, outcome_tokens

    def encode_episode(
        self,
        demos: Sequence[Union[Mapping[str, Any], Any]],
        query_state: Union[str, Mapping[str, Any], Any],
        max_len: int = 8192,
        allow_trim_demo_prefix: bool = False,
        query_blocked_squares: Optional[Iterable[Union[int, str]]] = None,
    ) -> dict[str, Any]:
        """Serialize an ICL episode into token IDs, segment IDs, and metadata.

        Segment assignment:
            - Segment 0: [CLS]
            - For each kept demo:
                - Segment s:   [DEMO_START, STATE, <70 state tokens>, MOVES, <moves>]
                - Segment s+1: [OUT_*, DEMO_END]
            - Final segment: [QUERY, STATE, <70 state tokens>, SEP]

        Safety guarantee:
            The query state is NEVER dropped or truncated. If the full episode
            exceeds ``max_len``, whole demos are dropped (oldest first, skipping
            any oversized demo) or—only if ``allow_trim_demo_prefix=True``—demo
            move prefixes are trimmed while preserving demo states and outcomes.
        """
        prefix_tokens = [self.cls_id]
        query_state_tokens = self.encode_state(query_state, blocked_squares=query_blocked_squares)
        query_tokens = [self.query_id, self.state_id] + query_state_tokens + [self.sep_id]

        fixed_overhead = len(prefix_tokens) + len(query_tokens)
        if fixed_overhead > max_len:
            raise ValueError(
                f"max_len={max_len} is too small for un-truncatable query "
                f"(requires at least {fixed_overhead} tokens)"
            )

        demo_budget = max_len - fixed_overhead
        encoded_demos = [self.encode_demo(d) for d in demos]

        # Select demos from newest to oldest so oldest demos are dropped first on overflow
        selected_reversed: list[tuple[list[int], list[int]]] = []
        used_budget = 0
        was_truncated = False

        # Prefix inside body_tokens before moves: [DEMO_START, STATE, <70>, MOVES] = 73 tokens
        body_header_len = 2 + STATE_TOKEN_LEN + 1

        for body_tokens, outcome_tokens in reversed(encoded_demos):
            demo_len = len(body_tokens) + len(outcome_tokens)
            if used_budget + demo_len <= demo_budget:
                selected_reversed.append((body_tokens, outcome_tokens))
                used_budget += demo_len
            else:
                was_truncated = True
                if allow_trim_demo_prefix:
                    min_demo_len = body_header_len + len(outcome_tokens)
                    remaining = demo_budget - used_budget
                    if remaining >= min_demo_len:
                        max_moves_keep = remaining - min_demo_len
                        header = body_tokens[:body_header_len]
                        moves = body_tokens[body_header_len:]
                        trimmed_moves = moves[-max_moves_keep:] if max_moves_keep > 0 else []
                        trimmed_body = header + trimmed_moves
                        selected_reversed.append((trimmed_body, outcome_tokens))
                        used_budget += len(trimmed_body) + len(outcome_tokens)

        kept_demos = list(reversed(selected_reversed))

        input_ids: list[int] = list(prefix_tokens)
        segment_ids: list[int] = [0] * len(prefix_tokens)
        seg_idx = 1

        for body_tokens, outcome_tokens in kept_demos:
            input_ids.extend(body_tokens)
            segment_ids.extend([seg_idx] * len(body_tokens))
            seg_idx += 1

            input_ids.extend(outcome_tokens)
            segment_ids.extend([seg_idx] * len(outcome_tokens))
            seg_idx += 1

        query_start_idx = len(input_ids)
        input_ids.extend(query_tokens)
        segment_ids.extend([seg_idx] * len(query_tokens))

        assert len(input_ids) <= max_len
        assert len(input_ids) == len(segment_ids)

        return {
            "input_ids": input_ids,
            "segment_ids": segment_ids,
            "attention_mask": [1] * len(input_ids),
            "was_truncated": was_truncated,
            "num_demos_kept": len(kept_demos),
            "num_demos_total": len(demos),
            "query_start_idx": query_start_idx,
        }


def collate_episodes(
    episodes: Sequence[Mapping[str, Any]],
    pad_token_id: int = 0,
) -> dict[str, list[list[int]]]:
    """Right-pad a batch of encoded episodes to uniform length."""
    if not episodes:
        return {"input_ids": [], "segment_ids": [], "attention_mask": []}

    max_t = max(len(ep["input_ids"]) for ep in episodes)
    batch_ids: list[list[int]] = []
    batch_segs: list[list[int]] = []
    batch_masks: list[list[int]] = []

    for ep in episodes:
        ids = list(ep["input_ids"])
        segs = list(ep["segment_ids"])
        masks = list(ep.get("attention_mask", [1] * len(ids)))
        pad_len = max_t - len(ids)
        if pad_len > 0:
            last_seg = segs[-1] if segs else 0
            ids.extend([pad_token_id] * pad_len)
            segs.extend([last_seg] * pad_len)
            masks.extend([0] * pad_len)
        batch_ids.append(ids)
        batch_segs.append(segs)
        batch_masks.append(masks)

    return {
        "input_ids": batch_ids,
        "segment_ids": batch_segs,
        "attention_mask": batch_masks,
    }


__all__ = [
    "ChessV2Tokenizer",
    "STANDARD_STARTING_FEN",
    "STATE_TOKEN_LEN",
    "collate_episodes",
]
