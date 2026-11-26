"""Non-neural baselines, feature extractors, and slicing helpers for chess_icl_v2."""

from __future__ import annotations

import re
from typing import Any, Literal, Optional, Protocol, Sequence, runtime_checkable

import chess
import numpy as np
import torch

PIECE_VALUES: dict[str, int] = {
    "P": 1,
    "N": 3,
    "B": 3,
    "R": 5,
    "Q": 9,
}

PIECE_TYPES_ORDER: tuple[str, ...] = ("P", "N", "B", "R", "Q")

BOARD_PIECE_CHANNELS: dict[str, int] = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
}

_UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbnkQRBNK]?$")


@runtime_checkable
class Predictor(Protocol):
    """Unified predictor protocol for baselines and neural models."""

    def predict_proba(self, episodes: list[dict[str, Any]]) -> np.ndarray:
        """Predict [p_win, p_draw, p_loss] probabilities of shape (N, 3)."""
        ...


# ---------------------------------------------------------------------------
# Episode & Demo Field Accessors
# ---------------------------------------------------------------------------


def get_query_fen(episode: dict[str, Any]) -> str:
    """Extract query FEN string from an episode dict."""
    if "query_fen" in episode and episode["query_fen"]:
        return str(episode["query_fen"])
    if "fen" in episode and episode["fen"]:
        return str(episode["fen"])
    query = episode.get("query")
    if isinstance(query, dict) and query.get("fen"):
        return str(query["fen"])
    raise KeyError("Episode dict missing query FEN ('query_fen', 'fen', or 'query.fen')")


def get_query_side(episode: dict[str, Any]) -> str:
    """Extract query side to move as 'w' (White) or 'b' (Black)."""
    for key in ("side_to_move", "query_side", "turn"):
        if key in episode and episode[key] is not None:
            val = episode[key]
            if isinstance(val, (bool, np.bool_)):
                return "w" if bool(val) else "b"
            s = str(val).strip().lower()
            if s in {"w", "white"}:
                return "w"
            if s in {"b", "black"}:
                return "b"
    if "is_white_to_move" in episode and episode["is_white_to_move"] is not None:
        return "w" if bool(episode["is_white_to_move"]) else "b"

    fen = get_query_fen(episode)
    parts = fen.strip().split()
    if len(parts) >= 2:
        s = parts[1].strip().lower()
        if s in {"w", "white"}:
            return "w"
        if s in {"b", "black"}:
            return "b"
    return "w"


def _outcome_to_vec(val: Any) -> np.ndarray:
    """Convert a discrete or soft outcome representation into a (3,) W/D/L vector."""
    if isinstance(val, (list, tuple, np.ndarray)):
        arr = np.asarray(val, dtype=np.float64)
        if arr.shape == (3,):
            s = float(np.sum(arr))
            return arr / s if s > 0 else np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    if isinstance(val, (int, np.integer)):
        idx = int(val)
        if idx in (0, 1, 2):
            out = np.zeros(3, dtype=np.float64)
            out[idx] = 1.0
            return out
    s = str(val).strip().lower()
    if s in {"win", "w", "white", "1-0"}:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if s in {"draw", "d", "1/2-1/2", "tie"}:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if s in {"loss", "l", "black", "0-1"}:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    raise ValueError(f"Unrecognized outcome value: {val!r}")


def get_episode_target(episode: dict[str, Any]) -> np.ndarray:
    """Extract soft [p_win, p_draw, p_loss] target from query side-to-move perspective."""
    for key in ("y_true", "target", "target_probs", "outcome_probs", "probs"):
        if key in episode and episode[key] is not None:
            return _outcome_to_vec(episode[key])
    if "label" in episode and episode["label"] is not None:
        return _outcome_to_vec(episode["label"])
    query = episode.get("query")
    if isinstance(query, dict):
        for key in ("y_true", "target", "target_probs", "outcome_probs", "label"):
            if key in query and query[key] is not None:
                return _outcome_to_vec(query[key])
    raise KeyError("Episode dict missing target probabilities or label")


def get_demo_moves(demo: dict[str, Any]) -> list[str]:
    """Extract move sequence from a demo game dict as a list of move strings."""
    for key in ("moves", "uci_moves", "san_moves"):
        if key in demo and demo[key] is not None:
            raw = demo[key]
            if isinstance(raw, str):
                return [tok for tok in raw.strip().split() if tok]
            return [str(tok) for tok in raw]
    return []


def get_demo_outcome_white(demo: dict[str, Any]) -> np.ndarray:
    """Extract White-perspective realized outcome [p_W, p_D, p_L] from a demo dict."""
    for key in ("outcome_probs_white", "y_white", "outcome_white", "outcome", "result", "winner"):
        if key in demo and demo[key] is not None:
            return _outcome_to_vec(demo[key])
    raise KeyError("Demo dict missing White-perspective outcome ('outcome', 'result', etc.)")


def white_to_stm_probs(p_white: np.ndarray, side: str) -> np.ndarray:
    """Convert [p_W, p_D, p_L] from White perspective to query side-to-move perspective."""
    if side == "w":
        return np.asarray(p_white, dtype=np.float64).copy()
    if side == "b":
        return np.array([p_white[2], p_white[1], p_white[0]], dtype=np.float64)
    raise ValueError(f"Invalid side {side!r}; expected 'w' or 'b'")


# ---------------------------------------------------------------------------
# Board Parsing, Material, & 64*13 + 1 Trajectory Reconstruction
# ---------------------------------------------------------------------------


def parse_fen_squares(fen: str) -> tuple[list[str], str]:
    """Parse FEN placement and side-to-move into 64 squares (a1..h8) and 'w'/'b'.

    Supports standard FEN characters plus 'X' for blocked squares (BlockedSquares variant).
    """
    parts = fen.strip().split()
    placement = parts[0]
    stm = "w"
    if len(parts) >= 2 and parts[1].lower() in {"w", "b"}:
        stm = parts[1].lower()

    ranks = placement.split("/")
    if len(ranks) != 8:
        raise ValueError(f"Invalid FEN placement (expected 8 ranks): {fen!r}")

    squares = ["."] * 64
    for r_idx, rank_str in enumerate(ranks):
        board_rank = 7 - r_idx  # FEN starts at rank 8 down to rank 1
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


def compute_piece_counts(fen: str) -> tuple[dict[str, int], dict[str, int]]:
    """Count White and Black pieces (P, N, B, R, Q) from a FEN string."""
    placement = fen.strip().split()[0]
    w_counts = {pt: 0 for pt in PIECE_TYPES_ORDER}
    b_counts = {pt: 0 for pt in PIECE_TYPES_ORDER}
    for ch in placement:
        up = ch.upper()
        if up in w_counts:
            if ch.isupper():
                w_counts[up] += 1
            else:
                b_counts[up] += 1
    return w_counts, b_counts


def compute_standard_material_diff(fen: str) -> int:
    """Compute White minus Black standard material (P=1, N=3, B=3, R=5, Q=9)."""
    w_counts, b_counts = compute_piece_counts(fen)
    w_mat = sum(PIECE_VALUES[pt] * w_counts[pt] for pt in PIECE_TYPES_ORDER)
    b_mat = sum(PIECE_VALUES[pt] * b_counts[pt] for pt in PIECE_TYPES_ORDER)
    return int(w_mat - b_mat)


def extract_query_material_features(episode: dict[str, Any]) -> np.ndarray:
    """Extract 9 static query features from the side-to-move perspective.

    Features:
      [material_diff_stm, own_material, opp_material,
       diff_P, diff_N, diff_B, diff_R, diff_Q, is_white_to_move]
    """
    fen = get_query_fen(episode)
    stm = get_query_side(episode)
    w_counts, b_counts = compute_piece_counts(fen)
    w_mat = float(sum(PIECE_VALUES[pt] * w_counts[pt] for pt in PIECE_TYPES_ORDER))
    b_mat = float(sum(PIECE_VALUES[pt] * b_counts[pt] for pt in PIECE_TYPES_ORDER))

    if stm == "w":
        own_mat, opp_mat = w_mat, b_mat
        diffs = [float(w_counts[pt] - b_counts[pt]) for pt in PIECE_TYPES_ORDER]
        is_w = 1.0
    else:
        own_mat, opp_mat = b_mat, w_mat
        diffs = [float(b_counts[pt] - w_counts[pt]) for pt in PIECE_TYPES_ORDER]
        is_w = 0.0

    return np.array(
        [own_mat - opp_mat, own_mat, opp_mat, *diffs, is_w],
        dtype=np.float64,
    )


def extract_demo_summary_features(episode: dict[str, Any]) -> np.ndarray:
    """Extract 7 demo summary features for LengthHistogramBaseline.

    Features:
      [mean_len, std_len, frac_w, frac_d, frac_l, is_white_to_move, K]
    """
    demos = episode.get("demos", [])
    k = len(demos)
    is_w = 1.0 if get_query_side(episode) == "w" else 0.0
    if k == 0:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, is_w, 0.0], dtype=np.float64)

    lengths = np.array([len(get_demo_moves(d)) for d in demos], dtype=np.float64)
    outcomes = np.mean([get_demo_outcome_white(d) for d in demos], axis=0)
    return np.array(
        [
            float(np.mean(lengths)),
            float(np.std(lengths)),
            float(outcomes[0]),
            float(outcomes[1]),
            float(outcomes[2]),
            is_w,
            float(k),
        ],
        dtype=np.float64,
    )


def encode_squares_64x13_plus_1(squares: Sequence[str], stm: str) -> np.ndarray:
    """Encode 64 board squares + side-to-move into a (833,) float32 vector.

    Each square uses 13 channels: 12 piece types (P..K, p..k) and channel 12 for empty '.'.
    Blocked squares 'X' leave all 13 channels at 0 so they stay distinct from pieces and empty.
    Index 832 encodes side-to-move (1.0 for White, 0.0 for Black).
    """
    vec = np.zeros(64 * 13 + 1, dtype=np.float32)
    for sq_idx, ch in enumerate(squares):
        base = sq_idx * 13
        if ch in BOARD_PIECE_CHANNELS:
            vec[base + BOARD_PIECE_CHANNELS[ch]] = 1.0
        elif ch == ".":
            vec[base + 12] = 1.0
        # For blocked square 'X' (or any unknown barrier), keep 0 across all 13 channels
    vec[64 * 13] = 1.0 if stm == "w" else 0.0
    return vec


def encode_fen_64x13_plus_1(fen: str) -> np.ndarray:
    """Encode a FEN string into a (833,) float32 vector."""
    squares, stm = parse_fen_squares(fen)
    return encode_squares_64x13_plus_1(squares, stm)


def _sq_name_to_idx(sq: str) -> int:
    file_idx = ord(sq[0]) - ord("a")
    rank_idx = int(sq[1]) - 1
    return rank_idx * 8 + file_idx


def _apply_uci_on_squares(squares: list[str], stm: str, move_str: str) -> tuple[list[str], str]:
    """Apply a UCI move directly on the 64-square board without standard legality checks.

    This preserves variant moves (e.g., DiagonalRook, StraightBishop, ShortKnight,
    BlockedSquares) that would fail standard python-chess legality validation.
    """
    new_sq = list(squares)
    from_idx = _sq_name_to_idx(move_str[0:2])
    to_idx = _sq_name_to_idx(move_str[2:4])
    piece = new_sq[from_idx]

    from_file, from_rank = from_idx % 8, from_idx // 8
    to_file, to_rank = to_idx % 8, to_idx // 8

    # Handle castling rook shift
    if piece in ("K", "k") and from_rank == to_rank and abs(to_file - from_file) == 2:
        if to_file == 6:  # kingside
            rook_from = from_rank * 8 + 7
            rook_to = from_rank * 8 + 5
        else:  # queenside (to_file == 2)
            rook_from = from_rank * 8 + 0
            rook_to = from_rank * 8 + 3
        new_sq[rook_to] = new_sq[rook_from]
        new_sq[rook_from] = "."

    # Handle en-passant diagonal pawn capture into an empty square
    if piece in ("P", "p") and from_file != to_file and new_sq[to_idx] == ".":
        ep_sq = from_rank * 8 + to_file
        if new_sq[ep_sq] in ("P", "p"):
            new_sq[ep_sq] = "."

    if len(move_str) == 5:
        promo_ch = move_str[4].upper() if stm == "w" else move_str[4].lower()
        new_sq[to_idx] = promo_ch
    else:
        new_sq[to_idx] = piece
    new_sq[from_idx] = "."

    next_stm = "b" if stm == "w" else "w"
    return new_sq, next_stm


def reconstruct_demo_states(
    demo: dict[str, Any],
    state_mode: Literal["trajectories", "final"] = "trajectories",
) -> list[np.ndarray]:
    """Reconstruct (833,) board vectors for a demo's trajectory or final state."""
    # Prefer precomputed state FENs if provided
    if state_mode == "final" and demo.get("final_fen"):
        return [encode_fen_64x13_plus_1(str(demo["final_fen"]))]

    precomputed_fens = demo.get("fens") or demo.get("states")
    if precomputed_fens:
        if state_mode == "final":
            return [encode_fen_64x13_plus_1(str(precomputed_fens[-1]))]
        return [encode_fen_64x13_plus_1(str(f)) for f in precomputed_fens]

    init_fen = str(demo.get("initial_fen") or demo.get("start_fen") or chess.STARTING_FEN)
    squares, stm = parse_fen_squares(init_fen)
    moves = get_demo_moves(demo)

    states: list[np.ndarray] = [encode_squares_64x13_plus_1(squares, stm)]
    std_board: Optional[chess.Board] = None
    if "X" not in init_fen.split()[0]:
        try:
            std_board = chess.Board(init_fen)
        except ValueError:
            std_board = None

    for mv in moves:
        if _UCI_RE.match(mv):
            squares, stm = _apply_uci_on_squares(squares, stm, mv)
            if std_board is not None:
                try:
                    std_board.push(chess.Move.from_uci(mv.lower()))
                except Exception:
                    std_board = None
        elif std_board is not None:
            try:
                parsed = std_board.parse_san(mv)
                uci = parsed.uci()
                std_board.push(parsed)
                squares, stm = _apply_uci_on_squares(squares, stm, uci)
            except Exception:
                # Variant SAN move not valid in standard chess: keep last known board, flip turn
                stm = "b" if stm == "w" else "w"
        else:
            stm = "b" if stm == "w" else "w"
        states.append(encode_squares_64x13_plus_1(squares, stm))

    if state_mode == "final":
        return [states[-1]]
    return states


# ---------------------------------------------------------------------------
# Pure PyTorch Multinomial Logistic Regression (Soft Targets)
# ---------------------------------------------------------------------------


class MultinomialLogisticRegression:
    """Convex L2-regularized multinomial logistic regression via PyTorch L-BFGS."""

    def __init__(self, l2_reg: float = 1e-4, max_iter: int = 200):
        self.l2_reg = float(l2_reg)
        self.max_iter = int(max_iter)
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.weight_: Optional[torch.Tensor] = None
        self.bias_: Optional[torch.Tensor] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultinomialLogisticRegression":
        """Fit linear softmax classifier on features X (N, D) and soft targets y (N, 3)."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if X_arr.ndim != 2 or y_arr.ndim != 2 or len(X_arr) != len(y_arr) or y_arr.shape[1] != 3:
            raise ValueError(f"Invalid shapes for fit: X={X_arr.shape}, y={y_arr.shape}")
        if len(X_arr) == 0:
            raise ValueError("Cannot fit MultinomialLogisticRegression on empty data")

        self.mean_ = np.mean(X_arr, axis=0)
        std = np.std(X_arr, axis=0)
        self.std_ = np.where(std < 1e-8, 1.0, std)
        X_norm = (X_arr - self.mean_) / self.std_

        x_t = torch.tensor(X_norm, dtype=torch.float64)
        y_t = torch.tensor(y_arr, dtype=torch.float64)
        d = X_norm.shape[1]

        # Deterministic zero initialization for convex objective
        w = torch.zeros((d, 3), dtype=torch.float64, requires_grad=True)
        b = torch.zeros((3,), dtype=torch.float64, requires_grad=True)

        optimizer = torch.optim.LBFGS(
            [w, b],
            lr=1.0,
            max_iter=self.max_iter,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            logits = x_t @ w + b
            log_probs = torch.log_softmax(logits, dim=-1)
            ce_loss = -torch.mean(torch.sum(y_t * log_probs, dim=-1))
            reg_loss = 0.5 * self.l2_reg * torch.sum(w * w)
            loss = ce_loss + reg_loss
            loss.backward()
            return loss

        optimizer.step(closure)
        self.weight_ = w.detach().clone()
        self.bias_ = b.detach().clone()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities of shape (N, 3)."""
        if self.weight_ is None or self.bias_ is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("Model must be fitted before calling predict_proba")
        X_arr = np.asarray(X, dtype=np.float64)
        if len(X_arr) == 0:
            return np.empty((0, 3), dtype=np.float64)
        X_norm = (X_arr - self.mean_) / self.std_
        with torch.no_grad():
            x_t = torch.tensor(X_norm, dtype=torch.float64)
            logits = x_t @ self.weight_ + self.bias_
            probs = torch.softmax(logits, dim=-1).numpy()
        # Ensure exact simplex normalization in float64
        probs = np.clip(probs, 0.0, 1.0)
        probs /= np.sum(probs, axis=-1, keepdims=True)
        return probs


# ---------------------------------------------------------------------------
# Baseline Predictors
# ---------------------------------------------------------------------------


class UniformBaseline:
    """Predicts uniform [1/3, 1/3, 1/3] for every query."""

    def predict_proba(self, episodes: list[dict[str, Any]]) -> np.ndarray:
        n = len(episodes)
        if n == 0:
            return np.empty((0, 3), dtype=np.float64)
        return np.full((n, 3), 1.0 / 3.0, dtype=np.float64)


class TrainingPriorBaseline:
    """Predicts the marginal class distribution of training query targets."""

    def __init__(self, alpha: float = 0.0):
        self.alpha = float(alpha)
        self.prior_: np.ndarray = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)

    def fit(
        self,
        train_episodes: Optional[Sequence[dict[str, Any]]] = None,
        y_train: Optional[np.ndarray] = None,
    ) -> "TrainingPriorBaseline":
        if y_train is not None:
            targets = np.asarray(y_train, dtype=np.float64)
        elif train_episodes is not None:
            if len(train_episodes) == 0:
                self.prior_ = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
                return self
            targets = np.vstack([get_episode_target(ep) for ep in train_episodes])
        else:
            raise ValueError("Provide either train_episodes or y_train to fit()")

        if len(targets) == 0:
            self.prior_ = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
            return self

        counts = np.sum(targets, axis=0) + self.alpha
        self.prior_ = counts / np.sum(counts)
        return self

    def predict_proba(self, episodes: list[dict[str, Any]]) -> np.ndarray:
        n = len(episodes)
        if n == 0:
            return np.empty((0, 3), dtype=np.float64)
        return np.tile(self.prior_, (n, 1))


class DemoMajorityBaseline:
    """Laplace-smoothed empirical W/D/L distribution of K demos, converted to query STM.

    Demo outcomes are stored from White's perspective; when the query is Black to move,
    Win and Loss probabilities are swapped. Falls back to `prior` (or uniform) when K=0.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        prior: Optional[Sequence[float]] = None,
    ):
        self.alpha = float(alpha)
        if prior is None:
            self.prior = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
        else:
            p = np.asarray(prior, dtype=np.float64)
            if p.shape != (3,) or np.sum(p) <= 0:
                raise ValueError(f"prior must be a positive 3-vector, got {prior!r}")
            self.prior = p / np.sum(p)

    def predict_proba(self, episodes: list[dict[str, Any]]) -> np.ndarray:
        n = len(episodes)
        if n == 0:
            return np.empty((0, 3), dtype=np.float64)

        out = np.empty((n, 3), dtype=np.float64)
        for i, ep in enumerate(episodes):
            demos = ep.get("demos", [])
            k = len(demos)
            if k == 0:
                out[i] = self.prior
                continue

            counts_white = np.sum([get_demo_outcome_white(d) for d in demos], axis=0)
            p_white = (counts_white + self.alpha * 3.0 * self.prior) / (
                float(k) + 3.0 * self.alpha
            )
            p_white /= np.sum(p_white)
            side = get_query_side(ep)
            out[i] = white_to_stm_probs(p_white, side)
        return out


class LengthHistogramBaseline:
    """Multinomial logistic regression on demo summary features.

    Features: [mean_len, std_len, frac_w, frac_d, frac_l, is_white_to_move, K].
    """

    def __init__(self, l2_reg: float = 1e-4, max_iter: int = 200):
        self.clf = MultinomialLogisticRegression(l2_reg=l2_reg, max_iter=max_iter)

    def featurize(self, episodes: Sequence[dict[str, Any]]) -> np.ndarray:
        if len(episodes) == 0:
            return np.empty((0, 7), dtype=np.float64)
        return np.vstack([extract_demo_summary_features(ep) for ep in episodes])

    def fit(
        self,
        train_episodes: Sequence[dict[str, Any]],
        y_train: Optional[np.ndarray] = None,
    ) -> "LengthHistogramBaseline":
        X = self.featurize(train_episodes)
        y = (
            np.asarray(y_train, dtype=np.float64)
            if y_train is not None
            else np.vstack([get_episode_target(ep) for ep in train_episodes])
        )
        self.clf.fit(X, y)
        return self

    def predict_proba(self, episodes: list[dict[str, Any]]) -> np.ndarray:
        X = self.featurize(episodes)
        return self.clf.predict_proba(X)


class QueryMaterialBaseline:
    """Multinomial logistic regression on query static material features.

    Features: [material_diff_stm, own_material, opp_material,
               piece_type_counts_diff(P, N, B, R, Q), is_white_to_move].
    """

    def __init__(self, l2_reg: float = 1e-4, max_iter: int = 200):
        self.clf = MultinomialLogisticRegression(l2_reg=l2_reg, max_iter=max_iter)

    def featurize(self, episodes: Sequence[dict[str, Any]]) -> np.ndarray:
        if len(episodes) == 0:
            return np.empty((0, 9), dtype=np.float64)
        return np.vstack([extract_query_material_features(ep) for ep in episodes])

    def fit(
        self,
        train_episodes: Sequence[dict[str, Any]],
        y_train: Optional[np.ndarray] = None,
    ) -> "QueryMaterialBaseline":
        X = self.featurize(train_episodes)
        y = (
            np.asarray(y_train, dtype=np.float64)
            if y_train is not None
            else np.vstack([get_episode_target(ep) for ep in train_episodes])
        )
        self.clf.fit(X, y)
        return self

    def predict_proba(self, episodes: list[dict[str, Any]]) -> np.ndarray:
        X = self.featurize(episodes)
        return self.clf.predict_proba(X)


class BoardKNNBaseline:
    """Nearest-neighbor board-state baseline using (64*13 + 1) encodings.

    Embeds demo states (all trajectory states or final states), finds top-k nearest
    demo states to the query board state by cosine or normalized Hamming similarity,
    computes similarity-weighted White-perspective W/D/L distribution, and converts
    to the query side-to-move perspective.
    """

    def __init__(
        self,
        top_k: int = 10,
        metric: Literal["cosine", "hamming"] = "cosine",
        state_mode: Literal["trajectories", "final"] = "trajectories",
        alpha: float = 0.5,
        prior: Optional[Sequence[float]] = None,
    ):
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if metric not in {"cosine", "hamming"}:
            raise ValueError(f"metric must be 'cosine' or 'hamming', got {metric!r}")
        if state_mode not in {"trajectories", "final"}:
            raise ValueError(f"state_mode must be 'trajectories' or 'final', got {state_mode!r}")

        self.top_k = int(top_k)
        self.metric = metric
        self.state_mode = state_mode
        self.alpha = float(alpha)
        if prior is None:
            self.prior = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
        else:
            p = np.asarray(prior, dtype=np.float64)
            self.prior = p / np.sum(p)

    def _compute_similarities(self, q_vec: np.ndarray, demo_vecs: np.ndarray) -> np.ndarray:
        """Compute similarities between query (D,) and demo_vecs (M, D)."""
        if self.metric == "cosine":
            q_norm = float(np.linalg.norm(q_vec)) + 1e-12
            d_norms = np.linalg.norm(demo_vecs, axis=1) + 1e-12
            sims = (demo_vecs @ q_vec) / (d_norms * q_norm)
            return np.clip(sims.astype(np.float64), 0.0, 1.0)
        # Normalized Hamming similarity for binary {0, 1} vectors: 1 - mean(|q - d|)
        dim = float(q_vec.shape[0])
        q_sum = float(np.sum(q_vec))
        d_sums = np.sum(demo_vecs, axis=1)
        l1_dists = q_sum + d_sums - 2.0 * (demo_vecs @ q_vec)
        sims = 1.0 - (l1_dists / dim)
        return np.clip(sims.astype(np.float64), 0.0, 1.0)

    def predict_proba(self, episodes: list[dict[str, Any]]) -> np.ndarray:
        n = len(episodes)
        if n == 0:
            return np.empty((0, 3), dtype=np.float64)

        out = np.empty((n, 3), dtype=np.float64)
        for i, ep in enumerate(episodes):
            demos = ep.get("demos", [])
            if not demos:
                out[i] = self.prior
                continue

            state_vecs: list[np.ndarray] = []
            state_outcomes: list[np.ndarray] = []
            for demo in demos:
                outcome_w = get_demo_outcome_white(demo)
                vecs = reconstruct_demo_states(demo, state_mode=self.state_mode)
                for v in vecs:
                    state_vecs.append(v)
                    state_outcomes.append(outcome_w)

            if not state_vecs:
                out[i] = self.prior
                continue

            demo_mat = np.vstack(state_vecs)  # (M, 833)
            outcome_mat = np.vstack(state_outcomes)  # (M, 3)
            q_vec = encode_fen_64x13_plus_1(get_query_fen(ep))

            sims = self._compute_similarities(q_vec, demo_mat)
            k_eff = min(self.top_k, len(sims))
            # Stable top-k selection (ties resolved deterministically by earlier index)
            top_idx = np.argsort(-sims, kind="stable")[:k_eff]
            top_sims = sims[top_idx]

            weights = np.maximum(top_sims, 0.0)
            w_sum = float(np.sum(weights))
            if w_sum <= 1e-12:
                weights = np.ones_like(weights)
                w_sum = float(len(weights))

            weighted_counts = weights @ outcome_mat[top_idx]
            p_white = (weighted_counts + self.alpha * 3.0 * self.prior) / (
                w_sum + 3.0 * self.alpha
            )
            p_white /= np.sum(p_white)

            side = get_query_side(ep)
            out[i] = white_to_stm_probs(p_white, side)
        return out


class BoardKNNTrajectoriesBaseline(BoardKNNBaseline):
    """Board-state kNN baseline indexing all states along demonstration trajectories."""

    def __init__(
        self,
        top_k: int = 10,
        metric: Literal["cosine", "hamming"] = "cosine",
        alpha: float = 0.5,
        prior: Optional[Sequence[float]] = None,
    ):
        super().__init__(
            top_k=top_k,
            metric=metric,
            state_mode="trajectories",
            alpha=alpha,
            prior=prior,
        )


class BoardKNNFinalBaseline(BoardKNNBaseline):
    """Board-state kNN baseline indexing only the final board state of each demonstration."""

    def __init__(
        self,
        top_k: int = 10,
        metric: Literal["cosine", "hamming"] = "cosine",
        alpha: float = 0.5,
        prior: Optional[Sequence[float]] = None,
    ):
        super().__init__(
            top_k=top_k,
            metric=metric,
            state_mode="final",
            alpha=alpha,
            prior=prior,
        )


DECLARED_BASELINE_NAMES: tuple[str, ...] = (
    "uniform",
    "training_prior",
    "demo_majority",
    "length_histogram",
    "query_material",
    "board_knn_trajectories",
    "board_knn_final",
)

CONTEXT_INVARIANT_BASELINE_NAMES: frozenset[str] = frozenset(
    {"uniform", "training_prior", "query_material"}
)


def build_declared_baselines(
    train_episodes: Sequence[dict[str, Any]],
    *,
    knn_top_k: int = 10,
    knn_metric: Literal["cosine", "hamming"] = "cosine",
    demo_alpha: float = 1.0,
    knn_alpha: float = 0.5,
    l2_reg: float = 1e-4,
    max_iter: int = 200,
) -> dict[str, Predictor]:
    """Instantiate and fit the 7 declared non-neural baselines strictly on `train_episodes`.

    Never reads validation or test splits. Parametric baselines (`training_prior`,
    `length_histogram`, `query_material`) are fitted exclusively on `train_episodes`, and
    the fitted training prior is passed as the fallback/smoothing prior to `demo_majority`,
    `board_knn_trajectories`, and `board_knn_final`.
    """
    if len(train_episodes) == 0:
        raise ValueError("train_episodes cannot be empty when fitting declared baselines")

    prior_model = TrainingPriorBaseline().fit(train_episodes)
    length_model = LengthHistogramBaseline(l2_reg=l2_reg, max_iter=max_iter).fit(train_episodes)
    material_model = QueryMaterialBaseline(l2_reg=l2_reg, max_iter=max_iter).fit(train_episodes)

    return {
        "uniform": UniformBaseline(),
        "training_prior": prior_model,
        "demo_majority": DemoMajorityBaseline(alpha=demo_alpha, prior=prior_model.prior_),
        "length_histogram": length_model,
        "query_material": material_model,
        "board_knn_trajectories": BoardKNNTrajectoriesBaseline(
            top_k=knn_top_k,
            metric=knn_metric,
            alpha=knn_alpha,
            prior=prior_model.prior_,
        ),
        "board_knn_final": BoardKNNFinalBaseline(
            top_k=knn_top_k,
            metric=knn_metric,
            alpha=knn_alpha,
            prior=prior_model.prior_,
        ),
    }


# ---------------------------------------------------------------------------
# Equal-Material Slicing Helper
# ---------------------------------------------------------------------------


def filter_equal_material(
    episodes: Sequence[dict[str, Any]],
    threshold: float = 2.0,
    return_indices: bool = False,
) -> Any:
    """Filter episodes to balanced-material positions: abs(standard_material_diff) <= threshold."""
    kept: list[dict[str, Any]] = []
    indices: list[int] = []
    for idx, ep in enumerate(episodes):
        diff = compute_standard_material_diff(get_query_fen(ep))
        if abs(diff) <= threshold:
            kept.append(ep)
            indices.append(idx)
    if return_indices:
        return kept, indices
    return kept


class EqualMaterialSlice:
    """Callable slice helper for filtering balanced-material query positions."""

    def __init__(self, threshold: float = 2.0):
        self.threshold = float(threshold)

    def filter(self, episodes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        return filter_equal_material(episodes, threshold=self.threshold, return_indices=False)

    def select_indices(self, episodes: Sequence[dict[str, Any]]) -> list[int]:
        _, indices = filter_equal_material(
            episodes, threshold=self.threshold, return_indices=True
        )
        return indices
