"""Data representations, vocabulary, and tokenization for chess ICL."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence, Union
import json

GameResult = Literal["win", "draw", "loss"]


@dataclass(frozen=True)
class Game:
    game_id: str
    variant: str
    moves: tuple[str, ...]
    result: GameResult
    policy: str = "heuristic"
    move_count: int = 0


@dataclass(frozen=True)
class Position:
    fen: str
    label: Optional[GameResult] = None
    source_game: Optional[str] = None
    move_number: Optional[int] = None


class ChessTokenizer:
    """Encode chess positions and games into integer token sequences."""

    # Special tokens
    PAD_TOKEN = "[PAD]"
    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"
    GAME_START_TOKEN = "[GAME_START]"
    GAME_END_TOKEN = "[GAME_END]"
    POS_TOKEN = "[POS]"
    WIN_TOKEN = "[WIN]"
    DRAW_TOKEN = "[DRAW]"
    LOSS_TOKEN = "[LOSS]"

    def __init__(self):
        self.vocab = self._build_vocab()
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.vocab_size = len(self.vocab)

        self.pad_id = self.token_to_id[self.PAD_TOKEN]
        self.cls_id = self.token_to_id[self.CLS_TOKEN]
        self.sep_id = self.token_to_id[self.SEP_TOKEN]
        self.game_start_id = self.token_to_id[self.GAME_START_TOKEN]
        self.game_end_id = self.token_to_id[self.GAME_END_TOKEN]
        self.pos_id = self.token_to_id[self.POS_TOKEN]
        self.win_id = self.token_to_id[self.WIN_TOKEN]
        self.draw_id = self.token_to_id[self.DRAW_TOKEN]
        self.loss_id = self.token_to_id[self.LOSS_TOKEN]

        self.result_token_map = {
            "win": self.WIN_TOKEN,
            "draw": self.DRAW_TOKEN,
            "loss": self.LOSS_TOKEN,
        }

    def _build_vocab(self) -> list[str]:
        tokens = [
            self.PAD_TOKEN,
            self.CLS_TOKEN,
            self.SEP_TOKEN,
            self.GAME_START_TOKEN,
            self.GAME_END_TOKEN,
            self.POS_TOKEN,
            self.WIN_TOKEN,
            self.DRAW_TOKEN,
            self.LOSS_TOKEN,
        ]

        # 64 chess squares: a1 to h8
        files = ["a", "b", "c", "d", "e", "f", "g", "h"]
        ranks = ["1", "2", "3", "4", "5", "6", "7", "8"]
        for r in ranks:
            for f in files:
                tokens.append(f"{f}{r}")

        # Piece characters
        pieces = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
        tokens.extend(pieces)

        # FEN specific characters
        fen_chars = ["/", "1", "2", "3", "4", "5", "6", "7", "8", "-", "w", "b"]
        for ch in fen_chars:
            if ch not in tokens:
                tokens.append(ch)

        return tokens

    def encode_fen(self, fen: str) -> list[int]:
        """Encode piece-placement + side-to-move from a FEN string."""
        parts = fen.split()
        placement = parts[0]
        side = parts[1] if len(parts) > 1 else "w"
        tokens = []
        for ch in placement:
            if ch in self.token_to_id:
                tokens.append(self.token_to_id[ch])
            else:
                continue
        if side in self.token_to_id:
            tokens.append(self.token_to_id[side])
        else:
            tokens.append(self.token_to_id["w"])
        return tokens

    def decode_fen(self, token_ids: list[int]) -> str:
        """Decode token IDs back to a FEN placement + side string."""
        filtered = [
            self.id_to_token[t]
            for t in token_ids
            if t in self.id_to_token
            and self.id_to_token[t] not in (
                self.PAD_TOKEN,
                self.CLS_TOKEN,
                self.SEP_TOKEN,
                self.POS_TOKEN,
            )
        ]
        if not filtered:
            return ""
        if filtered[-1] in ("w", "b") and len(filtered) > 1:
            return "".join(filtered[:-1]) + " " + filtered[-1]
        return "".join(filtered)

    def encode_move(self, uci_move: str) -> list[int]:
        """Encode a UCI move as from_square, to_square, and optional promotion."""
        if len(uci_move) < 4:
            return []
        from_sq = uci_move[:2]
        to_sq = uci_move[2:4]
        res = []
        if from_sq in self.token_to_id:
            res.append(self.token_to_id[from_sq])
        if to_sq in self.token_to_id:
            res.append(self.token_to_id[to_sq])
        if len(uci_move) >= 5:
            promo = uci_move[4]
            if promo in self.token_to_id:
                res.append(self.token_to_id[promo])
        return res

    def decode_move(self, token_ids: list[int]) -> str:
        """Decode token IDs into a UCI move string."""
        parts = [self.id_to_token.get(t, "") for t in token_ids]
        return "".join(parts)

    def encode_game(self, game: Union[dict, Game]) -> list[int]:
        """Encode a game: [GAME_START] move1 move2 ... [result] [GAME_END]."""
        if isinstance(game, Game):
            moves = game.moves
            result = game.result
        else:
            moves = game.get("moves", [])
            result = game.get("result", "draw")

        tokens = [self.game_start_id]
        for move in moves:
            tokens.extend(self.encode_move(move))

        result_token_str = self.result_token_map.get(result, self.DRAW_TOKEN)
        tokens.append(self.token_to_id[result_token_str])
        tokens.append(self.game_end_id)
        return tokens

    def encode_input(self, demos: Sequence[Union[dict, Game]], query_fen: str) -> list[int]:
        """Build full input sequence: [CLS] + K demo games + [POS] query [SEP]."""
        tokens = [self.cls_id]
        for game in demos:
            tokens.extend(self.encode_game(game))
        tokens.append(self.pos_id)
        tokens.extend(self.encode_fen(query_fen))
        tokens.append(self.sep_id)
        return tokens


def load_games(path: Union[str, Path]) -> list[dict]:
    """Load games from a JSONL file."""
    games = []
    p = Path(path)
    if not p.exists():
        return games
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                games.append(json.loads(line))
    return games


def load_positions(path: Union[str, Path]) -> list[dict]:
    """Load positions from a JSONL file."""
    positions = []
    p = Path(path)
    if not p.exists():
        return positions
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                positions.append(json.loads(line))
    return positions
