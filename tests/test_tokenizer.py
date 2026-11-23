"""Unit tests for tokenization and dataset encoding."""

import pytest
from chess_icl.data import ChessTokenizer, Game


def test_vocab_size():
    tokenizer = ChessTokenizer()
    assert tokenizer.vocab_size > 90
    assert tokenizer.pad_id is not None
    assert tokenizer.cls_id is not None
    assert tokenizer.sep_id is not None


def test_fen_roundtrip():
    tokenizer = ChessTokenizer()
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b"
    tokens = tokenizer.encode_fen(fen)
    assert len(tokens) > 0
    decoded = tokenizer.decode_fen(tokens)
    assert decoded.strip() == fen.strip()


def test_move_roundtrip():
    tokenizer = ChessTokenizer()
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "e7e8q"]
    for m in moves:
        encoded = tokenizer.encode_move(m)
        decoded = tokenizer.decode_move(encoded)
        assert decoded == m


def test_game_encoding():
    tokenizer = ChessTokenizer()
    game = {
        "game_id": "test_01",
        "moves": ["e2e4", "e7e5", "g1f3"],
        "result": "win",
    }
    tokens = tokenizer.encode_game(game)
    assert tokens[0] == tokenizer.game_start_id
    assert tokens[-1] == tokenizer.game_end_id
    assert tokens[-2] == tokenizer.win_id


def test_input_sequence_structure():
    tokenizer = ChessTokenizer()
    demos = [
        {"moves": ["e2e4", "e7e5"], "result": "win"},
        {"moves": ["d2d4", "d7d5"], "result": "loss"},
    ]
    query_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w"

    # K=0 demos
    tokens_k0 = tokenizer.encode_input([], query_fen)
    assert tokens_k0[0] == tokenizer.cls_id
    assert tokens_k0[-1] == tokenizer.sep_id
    assert tokenizer.pos_id in tokens_k0

    # K=2 demos
    tokens_k2 = tokenizer.encode_input(demos, query_fen)
    assert tokens_k2[0] == tokenizer.cls_id
    assert tokens_k2[-1] == tokenizer.sep_id
    assert tokens_k2.count(tokenizer.game_start_id) == 2
    assert tokens_k2.count(tokenizer.game_end_id) == 2
