"""Rule-blind fixed-square/from-to vocabulary for scratch models (337 tokens)."""
import json
import re
import chess


class SquareMoveTokenizer:
    vocab_size = 337
    pad_token_id = 0
    pieces = '.PNBRQKpnbrqk'

    def state(self, value):
        if chess.Board(value["initial_fen"]).promoted:
            raise ValueError("Scratch vocabulary v1 requires an unpromoted initial board; preserve these flags with the public FEN serializer")
        tokens = [7]
        for key in ('initial_fen', 'current_fen'):
            b = chess.Board(value[key])
            tokens += [16 + self.pieces.index(b.piece_at(s).symbol() if b.piece_at(s) else '.')
                       for s in chess.SQUARES]
            tokens += [30 if b.turn else 31]
            rights = b.castling_xfen()
            tokens += [32 + sum((1 << i) for i, c in enumerate('KQkq') if c in rights)]
            tokens += [48 + (b.ep_square if b.ep_square is not None else 64)]
            for count in (b.halfmove_clock, b.fullmove_number):
                tokens += [120 + int(d) for d in str(count)] + [130]
        tokens += [6] + self.moves(value['history_uci'])
        tokens += [131] + [140 + s for s in value.get('extras', {}).get('barriers', [])] + [132]
        return tokens

    @staticmethod
    def moves(moves):
        tokens = []
        for uci in moves:
            m = chess.Move.from_uci(uci)
            tokens += [204 + m.from_square, 268 + m.to_square]
            if m.promotion:
                tokens += [332 + m.promotion - 1]
        return tokens

    def encode(self, text, **kwargs):
        tokens = []
        while text.startswith('<DEMO>\n'):
            body, text = text[len('<DEMO>\n'):].split('\n</DEMO>\n', 1)
            demo = json.loads(body)
            tokens += [1] + self.state(demo['initial']) + [5] + self.moves(demo['moves_uci'])
            tokens += [{'W': 9, 'D': 10, 'L': 11}[demo['white_outcome']], 4]
        if not text.startswith('<QUERY>\n') or not text.endswith('\n</QUERY>\n<PREDICT_WDL>'):
            raise ValueError('Unknown episode syntax')
        body = text[len('<QUERY>\n'):-len('\n</QUERY>\n<PREDICT_WDL>')]
        return tokens + [2] + self.state(json.loads(body)) + [4, 3]
