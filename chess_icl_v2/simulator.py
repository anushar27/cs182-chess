"""Rule-correct move and attack generation sharing python-chess state bookkeeping.

Uses pinned python-chess bitboard/private APIs; upgrade only with the audit suite.
"""
import random
import chess

ORTH = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))
SHORT = DIAG + ((2, 0), (-2, 0), (0, 2), (0, -2))
SHORT_MASK = []
for square in chess.SQUARES:
    rank, file = chess.square_rank(square), chess.square_file(square)
    SHORT_MASK.append(sum(chess.BB_SQUARES[chess.square(file + df, rank + dr)]
                          for dr, df in SHORT if 0 <= rank + dr < 8 and 0 <= file + df < 8))


class VariantBoard(chess.Board):
    rule_name = "standard"

    def __init__(self, fen=chess.STARTING_FEN, *, chess960=False, barriers=()):
        if chess960:
            raise ValueError("Chess960 is outside this frozen protocol")
        if any(not isinstance(s, int) or not 0 <= s < 64 for s in barriers):
            raise ValueError("Invalid barrier square")
        self.barriers = frozenset(barriers)
        super().__init__(fen, chess960=False)
        if self.occupied & self.barrier_mask:
            raise ValueError("A barrier cannot overlap a piece")

    @property
    def barrier_mask(self):
        return sum(chess.BB_SQUARES[s] for s in self.barriers)

    def copy(self, *, stack=True):
        b = super().copy(stack=stack)
        b.barriers = self.barriers
        return b

    def root(self):
        b = super().root()
        b.barriers = self.barriers
        return b

    def mirror(self):
        b = super().mirror()
        b.barriers = frozenset(chess.square_mirror(s) for s in self.barriers)
        return b

    def clean_castling_rights(self):
        return 0 if self.rule_name in ("diagonal_rook", "mobile_king") else super().clean_castling_rights()

    def _transposition_key(self):
        return super()._transposition_key(), tuple(sorted(self.barriers)), self.rule_name

    def attackers_mask(self, color, square, occupied=None):
        if self.rule_name == "dark_square_immunity" and chess.BB_SQUARES[square] & chess.BB_DARK_SQUARES:
            return 0  # A royal king on a dark square is also immune.
        occupied = (self.occupied if occupied is None else occupied) | self.barrier_mask
        if square in self.barriers:
            return 0
        rookers, bishops = self.rooks | self.queens, self.bishops | self.queens
        if self.rule_name == "diagonal_rook":
            rookers &= ~self.rooks
            bishops |= self.rooks
        if self.rule_name == "straight_bishop":
            bishops &= ~self.bishops
            rookers |= self.bishops
        if self.rule_name == "weakened_queen":
            bishops &= ~self.queens
        if self.rule_name == "mobile_king":
            rookers |= self.kings
            bishops |= self.kings
        orth = (chess.BB_RANK_ATTACKS[square][chess.BB_RANK_MASKS[square] & occupied] |
                chess.BB_FILE_ATTACKS[square][chess.BB_FILE_MASKS[square] & occupied])
        diag = chess.BB_DIAG_ATTACKS[square][chess.BB_DIAG_MASKS[square] & occupied]
        knight = SHORT_MASK[square] if self.rule_name == "short_knight" else chess.BB_KNIGHT_ATTACKS[square]
        pawns = chess.BB_PAWN_ATTACKS[not color][square]
        if self.rule_name == "forward_pawn_capture":
            behind = square - 8 if color else square + 8
            if 0 <= behind < 64:
                pawns |= chess.BB_SQUARES[behind]
        attacks = (orth & rookers) | (diag & bishops) | (knight & self.knights) | (pawns & self.pawns)
        if self.rule_name != "mobile_king":
            attacks |= chess.BB_KING_ATTACKS[square] & self.kings
        return attacks & self.occupied_co[color]

    def attacks_mask(self, square):
        piece = self.piece_type_at(square)
        occupied = self.occupied | self.barrier_mask
        orth = (chess.BB_RANK_ATTACKS[square][chess.BB_RANK_MASKS[square] & occupied] |
                chess.BB_FILE_ATTACKS[square][chess.BB_FILE_MASKS[square] & occupied])
        diag = chess.BB_DIAG_ATTACKS[square][chess.BB_DIAG_MASKS[square] & occupied]
        if piece == chess.ROOK:
            attacks = diag if self.rule_name == "diagonal_rook" else orth
        elif piece == chess.BISHOP:
            attacks = orth if self.rule_name == "straight_bishop" else diag
        elif piece == chess.QUEEN:
            attacks = orth if self.rule_name == "weakened_queen" else orth | diag
        elif piece == chess.KNIGHT:
            attacks = SHORT_MASK[square] if self.rule_name == "short_knight" else chess.BB_KNIGHT_ATTACKS[square]
        elif piece == chess.KING:
            attacks = orth | diag if self.rule_name == "mobile_king" else chess.BB_KING_ATTACKS[square]
        elif piece == chess.PAWN:
            color = bool(self.occupied_co[chess.WHITE] & chess.BB_SQUARES[square])
            attacks = chess.BB_PAWN_ATTACKS[color][square]
            ahead = square + (8 if color else -8)
            if self.rule_name == "forward_pawn_capture" and 0 <= ahead < 64:
                attacks |= chess.BB_SQUARES[ahead]
        else:
            return 0
        attacks &= ~self.barrier_mask
        if self.rule_name == "dark_square_immunity":
            attacks &= ~chess.BB_DARK_SQUARES
        return attacks

    def _pawn_moves(self, square):
        step = 8 if self.turn else -8
        rank = chess.square_rank(square)
        start_rank, last_rank = (1, 7) if self.turn else (6, 0)
        max_push = 2 if rank == start_rank or self.rule_name == "extended_pawn" else 1
        if self.rule_name == "extended_initial_pawn" and rank == start_rank:
            max_push = 6
        destinations = []
        for distance in range(1, max_push + 1):
            target = square + step * distance
            if not 0 <= target < 64 or target in self.barriers or self.piece_at(target):
                break
            # Approximation preserves the old no-multi-push-promotion convention.
            if self.rule_name == "extended_initial_pawn" and distance > 1 and chess.square_rank(target) == last_rank:
                break
            destinations.append(target)
            if chess.square_rank(target) == last_rank:
                break
        attacks = self.attacks_mask(square) & self.occupied_co[not self.turn]
        # No king captures in royal variants.
        destinations += list(chess.scan_forward(attacks & ~self.kings))
        if self.ep_square is not None and self.ep_square not in self.barriers:
            # En passant is diagonal, even with forward captures.
            if chess.BB_PAWN_ATTACKS[self.turn][square] & chess.BB_SQUARES[self.ep_square]:
                captured_square = self.ep_square - step
                if self.piece_at(captured_square) == chess.Piece(chess.PAWN, not self.turn):
                    if not (self.rule_name == "dark_square_immunity" and
                            chess.BB_SQUARES[captured_square] & chess.BB_DARK_SQUARES):
                        destinations.append(self.ep_square)
        for target in destinations:
            promotions = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT) if chess.square_rank(target) == last_rank else (None,)
            for promotion in promotions:
                yield chess.Move(square, target, promotion=promotion)

    def generate_pseudo_legal_moves(self, from_mask=chess.BB_ALL, to_mask=chess.BB_ALL):
        ours = self.occupied_co[self.turn]
        for square in chess.scan_forward(ours & from_mask):
            if self.piece_type_at(square) == chess.PAWN:
                for move in self._pawn_moves(square):
                    if chess.BB_SQUARES[move.to_square] & to_mask:
                        yield move
            else:
                attacks = self.attacks_mask(square) & ~ours & ~self.kings & to_mask
                # Immunity forbids captures on dark squares, but allows quiet moves there.
                if self.rule_name == "dark_square_immunity":
                    piece = self.piece_type_at(square)
                    original = chess.Board.attacks_mask(self, square)
                    attacks |= original & ~self.occupied & ~self.barrier_mask & to_mask
                for target in chess.scan_forward(attacks):
                    yield chess.Move(square, target)
        yield from self.generate_castling_moves(from_mask, to_mask)

    def _to_chess960(self, move):
        return move if self.rule_name == "mobile_king" else super()._to_chess960(move)

    def _from_chess960(self, chess960, from_square, to_square, promotion=None, drop=None):
        if self.rule_name == "mobile_king":
            return chess.Move(from_square, to_square, promotion, drop)
        return super()._from_chess960(chess960, from_square, to_square, promotion, drop)

    def generate_castling_moves(self, from_mask=chess.BB_ALL, to_mask=chess.BB_ALL):
        if self.rule_name in ("mobile_king", "diagonal_rook"):
            return
        for move in super().generate_castling_moves(from_mask, to_mask):
            rank = chess.square_rank(move.from_square)
            rook_from = chess.square(7 if move.to_square > move.from_square else 0, rank)
            rook_to = chess.square(5 if move.to_square > move.from_square else 3, rank)
            path = (chess.between(move.from_square, move.to_square) | chess.BB_SQUARES[move.to_square] |
                    chess.between(rook_from, rook_to) | chess.BB_SQUARES[rook_to])
            if not path & self.barrier_mask:
                yield move

    def generate_legal_moves(self, from_mask=chess.BB_ALL, to_mask=chess.BB_ALL):
        color = self.turn
        if self.king(color) is None or self.king(not color) is None:
            return
        # Materialize: push changes self.turn during this generator.
        for move in list(self.generate_pseudo_legal_moves(from_mask, to_mask)):
            self.push(move)
            king = self.king(color)
            legal = king is not None and not self.is_attacked_by(not color, king)
            self.pop()
            if legal:
                yield move

    def generate_legal_ep(self, from_mask=chess.BB_ALL, to_mask=chess.BB_ALL):
        if self.ep_square is not None:
            for move in self.generate_legal_moves(from_mask, to_mask & chess.BB_SQUARES[self.ep_square]):
                if self.is_en_passant(move):
                    yield move

    def is_legal(self, move):
        return move in self.generate_legal_moves()

    def is_pseudo_legal(self, move):
        return move in self.generate_pseudo_legal_moves()

    def push(self, move):
        piece = self.piece_type_at(move.from_square)
        color = self.turn
        ep = self.is_en_passant(move)
        target = self.piece_type_at(move.to_square)
        castling_target = piece == chess.KING and target == chess.ROOK and self.color_at(move.to_square) == color
        captured = chess.PAWN if ep else target
        super().push(move)
        if self.rule_name == "extended_pawn" and piece == chess.PAWN and abs(move.to_square - move.from_square) == 16:
            self.ep_square = (move.to_square + move.from_square) // 2
        if self.rule_name == "piece_capture" and captured is not None and captured != chess.KING and not castling_target:
            # Keep the pre-push stack; public set_piece_at would clear it.
            self._set_piece_at(move.to_square, captured, color, False)
            self.halfmove_clock = 0

    def is_insufficient_material(self):
        # Orthodox heuristics are unsound when movement/capture rules change.
        # Knight Army changes setup only and therefore retains standard draws.
        return super().is_insufficient_material() if self.rule_name == "knight_army" else False


NAMES = ("diagonal_rook", "straight_bishop", "weakened_queen", "extended_pawn",
         "short_knight", "mobile_king", "piece_capture", "forward_pawn_capture",
         "dark_square_immunity", "blocked_squares", "extended_initial_pawn", "knight_army")
CLASSES = {}
for name in NAMES:
    cls_name = ''.join(s.title() for s in name.split('_')) + 'Board'
    cls = type(cls_name, (VariantBoard,), {"rule_name": name, "__module__": __name__})
    globals()[cls_name] = cls
    CLASSES[name] = cls

KNIGHT_ARMY_FEN = "nnnnknnn/nnnnnnnn/8/8/8/8/NNNNNNNN/NNNNKNNN w - - 0 1"


def custom_board(name, fen=None, *, barriers=None, seed=0):
    if name == "knight_army" and fen is None:
        fen = KNIGHT_ARMY_FEN
    fen = chess.STARTING_FEN if fen is None else fen
    if name == "blocked_squares" and barriers is None:
        b = chess.Board(fen)
        empty = [s for s in chess.SQUARES if not b.piece_at(s)]
        barriers = random.Random(seed).sample(empty, min(8, len(empty)))
    return CLASSES[name](fen, barriers=barriers or ())
