import random
import chess
import pytest
from chess_icl_v2.rules import RULES, make_board, outcome
from chess_icl_v2.state import State, capture
from chess_icl_v2.rollout import continuation, estimate
from chess_icl_v2.policy import choose


def legal(b): return {m.uci() for m in b.legal_moves}


@pytest.mark.parametrize('name', list(RULES))
def test_replay_save_load_copy_pop(name):
    b = make_board(name, seed=231)
    rng = random.Random(3)
    for _ in range(12):
        s = capture(name, b)
        assert s.to_dict() == capture(name, s.board()).to_dict()
        assert legal(b) == legal(s.board())
        assert legal(b.copy()) == legal(b)
        if b.is_game_over(): break
        before = capture(name, b).to_dict()
        move = rng.choice(sorted(b.legal_moves, key=lambda m: m.uci()))
        b.push(move); b.pop()
        assert capture(name, b).to_dict() == before
        b.push(move)


def test_variant_attack_detection():
    b = make_board('diagonal_rook', '7k/8/8/8/3r4/8/8/K7 w - - 0 1')
    assert b.is_check()
    assert 'a1b2' not in legal(b)
    b = make_board('diagonal_rook', '7k/8/8/8/r7/8/8/K7 w - - 0 1')
    assert not b.is_check()
    b = make_board('straight_bishop', '7k/8/8/8/b7/8/8/K7 w - - 0 1')
    assert b.is_check()
    b = make_board('weakened_queen', '7k/8/8/8/3q4/8/8/K7 w - - 0 1')
    assert not b.is_check()
    b = make_board('short_knight', '7k/8/8/8/8/8/2n5/4K3 w - - 0 1')
    assert not b.is_check()  # c2 to e1 is an orthodox knight, not FD.
    b = make_board('short_knight', '7k/8/8/8/8/8/8/2n1K3 w - - 0 1')
    assert b.is_check()


def test_knight_army_fen_is_not_overwritten():
    b = make_board('knight_army')
    assert len(b.pieces(chess.KNIGHT, chess.WHITE)) == 15
    fen = '7k/8/8/8/8/8/4N3/K7 b - - 12 19'
    b = make_board('knight_army', fen)
    assert b.fen(en_passant='fen') == fen


def test_barriers_block_attacks_pawn_paths_and_castling():
    b = make_board('blocked_squares', 'r6k/8/8/8/8/8/8/K7 w - - 0 1', extras={'barriers': [chess.A4]})
    assert not b.is_check()
    assert 'a1a2' in legal(b)
    b = make_board('blocked_squares', extras={'barriers': [chess.E3]})
    assert not {'e2e3', 'e2e4'} & legal(b)
    assert capture('blocked_squares', b).board().barriers == b.barriers
    b = make_board('blocked_squares', '4k3/8/8/8/8/8/8/4K2R w K - 0 1', extras={'barriers': [chess.F1]})
    assert 'e1g1' not in legal(b)


def test_conversion_preserves_history_and_counters():
    b = make_board('piece_capture', '7k/8/8/3r4/4Q3/8/8/K7 w - - 16 7')
    before = b.fen(en_passant='fen')
    b.push_uci('e4d5')
    assert b.piece_at(chess.D5) == chess.Piece(chess.ROOK, chess.WHITE)
    assert b.halfmove_clock == 0 and len(b.move_stack) == 1 and not b.turn
    assert capture('piece_capture', b).board().fen() == b.fen()
    b.pop()
    assert b.fen(en_passant='fen') == before


def test_extended_pawn_ep_and_promotions():
    b = make_board('extended_pawn', '7k/8/8/8/3p4/4P3/8/K7 w - - 0 1')
    b.push_uci('e3e5')
    assert b.ep_square == chess.E4
    assert 'd4e4' not in legal(b)
    # Adjacent enemy pawn at d5 can take on e4 after an off-start two-step.
    b = make_board('extended_pawn', '7k/8/8/3p4/8/4P3/8/K7 w - - 0 1')
    b.push_uci('e3e5')
    assert 'd5e4' in legal(b)
    b.push_uci('d5e4')
    assert b.piece_at(chess.E5) is None
    b = make_board('extended_pawn', '7k/8/4P3/8/8/8/8/K7 w - - 0 1')
    assert {f'e6e8{p}' for p in 'qrbn'} <= legal(b)


def test_immunity_and_forward_pawn_checks():
    b = make_board('dark_square_immunity', '7k/8/8/8/8/2n5/1P6/K7 w - - 0 1')
    assert 'b2c3' not in legal(b)
    b = make_board('dark_square_immunity', '7k/8/8/3Pp3/8/8/8/K7 w - e6 0 1')
    assert 'd5e6' not in legal(b)  # Actual victim is on immune e5.
    with pytest.raises(ValueError, match='not a symmetry'):
        capture('dark_square_immunity', b).mirror()
    b = make_board('forward_pawn_capture', '7k/8/8/8/4p3/4K3/8/8 w - - 0 1')
    assert b.is_check()


def test_mobile_king_slide_is_not_implicit_castling():
    b = make_board('mobile_king', '4k3/8/8/8/8/8/8/4K2R w - - 0 1')
    b.push_uci('e1g1')
    assert b.king(chess.WHITE) == chess.G1
    assert b.piece_at(chess.H1) == chess.Piece(chess.ROOK, chess.WHITE)
    assert b.piece_at(chess.F1) is None


def test_fivefold_history_survives_fen_roundtrip():
    b = make_board('standard')
    for _ in range(4):
        for move in ('g1f3','g8f6','f3g1','f6g8'): b.push_uci(move)
    s = capture('standard', b)
    assert s.board().is_fivefold_repetition()
    assert not chess.Board(b.fen()).is_fivefold_repetition()
    assert s.key != State('standard', s.rules_version, b.fen()).key
    assert s.mirror().board().is_fivefold_repetition()


def test_antichess_mandatory_capture_no_royalty_and_stalemate():
    b = make_board('antichess', '8/8/8/8/8/1p6/P7/8 w - - 0 1')
    assert legal(b) == {'a2b3'}
    b.push_uci('a2b3')
    assert outcome(b).winner == chess.BLACK
    b = make_board('antichess', '8/8/8/8/8/8/p7/P7 w - - 0 1')
    assert outcome(b).winner == chess.WHITE


def test_rollout_reproducibility_and_censoring():
    s = capture('standard', make_board('standard'))
    a = estimate(s, 'uniform_legal_v1', 3, 2, 8)
    b = estimate(s, 'uniform_legal_v1', 3, 2, 8, workers=2)
    assert [r['moves'] for r in a['runs']] == [r['moves'] for r in b['runs']]
    assert a['target_wdl'] is None and a['unresolved'] == 3
    assert a['bounds_wdl'] == [[0.,1.]] * 3
    assert len({r['seed'] for r in a['runs']}) == 3
    longer = continuation(s, 'uniform_legal_v1', a['runs'][0]['seed'], 4)
    assert longer['moves'][:2] == a['runs'][0]['moves']


def test_terminal_aware_policy_and_perspective():
    b = make_board('standard', '7k/5Q2/6K1/8/8/8/8/8 w - - 0 1')
    s = capture('standard', b)
    result = continuation(s, 'terminal_aware_v1', 4, 1)
    assert result['root_result'] == 'W'
    mirror = continuation(s.mirror(), 'terminal_aware_v1', 4, 1)
    assert mirror['root_result'] == 'W' and mirror['white_result'] == 'L'
    assert len(b.move_stack) == 0


def test_mobile_king_corner_slide_preserves_uci_history():
    b = make_board('mobile_king', '4K3/4b3/1r1P4/6p1/6k1/2p5/8/8 w - - 1 111')
    b.push_uci('e8a8')
    assert b.king(chess.WHITE) == chess.A8
    assert b.move_stack[-1].uci() == 'e8a8'
    assert capture('mobile_king', b).board().fen() == b.fen()


def test_promoted_flags_survive_arbitrary_initial_state_roundtrip():
    b=make_board('standard','7k/8/8/8/8/8/Q~7/K7 w - - 0 1')
    s=capture('standard',b)
    assert '~' in s.initial_fen and s.board().promoted==b.promoted
    assert s.mirror().board().promoted==chess.flip_vertical(b.promoted)
    from chess_icl_v2.tokenizer import SquareMoveTokenizer
    with pytest.raises(ValueError,match='unpromoted initial board'):
        SquareMoveTokenizer().state(s.public())


@pytest.mark.parametrize('piece','pnbrq')
def test_conversion_king_captures_do_not_disappear_or_masquerade_as_castling(piece):
    b=make_board('piece_capture',f'7k/8/8/8/8/8/4{piece}3/4K3 w - - 0 1')
    assert 'e1e2' not in legal(b)
    before=capture('piece_capture',b).to_dict()
    list(b.legal_moves)
    assert capture('piece_capture',b).to_dict()==before


def test_conversion_castling_is_not_a_capture():
    b=make_board('piece_capture','4k3/8/8/8/8/8/8/4K2R w K - 0 1')
    assert 'e1g1' in legal(b)
    b.push_uci('e1g1')
    assert b.king(chess.WHITE)==chess.G1 and b.piece_at(chess.F1)==chess.Piece(chess.ROOK,chess.WHITE)
