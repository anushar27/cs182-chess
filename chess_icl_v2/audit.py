import random
from pathlib import Path
import chess
from .common import digest, provenance, seed_for
from .rules import RULES, make_board, outcome
from .state import capture, State

REFERENCE_VARIANTS = ['diagonal_rook', 'straight_bishop', 'weakened_queen', 'short_knight',
                      'mobile_king', 'extended_pawn', 'forward_pawn_capture']


def run_audits(seed=42, games=3, plies=80, use_pyffish=True):
    sf = None
    if use_pyffish:
        import pyffish as sf
        sf.load_variant_config((Path(__file__).parent / 'reference.ini').read_text())
    report = {'provenance': provenance({'seed': seed, 'games': games, 'plies': plies}),
              'reference_version': list(sf.version()) if sf else None,
              'reference_config_sha256': digest((Path(__file__).parent / 'reference.ini').read_text()) if sf else None, 'variants': {}}
    for name in RULES:
        checks, compared, terminal_checks, errors = 0, 0, 0, []
        for index in range(games):
            rng = random.Random(seed_for(seed, name, index))
            b = make_board(name, seed=seed_for(seed, name, index))
            initial_fen = b.fen(en_passant='fen')
            history = []
            for ply in range(plies):
                state = capture(name, b)
                restored = State.from_dict(state.to_dict()).board()
                legal = {m.uci() for m in b.legal_moves}
                if {m.uci() for m in restored.legal_moves} != legal or restored.fen(en_passant='fen') != b.fen(en_passant='fen') or restored.outcome() != b.outcome():
                    errors.append({'ply': ply, 'kind': 'state_roundtrip'})
                if name != 'dark_square_immunity':
                    mirror = state.mirror().board()
                    expected_mirror = {chess.Move(chess.square_mirror(m.from_square), chess.square_mirror(m.to_square), promotion=m.promotion).uci() for m in b.legal_moves}
                    if {m.uci() for m in mirror.legal_moves} != expected_mirror:
                        errors.append({'ply': ply, 'kind': 'perspective_mirror'})
                reference = 'icl_' + name if name in REFERENCE_VARIANTS else ('chess' if name in ('standard', 'knight_army') else ('antichess' if name == 'antichess' else None))
                if sf is not None and reference:
                    ref_moves = set(sf.legal_moves(reference, initial_fen, history))
                    compared += 1
                    if legal != ref_moves:
                        errors.append({'ply': ply, 'kind': 'legal_moves', 'fen': b.fen(en_passant='fen'),
                                       'only_ours': sorted(legal - ref_moves), 'only_reference': sorted(ref_moves - legal),
                                       'initial_fen': initial_fen, 'history': list(history)})
                        break
                    # Compare checkmate/stalemate result independently of optional draw conventions.
                    if not legal:
                        result = outcome(b)
                        relative = 0 if result.winner is None else (1 if result.winner == b.turn else -1)
                        reference_result = sf.game_result(reference, initial_fen, history)
                        if relative != (0 if reference_result == 0 else (1 if reference_result > 0 else -1)):
                            errors.append({'ply': ply, 'kind': 'terminal_result'})
                        terminal_checks += 1
                checks += 1
                if b.is_game_over() or not legal:
                    break
                uci = rng.choice(sorted(legal))
                before = state.to_dict()
                b.push_uci(uci)
                after = capture(name, b).to_dict()
                b.pop()
                if capture(name, b).to_dict() != before:
                    errors.append({'ply': ply, 'kind': 'push_pop'})
                b.push_uci(uci)
                if capture(name, b).to_dict() != after:
                    errors.append({'ply': ply, 'kind': 'replay'})
                history.append(uci)
        report['variants'][name] = {'passed': not errors, 'reachable_positions': checks,
            'independent_move_comparisons': compared, 'independent_terminal_comparisons': terminal_checks,
            'reference': 'pyffish' if compared else 'golden/property tests required; no independent engine',
            'errors': errors}
    report['passed'] = all(v['passed'] for v in report['variants'].values())
    return report
