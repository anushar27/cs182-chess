from copy import deepcopy
from pathlib import Path
import numpy as np
import chess
import time
from .common import digest, provenance
from .data import audit_dataset, states_in_game
from .episodes import build_episode
from .rollout import continuation, estimate
from .state import State, is_near


def extend(data, max_plies, workers=1):
    extension_start = time.perf_counter()
    extended = deepcopy(data)
    for game in extended['demonstrations']:
        if game['status'] == 'unresolved':
            if max_plies <= game['max_plies']:
                raise ValueError('Extension must increase the budget')
            old = list(game['moves'])
            run = continuation(State.from_dict(game['initial']), game['policy_id'], game['seed'], max_plies)
            if run['moves'][:len(old)] != old:
                raise ValueError('Extension changed the sampled trajectory')
            game.update(run)
    master = data['provenance']['config']['seed']
    for query in extended['queries']:
        label = query['label']
        if label['unresolved']:
            if any(max_plies <= r['max_plies'] for r in label['runs']):
                raise ValueError('Extension must increase the budget')
            new = estimate(State.from_dict(query['state']), label['policy_id'], label['replicates'], max_plies, master, workers)
            for a,b in zip(label['runs'],new['runs']):
                if a['moves'] != b['moves'][:len(a['moves'])]:
                    raise ValueError('Extension changed a label continuation')
            query['label'] = new
    extended.setdefault('extensions', []).append({'parent_dataset_sha256': digest(data),
        'provenance': provenance({'max_plies': max_plies, 'workers': workers}),
        'convention': 'same random streams, fixed roots, no resampling'})
    # Full later demo trajectories can newly collide with fixed query roots.
    # Retain raw excluded games, remove entire demo families, never alter a root or label.
    roots = [(q['id'], State.from_dict(q['state'])) for q in extended['queries']]
    threshold = data['provenance']['config']['near_distance']
    kept = []
    for game in extended['demonstrations']:
        conflict = None
        for state in states_in_game(game):
            conflict = next((qid for qid, root in roots if is_near(root, state, threshold)), None)
            if conflict is not None:
                break
        if conflict is None:
            kept.append(game)
        else:
            extended.setdefault('excluded_demonstrations', []).append(game)
            extended['exclusions'].append({'family_id':game['family_id'], 'query_id':conflict,
                                           'reason':'extended_demo_near_query'})
    extended['demonstrations'] = kept
    extended['integrity'] = audit_dataset(extended)
    elapsed = time.perf_counter() - extension_start
    extended['extensions'][-1]['elapsed_seconds'] = elapsed
    extended['total_generation_seconds'] = data.get('total_generation_seconds', data.get('generation_seconds', 0)) + elapsed
    return extended


def token_lengths(data, tokenizer, model_config=None):
    rows = []
    for g in data['demonstrations']:
        board = State.from_dict(g['initial']).board()
        # SAN for custom attacks is not supplied by python-chess. Only measure reference variants.
        san = [] if g['variant'] in ('standard', 'antichess') else None
        if san is not None:
            for uci in g['moves']:
                move = chess.Move.from_uci(uci)
                san.append(board.san(move)); board.push(move)
        enc = lambda x: len(tokenizer.encode(x, add_special_tokens=False))
        rows.append({'variant': g['variant'], 'family_id': g['family_id'], 'split': g['split'],
                     'status': g['status'], 'plies': len(g['moves']),
                     'uci_move_tokens': enc(' '.join(g['moves'])),
                     'san_move_tokens': enc(' '.join(san)) if san is not None else None})
    episodes = []
    for q in data['queries']:
        for k in (0,1,2,4,8,16):
            try:
                ep = build_episode(q, data['demonstrations'], tokenizer, k, 1000000, 'relevant' if k else 'zero')
                episodes.append({'query_id': q['id'], 'variant': q['variant'], 'k': k, 'tokens': len(ep.input_ids)})
            except ValueError as e:
                episodes.append({'query_id': q['id'], 'variant': q['variant'], 'k': k, 'unavailable': str(e)})
    return {'tokenizer_class': type(tokenizer).__name__, 'model_config': model_config,
            'games': rows, 'episodes': episodes, 'notation_decision': 'UCI v1; SAN comparison restricted to supported reference rules'}


def noise_intervals(label, seed=0, samples=10000):
    if label['unresolved']:
        return {'identification_bounds': label['bounds_wdl'], 'posterior': None}
    # Jeffreys multinomial posterior; intervals are explicitly Bayesian, not zero plug-in SE at boundaries.
    posterior = np.random.default_rng(seed).dirichlet(np.array(label['counts_wdl']) + .5, size=samples)
    return {'jeffreys_marginal_ci95': np.quantile(posterior, [.025,.975], axis=0).T.tolist()}
