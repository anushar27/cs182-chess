"""Versioned, resumable large fixed-policy datasets with complete trace shards.

The root cohort is fixed before any labels are generated. Fixed source-family
quotas, global conservative geometry checks, and all administrative extensions
are recorded. No game is replaced because its outcome is inconvenient.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
import gzip
import hashlib
import json
from itertools import zip_longest
import multiprocessing
from pathlib import Path
import random
import time

import chess

from .common import canonical, digest, provenance, read_json, seed_for, write_json
from .data import SPLITS, family_seeds, game, split_for
from .near import NearIndex, game_geometries, geometry
from .policy import choose
from .rules import RULES, make_board, white_result
from .state import State, capture, position_signature


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def critical_source_hash():
    root = Path(__file__).parent
    names = ('common.py', 'data.py', 'large_data.py', 'near.py', 'policy.py',
             'rollout.py', 'rules.py', 'simulator.py', 'state.py')
    return digest({name: (root / name).read_text() for name in names})


def iter_rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            yield json.loads(line)


def store_rows(path, rows):
    """Atomic gzip JSONL; timestamps and gzip filenames do not enter its header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    count = 0
    uncompressed = 0
    with temporary.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, filename='', mode='wb', compresslevel=5, mtime=0) as handle:
            for row in rows:
                payload = (canonical(row) + '\n').encode()
                handle.write(payload)
                uncompressed += len(payload)
                count += 1
    temporary.replace(path)
    metadata = {'rows': count, 'bytes': path.stat().st_size, 'uncompressed_jsonl_bytes': uncompressed,
                'sha256': file_sha(path)}
    write_json(str(path) + '.meta.json', metadata)
    return metadata


def check_rows(path):
    path = Path(path)
    meta = read_json(str(path) + '.meta.json')
    if meta['sha256'] != file_sha(path) or meta['bytes'] != path.stat().st_size:
        raise ValueError(f'Shard checksum mismatch: {path}')
    return meta


def summarize_runs(runs, policy):
    n = len(runs)
    counts = [sum(r['root_result'] == c for r in runs) for c in ('W', 'D', 'L')]
    unresolved = n - sum(counts)
    target = [c / n for c in counts] if not unresolved else None
    return {'policy_id': policy, 'replicates': n, 'counts_wdl': counts,
            'unresolved': unresolved, 'target_wdl': target,
            'bounds_wdl': [[c / n, (c + unresolved) / n] for c in counts],
            'mc_standard_error': [((p * (1-p)) / n)**.5 for p in target] if target is not None else None,
            'runs': runs}


def _run_board(root, policy, seed, max_plies):
    """Same trajectory as rollout.continuation, copying the entire root history."""
    board = root.copy(stack=True)
    root_white = board.turn
    rng = random.Random(seed)
    moves = []
    start = time.perf_counter()
    while white_result(board) is None and len(moves) < max_plies:
        move = choose(board, rng, policy)
        moves.append(move.uci())
        board.push(move)
    result = white_result(board)
    relative = result if root_white or result in (None, 'D') else {'W': 'L', 'L': 'W'}[result]
    return {'seed': seed, 'policy_id': policy, 'max_plies': max_plies,
            'moves': moves, 'white_result': result, 'root_result': relative,
            'status': 'terminal' if result is not None else 'unresolved',
            'termination': board.outcome().termination.name if result is not None else 'ply_budget',
            'seconds': time.perf_counter() - start, 'final_fen': board.fen(en_passant='fen')}


def _extend_run(root, policy, seed, horizons):
    previous = None
    extensions = []
    for horizon in horizons:
        current = _run_board(root, policy, seed, horizon)
        if previous is not None:
            if current['moves'][:len(previous['moves'])] != previous['moves']:
                raise ValueError('Administrative extension changed a previous prefix')
            extensions.append({'max_plies': previous['max_plies'], 'plies': len(previous['moves']),
                               'moves_sha256': digest(previous['moves']), 'seconds': previous['seconds']})
        if current['status'] == 'terminal':
            break
        previous = current
    if extensions:
        current['administrative_extensions'] = extensions
    return current


def label_root(args):
    query, config = args
    start = time.perf_counter()
    state = State.from_dict(query['state'])
    board = state.board()
    replay_seconds = time.perf_counter()-start
    before = capture(state.variant, board).to_dict()
    runs = [_extend_run(board, config['policy_id'], seed_for(config['seed'], 'label', state.key, rep),
                        config['horizons']) for rep in range(config['replicates'])]
    if capture(state.variant, board).to_dict() != before:
        raise ValueError('Label worker mutated the shared root history')
    label = summarize_runs(runs, config['policy_id'])
    label['generation_timing'] = {'worker_elapsed_seconds_including_root_replay': time.perf_counter()-start,
                                  'initial_root_replay_seconds': replay_seconds}
    return {**query, 'label': label}


def _game_job(args):
    variant, policy, seed, family, split, kind, horizons = args
    if kind != 'demonstration':
        result = game(variant, policy, seed, family, split, kind, horizons[0])
    else:
        board = make_board(variant, seed=seed)
        run = _extend_run(board, policy, seed, horizons)
        result = {'family_id': family, 'split': split, 'pool': kind, 'variant': variant,
                  'initial': capture(variant, board).to_dict(), **run}
    _audit_game(result)
    result['generation_audit'] = {'passed': True, 'scope': 'every move, stopping, final FEN and result replayed'}
    return result


def checked_label_root(args):
    query = label_root(args)
    start = time.perf_counter()
    _audit_query((query, args[1]))
    query['label']['generation_audit'] = {'passed': True, 'seconds': time.perf_counter()-start,
        'scope': 'every continuation move, stopping, FEN/result, perspective, indexed seed and target; replicate0 policy regeneration'}
    return query


def validate_config(config):
    if config['policy_id'] != 'terminal_aware_v1':
        raise ValueError('The large-study policy is frozen to terminal_aware_v1')
    if not config['variants'] or set(config['variants']) - set(RULES):
        raise ValueError('Unknown or empty variant set')
    if config['replicates'] < 1 or config['workers'] < 1 or config['label_shard_roots'] < 1:
        raise ValueError('Positive replication, worker and shard counts required')
    if config['horizons'] != sorted(set(config['horizons'])) or config['horizons'][0] < 1:
        raise ValueError('Horizons must increase strictly')
    for name in ('source_families', 'demo_families'):
        if set(config[name]) != set(SPLITS) or any(v < 1 for v in config[name].values()):
            raise ValueError('Explicit positive family quotas required for every fold')
    if config['near_distance'] != 2:
        raise ValueError('The conservative pilot radius remains2')


@contextmanager
def worker_pool(workers):
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        yield pool


class UniqueIndex:
    """Omitting repeated identical geometries cannot change a radius-existence test."""
    def __init__(self, radius):
        self.index = NearIndex(radius)
        self.seen = set()
        self.total_seen = 0

    def add(self, item):
        self.total_seen += 1
        if item not in self.seen:
            self.seen.add(item)
            self.index.add(item)

    def has_near(self, item):
        return self.index.has_near(item)


def _freeze(config, output):
    validate_config(config)
    output.mkdir(parents=True, exist_ok=True)
    reference = Path(config['reference_dataset']) if config.get('reference_dataset') else None
    design = {'config': config, 'config_sha256': digest(config),
              'rules_versions': {v: RULES[v]['version'] for v in config['variants']},
              'critical_source_sha256': critical_source_hash(),
              'reference_file_sha256': file_sha(reference) if reference else None,
              'historical_label_files_sha256': {path: file_sha(path) for path in config.get('reference_labels', [])},
              'selection': 'Fixed fresh source-family quotas; one uniformly selected adjacent pair after ply8, before source stopping; geometry-only exclusions before any label generation.',
              'test_status': 'Fresh future-test families; no model predictions or checkpoint selection performed.'}
    path = output / 'design.json'
    if path.exists():
        old = read_json(path)
        if any(old[k] != value for k, value in design.items()):
            raise ValueError('Frozen design/source/reference changed; use a new version')
        return old
    design['provenance'] = provenance(config)
    write_json(path, design)
    return design


def _games(config, output, pool):
    records = []
    for kind, quota in (('demonstration', 'demo_families'), ('query_source', 'source_families')):
        for variant in config['variants']:
            for split in SPLITS:
                families = list(family_seeds(config['seed'], variant, kind, split, config[quota][split]))
                path = output / 'games' / f'{variant}.{split}.{kind}.jsonl.gz'
                if path.exists() and Path(str(path) + '.meta.json').exists():
                    meta = check_rows(path)
                    if [g['family_id'] for g in iter_rows(path)] != [f for f, _ in families]:
                        raise ValueError('Game cache differs from frozen quota/seed')
                else:
                    jobs = [(variant, config['policy_id'], seed, family, split, kind, config['horizons'])
                            for family, seed in families]
                    meta = store_rows(path, pool.map(_game_job, jobs, chunksize=1))
                records.append({'variant': variant, 'split': split, 'pool': kind,
                                'path': str(path.relative_to(output)), **meta})
                write_json(output / 'progress.json', {'stage': 'games', 'completed_buckets': len(records),
                           'total_buckets': len(config['variants']) * 6})
                print('games', variant, split, kind, meta['rows'], flush=True)
    write_json(output / 'games_index.json', records)
    return records


def all_games(output, records, kind=None):
    for record in records:
        if kind is None or record['pool'] == kind:
            yield from iter_rows(output / record['path'])


def _root_cohort(config, output, records, reference):
    path = output / 'roots.jsonl.gz'
    if path.exists() and Path(str(path) + '.meta.json').exists():
        check_rows(path)
        check_rows(output / 'exclusions.jsonl.gz')
        return list(iter_rows(path))
    demos = UniqueIndex(config['near_distance'])
    for g in all_games(output, records, 'demonstration'):
        if g['status'] == 'terminal':
            for item in game_geometries(g):
                demos.add(item)
    historical = UniqueIndex(config['near_distance'])
    if reference:
        for g in reference['demonstrations']:
            for item in game_geometries(g):
                historical.add(item)
        for q in reference['queries']:
            historical.add(geometry(State.from_dict(q['state']).board()))
    folds = {split: UniqueIndex(config['near_distance']) for split in SPLITS}
    signatures = set()
    roots, exclusions = [], []
    for g in all_games(output, records, 'query_source'):
        choices = list(range(8, len(g['moves'])-1))
        base = {k: g[k] for k in ('family_id', 'variant', 'split')}
        if not choices:
            exclusions.append({**base, 'reason': 'no_root_pair'})
            continue
        ply = random.Random(seed_for(g['seed'], 'root_selection')).choice(choices)
        board = State.from_dict(g['initial']).board()
        for uci in g['moves'][:ply]:
            board.push_uci(uci)
        for offset in range(2):
            if offset:
                board.push_uci(g['moves'][ply])
            state = capture(g['variant'], board)
            item, signature = geometry(board), position_signature(state)
            reason = ('duplicate_position' if signature in signatures else
                      'near_new_demo' if demos.has_near(item) else
                      'near_historical_pilot_input' if historical.has_near(item) else
                      'near_cross_fold_root' if any(index.has_near(item) for fold, index in folds.items() if fold != g['split']) else None)
            if reason:
                exclusions.append({**base, 'source_ply': ply+offset, 'state_key': state.key, 'reason': reason})
            else:
                signatures.add(signature)
                folds[g['split']].add(item)
                roots.append({**base, 'id': digest([g['family_id'], state.key]),
                              'source_ply': ply+offset, 'state': state.to_dict()})
    # Admission stays in reference order; label scheduling is train-first and
    # round-robin by variant so an immutable early prefix spans all rules.
    grouped = {(split, variant): [q for q in roots if q['split'] == split and q['variant'] == variant]
               for split in SPLITS for variant in config['variants']}
    roots = [q for split in SPLITS
             for batch in zip_longest(*(grouped[(split, v)] for v in config['variants']))
             for q in batch if q is not None]
    store_rows(path, roots)
    store_rows(output / 'exclusions.jsonl.gz', exclusions)
    write_json(output / 'cohort.json', {'queries': len(roots), 'families': len({q['family_id'] for q in roots}),
        'counts': {f'{v}|{split}': sum(q['variant'] == v and q['split'] == split for q in roots)
                   for v in config['variants'] for split in SPLITS},
        'exclusions': dict(Counter(e['reason'] for e in exclusions)),
        'new_demo_geometry_count': len(demos.seen), 'historical_geometry_count': len(historical.seen),
        'selection_precedes_label_generation': True,
        'label_order': 'train, validation, test; round-robin variants within each fold'})
    print('frozen root cohort', len(roots), 'queries', len(exclusions), 'exclusions', flush=True)
    return roots


def _labels(config, output, roots, pool):
    size = config['label_shard_roots']
    shards = []
    seen_seeds = set()
    if config.get('reference_dataset'):
        reference = read_json(config['reference_dataset'])
        seen_seeds.update(run['seed'] for q in reference['queries'] for run in q['label']['runs'])
    for historical_path in config.get('reference_labels', []):
        historical = read_json(historical_path)
        seen_seeds.update(run['seed'] for row in historical['rows'] for run in row['label']['runs'])
    def checked_rows(queries):
        for query in queries:
            if not query['label'].get('generation_audit', {}).get('passed'):
                raise ValueError('Cannot publish a label without full inline replay audit')
            for run in query['label']['runs']:
                if run['seed'] in seen_seeds:
                    raise ValueError('Continuation stream reused across a published prefix/history')
                seen_seeds.add(run['seed'])
            yield query
    start = 0
    while start < len(roots):
        path = output / 'labels' / f'part-{start//size:05d}.jsonl.gz'
        if not (path.exists() and Path(str(path) + '.meta.json').exists()):
            break
        meta = check_rows(path)
        ids = [q['id'] for q in checked_rows(iter_rows(path))]
        if ids != [q['id'] for q in roots[start:start+size]]:
            raise ValueError('Label cache differs from the frozen cohort order')
        shards.append({'path': str(path.relative_to(output)), **meta})
        start += len(ids)
    written, submitted = start, start
    ready, pending = {}, {}
    window = max(size, 4 * config['workers'])
    while written < len(roots):
        while submitted < len(roots) and submitted-written < window:
            future = pool.submit(checked_label_root, (roots[submitted], config))
            pending[future] = submitted
            submitted += 1
        completed, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in completed:
            ready[pending.pop(future)] = future.result()
        last = min(len(roots), written+size)
        while all(i in ready for i in range(written, last)):
            path = output / 'labels' / f'part-{written//size:05d}.jsonl.gz'
            meta = store_rows(path, checked_rows(ready.pop(i) for i in range(written, last)))
            shards.append({'path': str(path.relative_to(output)), **meta})
            written = last
            write_json(output / 'labels_index.json', shards)
            write_json(output / 'progress.json', {'stage': 'labels', 'completed_queries': written,
                       'total_queries': len(roots), 'replicates_per_query': config['replicates']})
            print('labels', written, '/', len(roots), flush=True)
            if written == size or written % 128 == 0 or written == len(roots):
                design = read_json(output / 'design.json')
                snapshot = {'schema': 'chess_icl_v2.sharded.partial.1', 'complete_dataset': False,
                    'design_sha256': digest(design), 'config': config,
                    'games_index': {'path': 'games_index.json', 'sha256': file_sha(output / 'games_index.json')},
                    'frozen_roots_sha256': file_sha(output / 'roots.jsonl.gz'),
                    'query_shards': list(shards), 'queries': written, 'frozen_cohort_queries': len(roots),
                    'integrity': {'global_cohort_geometry_frozen': True, 'included_games_replayed': True,
                        'included_label_traces_replayed': True, 'included_seeds_checked_against_history': True,
                        'final_full_corpus_audit_pending': True},
                    'test_model_scores_opened': False,
                    'note': 'Immutable audited prefix of the already fixed cohort. Filter query and demo folds/rules before training. It is not the complete corpus.'}
                target = output / 'snapshots' / f'labels-prefix-{written:06d}.json'
                if target.exists():
                    if read_json(target) != snapshot:
                        raise ValueError('Attempt to change an immutable prefix snapshot')
                else:
                    write_json(target, snapshot)
            if written == len(roots):
                break
            last = min(len(roots), written+size)
    write_json(output / 'labels_index.json', shards)
    return shards


def iter_queries(output):
    output = Path(output)
    for record in read_json(output / 'labels_index.json'):
        yield from iter_rows(output / record['path'])


def iter_snapshot_queries(snapshot, split=None):
    """Stream a fixed prefix, never the mutable live index, for a training run."""
    snapshot = Path(snapshot)
    output = snapshot.parent.parent
    manifest = read_json(snapshot)
    for record in manifest['query_shards']:
        if file_sha(output / record['path']) != record['sha256']:
            raise ValueError('Frozen prefix checksum mismatch')
        for query in iter_rows(output / record['path']):
            if split is None or query['split'] == split:
                yield query


def generate(config, output):
    """Build and resume only this version's atomic game/root/label shards."""
    start = time.perf_counter()
    output = Path(output)
    design = _freeze(config, output)
    if (output / 'manifest.json').exists():
        previous = read_json(output / 'manifest.json')
        if previous['design_sha256'] != digest(design):
            raise ValueError('Completed manifest/design mismatch')
        for record in previous['games']+previous['query_shards']:
            check_rows(output / record['path'])
        print('generation already complete; retaining original manifest and timing', flush=True)
        return previous
    reference = read_json(config['reference_dataset']) if config.get('reference_dataset') else None
    with worker_pool(config['workers']) as pool:
        records = _games(config, output, pool)
        roots = _root_cohort(config, output, records, reference)
        shards = _labels(config, output, roots, pool)
    manifest = {'schema': 'chess_icl_v2.sharded.1', 'design_sha256': digest(design),
                'config': config, 'games': records, 'query_shards': shards,
                'queries': len(roots), 'label_continuations': len(roots)*config['replicates'],
                'generation_or_resume_seconds': time.perf_counter()-start,
                'integrity': {'status': 'awaiting_full_audit'}, 'test_model_scores_opened': False}
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'progress.json', {'stage': 'generation_complete_audit_pending',
               'completed_queries': len(roots), 'total_queries': len(roots)})
    return manifest


def _audit_game(g):
    board = State.from_dict(g['initial']).board()
    for uci in g['moves']:
        if white_result(board) is not None:
            raise ValueError('A game continued after a terminal state')
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError('Illegal move in stored game')
        board.push(move)
    result = white_result(board)
    if board.fen(en_passant='fen') != g['final_fen'] or result != g['white_result']:
        raise ValueError('Stored game ending does not replay')
    if (result is None) != (g['status'] == 'unresolved'):
        raise ValueError('Stored game status does not match its result')
    return {'family_id': g['family_id'], 'variant': g['variant'], 'split': g['split'],
            'pool': g['pool'], 'terminal': result is not None, 'plies': len(g['moves'])}


def _audit_query(args):
    query, config = args
    state = State.from_dict(query['state'])
    root = state.board()
    label = query['label']
    if label['replicates'] != config['replicates'] or len(label['runs']) != config['replicates']:
        raise ValueError('Incorrect label replication count')
    for rep, run in enumerate(label['runs']):
        if run['seed'] != seed_for(config['seed'], 'label', state.key, rep):
            raise ValueError('Incorrect independent continuation seed')
        if run['policy_id'] != config['policy_id'] or run['max_plies'] not in config['horizons']:
            raise ValueError('Continuation policy/horizon mismatch')
        if len(run['moves']) > run['max_plies']:
            raise ValueError('Trace exceeds its administrative horizon')
        board = root.copy(stack=True)
        for uci in run['moves']:
            if white_result(board) is not None:
                raise ValueError('Continuation did not stop at a true terminal')
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                raise ValueError('Illegal continuation move')
            board.push(move)
        result = white_result(board)
        relative = result if root.turn or result in (None, 'D') else {'W': 'L', 'L': 'W'}[result]
        if (board.fen(en_passant='fen') != run['final_fen'] or result != run['white_result'] or
                relative != run['root_result'] or (result is None) != (run['status'] == 'unresolved')):
            raise ValueError('Continuation ending/status/perspective mismatch')
        if result is None and len(run['moves']) != run['max_plies']:
            raise ValueError('Unresolved trace stopped before its declared horizon')
        for previous in run.get('administrative_extensions', []):
            if digest(run['moves'][:previous['plies']]) != previous['moves_sha256']:
                raise ValueError('Extension prefix digest mismatch')
    expected = summarize_runs(label['runs'], config['policy_id'])
    if any(label[k] != value for k, value in expected.items()):
        raise ValueError('Stored empirical target/count/bounds/SE mismatch')
    # Independently regenerate replicate0 from its complete root to verify policy sampling.
    first = label['runs'][0]
    check = _run_board(root, config['policy_id'], first['seed'], first['max_plies'])
    if any(check[k] != first[k] for k in ('moves', 'white_result', 'root_result', 'final_fen', 'status', 'termination')):
        raise ValueError('Policy/seed regeneration mismatch')
    return {'id': query['id'], 'family_id': query['family_id'], 'variant': query['variant'], 'split': query['split'],
            'side': 'white' if root.turn else 'black', 'counts_wdl': label['counts_wdl'],
            'replicates': label['replicates'], 'unresolved': label['unresolved'],
            'seeds': [r['seed'] for r in label['runs']],
            'plies': sum(len(r['moves']) for r in label['runs']),
            'worker_elapsed_seconds': label['generation_timing']['worker_elapsed_seconds_including_root_replay'],
            'extension_replays': sum(len(r.get('administrative_extensions', [])) for r in label['runs'])}


def bounded_map(pool, function, jobs, window):
    iterator = iter(jobs)
    pending = set()
    exhausted = False
    while pending or not exhausted:
        while len(pending) < window and not exhausted:
            try:
                job = next(iterator)
            except StopIteration:
                exhausted = True
                break
            pending.add(pool.submit(function, job))
        if not pending:
            break
        completed, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in completed:
            yield future.result()


def audit(output, workers=None):
    """Audit every move/result/count/seed and every cross-pool/root geometry."""
    start = time.perf_counter()
    output = Path(output)
    manifest = read_json(output / 'manifest.json')
    config = manifest['config']
    design = _freeze(config, output)
    if manifest['design_sha256'] != digest(design):
        raise ValueError('Manifest is not bound to the frozen design')
    roots = list(iter_rows(output / 'roots.jsonl.gz'))
    root_by_id = {q['id']: q for q in roots}
    if len(root_by_id) != len(roots):
        raise ValueError('Repeated root ID')
    for record in [*manifest['games'], *manifest['query_shards']]:
        meta = check_rows(output / record['path'])
        if any(meta[k] != record[k] for k in meta):
            raise ValueError('Manifest/shard mismatch')
    families, game_stats = {}, []
    reference = read_json(config['reference_dataset']) if config.get('reference_dataset') else None
    reference_seeds = set()
    historical = UniqueIndex(config['near_distance'])
    if reference:
        for g in reference['demonstrations']:
            for item in game_geometries(g):
                historical.add(item)
        for q in reference['queries']:
            historical.add(geometry(State.from_dict(q['state']).board()))
            reference_seeds.update(r['seed'] for r in q['label']['runs'])
    for path in config.get('reference_labels', []):
        historical_labels = read_json(path)
        reference_seeds.update(run['seed'] for row in historical_labels['rows'] for run in row['label']['runs'])
    demos = UniqueIndex(config['near_distance'])
    source_metadata = {}
    expected_families = {}
    for variant in config['variants']:
        for split in SPLITS:
            for kind, quota in (('demonstration', 'demo_families'), ('query_source', 'source_families')):
                for family, seed in family_seeds(config['seed'], variant, kind, split, config[quota][split]):
                    expected_families[family] = (variant, split, kind, seed)
    for g in all_games(output, manifest['games']):
        family = g['family_id']
        if family in families or expected_families.get(family) != (g['variant'], g['split'], g['pool'], g['seed']):
            raise ValueError('Family/pool/quota/seed mismatch')
        if split_for(family) != g['split']:
            raise ValueError('Incorrect stable family fold')
        families[family] = (g['variant'], g['split'], g['pool'])
        if g['pool'] == 'demonstration' and g['status'] == 'terminal':
            for item in game_geometries(g):
                demos.add(item)
        if g['pool'] == 'query_source':
            source_metadata[family] = (g['initial'], g['moves'], g['seed'])
    if set(families) != set(expected_families):
        raise ValueError('A fixed-quota game family is missing')
    folds = {split: UniqueIndex(config['near_distance']) for split in SPLITS}
    signatures = set()
    for query in roots:
        state = State.from_dict(query['state'])
        initial, moves, source_seed = source_metadata[query['family_id']]
        ply = query['source_ply']
        selected_ply = random.Random(seed_for(source_seed, 'root_selection')).choice(list(range(8, len(moves)-1)))
        if (query['variant'] != state.variant or query['id'] != digest([query['family_id'], state.key]) or
                families[query['family_id']] != (query['variant'], query['split'], 'query_source') or
                state.initial_fen != initial['initial_fen'] or state.extras != initial['extras'] or
                list(state.moves) != moves[:ply] or ply not in (selected_ply, selected_ply+1)):
            raise ValueError('Root provenance/replay prefix mismatch')
        board = state.board()
        item, signature = geometry(board), position_signature(state)
        if (signature in signatures or demos.has_near(item) or historical.has_near(item) or
                any(index.has_near(item) for fold, index in folds.items() if fold != query['split'])):
            raise ValueError('Cross-pool/fold/history geometry audit failed')
        signatures.add(signature)
        folds[query['split']].add(item)
    statistics = []
    query_ids, seeds = set(), set()
    with worker_pool(workers or config['workers']) as pool:
        game_stats = list(bounded_map(pool, _audit_game, all_games(output, manifest['games']), 2*(workers or config['workers'])))
        def jobs():
            for query in iter_queries(output):
                root = {k: v for k, v in query.items() if k != 'label'}
                if root_by_id.get(query['id']) != root:
                    raise ValueError('Labeled root differs from the frozen cohort')
                yield query, config
        for index, stat in enumerate(bounded_map(pool, _audit_query, jobs(), 2*(workers or config['workers']))):
            if stat['id'] in query_ids:
                raise ValueError('Duplicate labeled root')
            if any(seed in seeds or seed in reference_seeds for seed in stat['seeds']):
                raise ValueError('Reused label random stream')
            query_ids.add(stat['id'])
            seeds.update(stat.pop('seeds'))
            statistics.append(stat)
            if (index+1) % 128 == 0:
                write_json(output / 'audit_progress.json', {'completed_queries': index+1, 'total_queries': len(roots)})
                print('audit labels', index+1, '/', len(roots), flush=True)
    if query_ids != set(root_by_id):
        raise ValueError('Incomplete labeled root cohort')
    groups = []
    for variant in config['variants']:
        for split in SPLITS:
            group = [s for s in statistics if s['variant'] == variant and s['split'] == split]
            games = [g for g in game_stats if g['variant'] == variant and g['split'] == split]
            n = sum(s['replicates'] for s in group)
            groups.append({'variant': variant, 'split': split, 'roots': len(group),
                'retained_source_families': len({s['family_id'] for s in group}),
                'sides': dict(Counter(s['side'] for s in group)), 'continuations': n,
                'usable_complete_targets': sum(s['unresolved'] == 0 for s in group),
                'counts_wdl': [sum(s['counts_wdl'][i] for s in group) for i in range(3)] if split != 'test' else None,
                'test_outcomes_withheld_from_compact_summary': split == 'test',
                'unresolved': sum(s['unresolved'] for s in group),
                'mean_continuation_plies': sum(s['plies'] for s in group)/n if n else None,
                'worker_elapsed_seconds': sum(s['worker_elapsed_seconds'] for s in group),
                'extension_replays': sum(s['extension_replays'] for s in group),
                'terminal_demonstrations': sum(g['pool'] == 'demonstration' and g['terminal'] for g in games),
                'unfinished_demo_candidates': sum(g['pool'] == 'demonstration' and not g['terminal'] for g in games),
                'unfinished_sources': sum(g['pool'] == 'query_source' and not g['terminal'] for g in games)})
    summary = {'passed': True, 'queries': len(query_ids), 'retained_source_families': len({s['family_id'] for s in statistics}),
        'label_continuations': len(seeds), 'unresolved': sum(s['unresolved'] for s in statistics),
        'groups': groups, 'fixed_source_families': sum(g['pool'] == 'query_source' for g in game_stats),
        'fixed_demo_candidates': sum(g['pool'] == 'demonstration' for g in game_stats),
        'compressed_shard_bytes': sum(record['bytes'] for record in manifest['games']+manifest['query_shards']),
        'uncompressed_jsonl_bytes': sum(record['uncompressed_jsonl_bytes'] for record in manifest['games']+manifest['query_shards']),
        'audit_seconds': time.perf_counter()-start, 'critical_source_sha256': critical_source_hash(),
        'checks': ['all stored games and label moves legal', 'no moves after a true terminal',
                   'all final results/FENs and root perspectives match replay', 'all seeds match fresh indexed streams',
                   'no pilot label-stream reuse', 'all target counts/bounds/SE match full traces',
                   'replicate0 regenerated for every root under the documented policy',
                   'all source prefixes/families/folds match', 'all query-vs-full-demo and cross-fold near checks',
                   'all new roots distant from pilot queries/full-demo states', 'all shard hashes verified'],
        'test_model_scores_opened': False,
        'cost_note': 'Worker elapsed time includes sharing/scheduling; it is not measured CPU core time. Test WDL class totals intentionally omitted from this compact summary.'}
    write_json(output / 'audit.json', summary)
    manifest['integrity'] = {'passed': True, 'audit_sha256': file_sha(output / 'audit.json')}
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'progress.json', {'stage': 'complete_and_audited', 'completed_queries': len(query_ids),
               'total_queries': len(roots), 'label_continuations': len(seeds), 'unresolved': summary['unresolved']})
    print('large dataset audited', len(query_ids), 'roots', len(seeds), 'continuations', flush=True)
    return summary


def export_legacy_json(output, destination):
    """Stream a compatibility JSON object for the existing CLI without a giant list."""
    output, destination = Path(output), Path(destination)
    manifest = read_json(output / 'manifest.json')
    if not manifest['integrity'].get('passed'):
        raise ValueError('Export requires a complete audited dataset')
    design = read_json(output / 'design.json')
    temporary = destination.with_suffix(destination.suffix+'.tmp')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open('w') as handle:
        header = {'schema': 'chess_icl_v2.1', 'provenance': design['provenance'],
                  'generation_seconds': manifest['generation_or_resume_seconds'],
                  'integrity': manifest['integrity'], 'sharded_parent_manifest_sha256': file_sha(output/'manifest.json')}
        handle.write(canonical(header)[:-1])
        for name, rows in [('demonstrations', (g for g in all_games(output, manifest['games'], 'demonstration') if g['status'] == 'terminal')),
                           ('excluded_demonstrations', (g for g in all_games(output, manifest['games'], 'demonstration') if g['status'] != 'terminal')),
                           ('source_games', all_games(output, manifest['games'], 'query_source')),
                           ('queries', iter_queries(output)), ('exclusions', iter_rows(output/'exclusions.jsonl.gz'))]:
            handle.write(',"'+name+'":[')
            first = True
            for row in rows:
                if not first:
                    handle.write(',')
                handle.write(canonical(row))
                first = False
            handle.write(']')
        handle.write('}\n')
    temporary.replace(destination)
    write_json(str(destination)+'.sha256.json', {'bytes': destination.stat().st_size, 'sha256': file_sha(destination)})
