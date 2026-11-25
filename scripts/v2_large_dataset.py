"""Generate, audit, export and benchmark versioned larger fixed-policy corpora."""
import argparse
import fcntl
from pathlib import Path
import time

from chess_icl_v2.common import digest, read_json, write_json
from chess_icl_v2.large_data import (audit, export_legacy_json, file_sha, generate,
                                     label_root, worker_pool)


def benchmark(args):
    data = read_json(args.data)
    config = {'policy_id': 'terminal_aware_v1', 'seed': data['provenance']['config']['seed'],
              'replicates': 32, 'horizons': [2560, 5120, 10240]}
    queries = [{k: v for k, v in q.items() if k != 'label'} for q in data['queries']]
    start = time.perf_counter()
    with worker_pool(args.workers) as pool:
        new = list(pool.map(label_root, ((q, config) for q in queries), chunksize=1))
    seconds = time.perf_counter()-start
    for original, optimized in zip(data['queries'], new):
        a, b = original['label'], optimized['label']
        for key in a:
            if key != 'runs' and a[key] != b[key]:
                raise ValueError('Optimized label differs: '+key)
        for old_run, new_run in zip(a['runs'], b['runs']):
            if any(old_run[key] != new_run[key] for key in old_run if key != 'seconds'):
                raise ValueError('Optimized continuation trajectory differs')
    result = {'passed': True, 'reference_file_sha256': file_sha(args.data),
        'reference_queries': len(new), 'reference_continuations': sum(q['label']['replicates'] for q in new),
        'workers': args.workers, 'optimized_label_seconds_including_startup': seconds,
        'optimized_worker_elapsed_seconds_including_root_replay': sum(q['label']['generation_timing']['worker_elapsed_seconds_including_root_replay'] for q in new),
        'reference_original_generation_seconds': data['generation_seconds'],
        'comparison': 'Exact counts, probabilities, every seed, every continuation move, final FEN, status and termination match; timing fields excluded. Original generation includes additional games/dedup/audit, so its wall time is not a direct speedup denominator.',
        'optimized_query_digest_excluding_timing': digest([{k: v for k, v in q.items() if k != 'label'} for q in new])}
    write_json(args.output, result)
    print(result, flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('generate');p.add_argument('--config', required=True);p.add_argument('--output', required=True)
    p.add_argument('--skip-audit', action='store_true')
    p = sub.add_parser('audit');p.add_argument('--output', required=True);p.add_argument('--workers', type=int)
    p = sub.add_parser('export');p.add_argument('--output', required=True);p.add_argument('--destination', required=True)
    p = sub.add_parser('benchmark');p.add_argument('--data', required=True);p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    if args.command == 'generate':
        Path(args.output).mkdir(parents=True, exist_ok=True)
        with (Path(args.output)/'.generation.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest = generate(read_json(args.config), args.output)
            if not args.skip_audit and not manifest['integrity'].get('passed'):
                audit(args.output)
    elif args.command == 'audit':
        with (Path(args.output)/'.generation.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            audit(args.output, args.workers)
    elif args.command == 'export':export_legacy_json(args.output, args.destination)
    else:benchmark(args)


if __name__ == '__main__':main()
