"""Produce a fixed fresh validation panel while the full corpus keeps generating.

These are duplicate re-estimates of the SAME indexed R32 streams which later
appear in the full corpus. They add zero independent samples to corpus totals.
"""
import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import time

from chess_icl_v2.common import digest, read_json, write_json
from chess_icl_v2.large_data import (bounded_map, checked_label_root, check_rows,
    file_sha, iter_queries, iter_rows, iter_snapshot_queries, store_rows, worker_pool)


def semantic_label(label):
    value = {k: v for k, v in label.items() if k not in ('generation_timing', 'generation_audit', 'runs')}
    value['runs'] = []
    for original in label['runs']:
        run = {k: v for k, v in original.items() if k not in ('seconds', 'administrative_extensions')}
        if original.get('administrative_extensions'):
            run['administrative_extensions'] = [
                {k: v for k, v in e.items() if k != 'seconds'} for e in original['administrative_extensions']]
        value['runs'].append(run)
    return value


def build(args):
    corpus, output = Path(args.corpus), Path(args.output)
    while not (corpus / 'cohort.json').exists():
        time.sleep(5)
    started = time.perf_counter()
    check_rows(corpus / 'roots.jsonl.gz')
    roots = list(iter_rows(corpus / 'roots.jsonl.gz'))
    design = read_json(corpus / 'design.json')
    config = design['config']
    selected, families_by_rule = [], {}
    for variant in config['variants']:
        eligible = [q for q in roots if q['split'] == 'validation' and q['variant'] == variant]
        families = sorted({q['family_id'] for q in eligible})[:args.families]
        families_by_rule[variant] = families
        selected.extend(q for q in eligible if q['family_id'] in families)
    selected.sort(key=lambda q: (q['variant'], q['family_id'], q['id']))
    if not selected or any(not families for families in families_by_rule.values()):
        raise ValueError('Validation panel is missing a requested rule')
    panel_design = {'parent_design_sha256': digest(design), 'parent_roots_sha256': file_sha(corpus/'roots.jsonl.gz'),
        'parent_games_index_sha256': file_sha(corpus/'games_index.json'),
        'selection': f'First{args.families} retained validation-family hashes per rule, all admitted siblings; label/outcome independent.',
        'families_per_rule_requested': args.families, 'families_by_rule': families_by_rule,
        'query_ids': [q['id'] for q in selected], 'config': config,
        'workers': args.workers, 'panel_script_sha256': file_sha(__file__),
        'policy_stream_relation': f'Exact duplicate root/replicate seeds and R{config["replicates"]} samples of the full corpus; zero additional independent samples.',
        'operational_reason': 'Early validation availability without stopping or changing the ongoing full generation; automatic approval review rejected stopping that authorized process.',
        'test_model_scores_opened': False}
    output.mkdir(parents=True, exist_ok=True)
    if (output/'design.json').exists():
        if read_json(output/'design.json') != panel_design:
            raise ValueError('Validation panel design changed')
    else:write_json(output/'design.json', panel_design)
    if (output/'snapshots/validation-panel.json').exists():
        print('Fixed validation panel already complete', flush=True)
        return
    size = 32
    records = []
    completed = set()
    for path in sorted((output/'labels').glob('part-*.jsonl.gz')):
        if not Path(str(path)+'.meta.json').exists():continue
        meta = check_rows(path)
        for q in iter_rows(path):
            if q['id'] in completed or q['id'] not in panel_design['query_ids']:
                raise ValueError('Invalid cached panel query')
            completed.add(q['id'])
        records.append({'path': str(path.relative_to(output)), **meta})
    todo = [q for q in selected if q['id'] not in completed]
    batch = []
    seeds = set()
    with worker_pool(args.workers) as pool:
        for query in bounded_map(pool, checked_label_root, ((q,config) for q in todo), 2*args.workers):
            for run in query['label']['runs']:
                if run['seed'] in seeds:raise ValueError('Panel continuation seed reused')
                seeds.add(run['seed'])
            batch.append(query)
            if len(batch) == size:
                path = output/'labels'/f'part-{len(records):05d}.jsonl.gz'
                meta = store_rows(path, sorted(batch,key=lambda q:q['id']))
                records.append({'path':str(path.relative_to(output)),**meta})
                completed.update(q['id'] for q in batch);batch=[]
                write_json(output/'progress.json',{'stage':'labels','completed':len(completed),'total':len(selected)})
                print('validation panel',len(completed),'/',len(selected),flush=True)
    if batch:
        path=output/'labels'/f'part-{len(records):05d}.jsonl.gz'
        meta=store_rows(path,sorted(batch,key=lambda q:q['id']))
        records.append({'path':str(path.relative_to(output)),**meta})
        completed.update(q['id'] for q in batch)
    if completed != set(panel_design['query_ids']):raise ValueError('Incomplete panel')
    root_relative=Path(os.path.relpath(corpus,output))
    manifest={'schema':'chess_icl_v2.sharded.partial.1','complete_dataset':False,
        'design_sha256':digest(design),'panel_design_sha256':digest(panel_design),'config':config,
        'games_index':{'path':str(root_relative/'games_index.json'),'sha256':file_sha(corpus/'games_index.json')},
        'frozen_roots_sha256':file_sha(corpus/'roots.jsonl.gz'),'query_shards':records,'queries':len(selected),
        'frozen_cohort_queries':len(roots),'integrity':{'global_cohort_geometry_frozen':True,'included_games_replayed':True,
            'included_label_traces_replayed':True,'replicate0_policy_regenerated_per_root':True,'final_full_corpus_audit_pending':True},
        'test_model_scores_opened':False,'independent_samples_in_addition_to_full_corpus':0,
        'relation_to_full_corpus':'Same query IDs and indexed R32 streams. Do not pool duplicate labels. Exact comparison required after full generation.',
        'generation_seconds':time.perf_counter()-started}
    write_json(output/'snapshots/validation-panel.json',manifest)
    write_json(output/'progress.json',{'stage':'complete_audited_panel','completed':len(selected),'total':len(selected),
                                     'counts_by_rule':dict(Counter(q['variant'] for q in selected))})
    print('Fresh validation panel ready',len(selected),'roots',flush=True)


def verify(args):
    panel = Path(args.output)/'snapshots/validation-panel.json'
    subset = {q['id']:q for q in iter_snapshot_queries(panel)}
    found=set()
    for q in iter_queries(args.corpus):
        if q['id'] not in subset:continue
        other=subset[q['id']]
        if q['state'] != other['state'] or semantic_label(q['label']) != semantic_label(other['label']):
            raise ValueError('Duplicate full/panel stream differs')
        found.add(q['id'])
    if found!=set(subset):raise ValueError('Full corpus does not yet contain the complete panel')
    result={'passed':True,'identical_queries':len(found),'identical_duplicate_continuations':sum(q['label']['replicates'] for q in subset.values()),
        'comparison':'All labels, indexed seeds, moves, final FENs/results/status/terminations and extension prefixes identical; timing/audit runtime metadata excluded.',
        'independent_extra_samples':0}
    write_json(Path(args.output)/'full_corpus_equivalence.json',result)
    print(result,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['build','verify'])
    parser.add_argument('--corpus',default='results/large_dataset_v1/corpus')
    parser.add_argument('--output',default='results/large_dataset_v1/validation_panel')
    parser.add_argument('--workers',type=int,default=16);parser.add_argument('--families',type=int,default=16)
    args=parser.parse_args();Path(args.output).mkdir(parents=True,exist_ok=True)
    with (Path(args.output)/'.panel.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.command=='build':build(args)
        else:verify(args)


if __name__=='__main__':main()
