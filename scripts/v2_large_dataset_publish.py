"""Publish completed large-data evidence without putting raw traces into Git."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import shutil
import zipfile

from chess_icl_v2.common import digest, read_json, write_json
from chess_icl_v2.large_data import critical_source_hash, file_sha, iter_queries, iter_snapshot_queries

ROOT = Path(__file__).resolve().parents[1]


def publish_panel(args):
    corpus=Path(args.corpus);panel=corpus.parent/'validation_panel'
    parent=read_json(corpus/'design.json');design=read_json(panel/'design.json')
    manifest=read_json(panel/'snapshots/validation-panel.json')
    if manifest['design_sha256']!=digest(parent) or design['parent_roots_sha256']!=file_sha(corpus/'roots.jsonl.gz'):
        raise ValueError('Panel is not bound to this fixed cohort')
    if parent['critical_source_sha256']!=critical_source_hash():
        raise ValueError('Pinned generation sources changed')
    rows=list(iter_snapshot_queries(panel/'snapshots/validation-panel.json'))
    if {q['id'] for q in rows}!=set(design['query_ids']) or len(rows)!=manifest['queries']:
        raise ValueError('Panel ID/count mismatch')
    old=read_json(parent['config']['reference_dataset'])
    old_seeds={r['seed'] for q in old['queries'] for r in q['label']['runs']}
    for path in parent['config'].get('reference_labels',[]):
        high=read_json(path);old_seeds.update(r['seed'] for row in high['rows'] for r in row['label']['runs'])
    seeds=set()
    counts=[]
    for q in rows:
        if q['split']!='validation' or not q['label']['generation_audit']['passed']:
            raise ValueError('Unvalidated/non-validation panel record')
        for r in q['label']['runs']:
            if r['seed'] in seeds or r['seed'] in old_seeds:raise ValueError('Panel/history stream reuse')
            seeds.add(r['seed'])
    for variant in parent['config']['variants']:
        group=[q for q in rows if q['variant']==variant]
        counts.append({'variant':variant,'roots':len(group),'families':len({q['family_id'] for q in group}),
            'replicates':parent['config']['replicates'],'unresolved':sum(q['label']['unresolved'] for q in group)})
    qa={'passed':True,'roots':len(rows),'unique_panel_continuations':len(seeds),
        'unresolved':sum(q['label']['unresolved'] for q in rows),'groups':counts,
        'critical_source_sha256':critical_source_hash(),'bound_to_full_frozen_cohort':True,
        'all_included_trace_replay_audits_passed':True,'disjoint_from_historical_R32_R256_streams':True,
        'full_corpus_duplicate_equivalence':'pending full corpus generation',
        'independent_samples_in_addition_to_full_corpus':0,'test_model_scores_opened':False,
        'generation_seconds_excluding_wait_for_parent_cohort':manifest['generation_seconds']}
    write_json(panel/'readiness_audit.json',qa)
    target=ROOT/'results/published/large-validation-panel-v1'
    if target.exists():raise FileExistsError('Immutable panel snapshot exists')
    target.mkdir(parents=True)
    for source,name in [(panel/'design.json','design.json'),(panel/'snapshots/validation-panel.json','manifest.json'),
        (panel/'readiness_audit.json','readiness_audit.json'),(corpus/'cohort.json','parent_cohort.json'),
        (corpus.parent/'panel_tests.log','panel_tests.log'),(corpus.parent/'panel_tests.xml','panel_tests.xml')]:
        shutil.copy2(source,target/name)
    (target/'README.md').write_text(f'''# Fixed fresh validation panel, ready

This immutable panel contains{len(rows)} validation roots,16 retained source families
and32 roots per rule/control, across13 variants plus standard. Every included
R32 trace passed replay/status/FEN/result/seed/count checks; history disjointness
checks passed and{qa['unresolved']} samples are unresolved. No model predictions are published.

The panel uses the exact indexed streams which later appear in the full larger
corpus. It adds zero independent samples to that corpus; final exact equivalence
verification remains pending until full generation reaches these validation roots.
The parent root cohort and all geometry decisions were already frozen. The main
48-worker generator was not interrupted. This16-worker side job only supplied
validation early, after approval review rejected a proposed checkpoint-stop.

Bind the raw manifest at results/large_dataset_v1/validation_panel/snapshots/validation-panel.json.
The raw shards remain local. Filter exposed rules for each experiment's checkpoint
selection. The original test/pilot records and completed benchmark snapshots stay
unchanged. This is not a completed large corpus or proof of adequate500M training
sample size. Read docs/LARGE_DATASET.md and the parent frozen cohort counts here.
''')
    print('Published audited ready validation panel',len(rows),'roots',flush=True)


def publish(args):
    corpus = Path(args.corpus)
    manifest = read_json(corpus/'manifest.json')
    audit = read_json(corpus/'audit.json')
    if not manifest['integrity'].get('passed') or not audit['passed']:
        raise ValueError('Only a completed, fully audited corpus can be published')
    if manifest['integrity']['audit_sha256'] != file_sha(corpus/'audit.json'):
        raise ValueError('Manifest/audit hash mismatch')
    panel = corpus.parent/'validation_panel'
    proof = read_json(panel/'full_corpus_equivalence.json')
    if not proof['passed'] or proof['independent_extra_samples'] != 0:
        raise ValueError('Same-seed panel/full-corpus equivalence must pass')
    output = ROOT/'results/published/large-data-v1'
    if output.exists():
        raise FileExistsError('Immutable publication already exists')
    output.mkdir(parents=True)
    for source, name in [(corpus/'manifest.json','manifest.json'),(corpus/'design.json','design.json'),
        (corpus/'audit.json','audit.json'),(corpus/'cohort.json','cohort.json'),
        (panel/'design.json','validation_panel.design.json'),
        (panel/'snapshots/validation-panel.json','validation_panel.manifest.json'),
        (panel/'full_corpus_equivalence.json','validation_panel.full_corpus_equivalence.json')]:
        shutil.copy2(source, output/name)
    for name in ('generator_tests.log','generator_tests.xml','panel_tests.log','panel_tests.xml'):
        shutil.copy2(corpus.parent/name,output/name)
    costs=defaultdict(lambda:{'label_generation_worker_seconds':0.,'inline_replay_audit_worker_seconds':0.})
    for q in iter_queries(corpus):
        key=(q['variant'],q['split'])
        costs[key]['label_generation_worker_seconds'] += q['label']['generation_timing']['worker_elapsed_seconds_including_root_replay']
        costs[key]['inline_replay_audit_worker_seconds'] += q['label']['generation_audit']['seconds']
    rows=[]
    for group in audit['groups']:
        rows.append({**group,**costs[(group['variant'],group['split'])]})
    summary={'queries':audit['queries'],'retained_source_families':audit['retained_source_families'],
        'label_continuations':audit['label_continuations'],'unresolved':audit['unresolved'],
        'generation_seconds':manifest['generation_or_resume_seconds'],'final_audit_seconds':audit['audit_seconds'],
        'groups':rows,'compressed_shard_bytes':audit['compressed_shard_bytes'],
        'uncompressed_jsonl_bytes':audit['uncompressed_jsonl_bytes'],
        'duplicate_panel_roots':proof['identical_queries'],
        'duplicate_panel_continuation_executions':proof['identical_duplicate_continuations'],
        'extra_independent_samples_from_panel':0,'test_model_scores_opened':False,
        'note':'Requested source families384/rule versus pilot24,16x. Root/label counts are admitted actuals. New test WDL totals remain omitted. Worker elapsed time is not CPU core time; panel work duplicates indexed streams.'}
    write_json(output/'dataset_summary.json',summary)
    columns=['variant','split','roots','retained_source_families','continuations','usable_complete_targets',
        'unresolved','terminal_demonstrations','unfinished_demo_candidates','unfinished_sources',
        'mean_continuation_plies','label_generation_worker_seconds','inline_replay_audit_worker_seconds','extension_replays']
    with (output/'dataset_summary.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns,lineterminator='\n');writer.writeheader()
        writer.writerows({key:row[key] for key in columns} for row in rows)
    raw=[]
    for path in sorted(corpus.rglob('*')):
        if path.is_file() and not path.name.endswith('.tmp') and not path.name.startswith('.'):
            raw.append({'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':file_sha(path)})
    for path in sorted(panel.rglob('*')):
        if path.is_file() and not path.name.endswith('.tmp') and not path.name.startswith('.'):
            raw.append({'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':file_sha(path)})
    write_json(output/'raw_artifact_manifest.json',{'artifacts':raw,'base':'repository root',
        'storage':'Full raw traces in separately versioned local large-dataset ZIP, not Git. Historical pilot inputs needed for regeneration are identified by frozen input hashes.'})
    (output/'README.md').write_text(f'''# Completed fresh larger corpus, v1

The final corpus contains{audit['queries']:,} admitted roots from{audit['retained_source_families']:,}
independent retained query-source families, with{audit['label_continuations']:,} distinctR32
continuations and{audit['unresolved']} unresolved outcomes. This is a new seed/family corpus,
not an extension of the previously opened pilot test. See docs/LARGE_DATASET.md.

Every stored game and label trace passed the full replay/status/FEN/result/seed
audit and global geometry checks. The separate early validation panel exactly
duplicates its full-corpus indexed streams and adds zero independent samples.
New test model scores remain unopened; compact test WDL totals are withheld.

This publishes data evidence, not model performance or proof that the data scale
is adequate for500M scratch training. Train/validation/test and rule filters must
be declared per experiment. Full raw data remain in the local dataset ZIP with
the file/checksum manifest here; the previous pilot snapshots stay unchanged.
''')
    print('Published completed large corpus',output,flush=True)


def bundle(args):
    corpus=Path(args.corpus);panel=corpus.parent/'validation_panel'
    published=ROOT/'results/published/large-data-v1'
    if not published.exists():raise ValueError('Publish final audited evidence first')
    archive=ROOT.parent/'chess_icl_large_dataset_v1.zip'
    sources=[ROOT/'docs/LARGE_DATASET.md',ROOT/'configs/v2_large_dataset_v1.json',
             ROOT/'configs/v2_large_dataset_benchmark.json',ROOT/'scripts/v2_large_dataset.py',
             ROOT/'scripts/v2_large_dataset_panel.py',ROOT/'scripts/v2_large_dataset_publish.py',
             ROOT/'tests/test_v2_large_data.py',ROOT/'tests/test_v2_large_dataset_panel.py']
    critical=('__init__.py','common.py','data.py','large_data.py','near.py','policy.py','rollout.py','rules.py','simulator.py','state.py')
    sources.extend(ROOT/'chess_icl_v2'/name for name in critical)
    for directory in (corpus,panel,published,corpus.parent/'benchmark'):
        sources.extend(p for p in directory.rglob('*') if p.is_file() and not p.name.endswith('.tmp') and not p.name.startswith('.'))
    for name in ('generation.log','validation_panel.log','generator_tests.log','generator_tests.xml','panel_tests.log','panel_tests.xml'):
        sources.append(corpus.parent/name)
    sources=sorted(set(sources))
    manifest=[{'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':file_sha(path)} for path in sources]
    write_json(ROOT.parent/'chess_icl_large_dataset_v1.manifest.json',{'files':manifest})
    with zipfile.ZipFile(archive,'w') as handle:
        for path in sources:
            handle.write(path,arcname='cs182-chess/'+str(path.relative_to(ROOT)),
                         compress_type=zipfile.ZIP_STORED if path.suffix=='.gz' else zipfile.ZIP_DEFLATED,compresslevel=5)
    checksum=file_sha(archive)
    (ROOT.parent/'chess_icl_large_dataset_v1.sha256').write_text(checksum+'  '+archive.name+'\n')
    print('Large corpus bundle',len(sources),'files',archive.stat().st_size,'bytes',checksum,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['panel','publish','bundle'])
    parser.add_argument('--corpus',default='results/large_dataset_v1/corpus');args=parser.parse_args()
    if args.command=='panel':publish_panel(args)
    elif args.command=='publish':publish(args)
    else:bundle(args)


if __name__=='__main__':main()
