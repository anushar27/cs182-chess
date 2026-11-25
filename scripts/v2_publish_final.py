"""Publish an immutable compact snapshot and a full local research-artifact bundle."""
from pathlib import Path
import hashlib,shutil,subprocess,zipfile
from chess_icl_v2.common import read_json,write_json,provenance


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    root=Path(__file__).resolve().parents[1];r=root/'results';out=r/'published/final-test-v2'
    if out.exists():raise FileExistsError('Publication snapshots are immutable; choose a new version')
    audit=read_json(r/'final_artifact_validation.json')
    if not audit['passed'] or not audit.get('complete_matrix'):raise ValueError('Complete artifact audit required')
    if not read_json(r/'baseline_fit_validation.json')['passed']:raise ValueError('Baseline refit audit failed')
    if not read_json(r/'high_r_label_validation.json')['passed']:raise ValueError('High-R audit failed')
    for name in ('qwen_test_aggregate.json','qwen_test_coverage_aggregate.json','high_r_analysis.json','final_analysis.json'):
        if not (r/name).exists():raise ValueError('Final analysis incomplete: '+name)
    out.mkdir(parents=True)
    raw=[]
    def record(path):
        raw.append({'path':str(path.relative_to(root)),'bytes':path.stat().st_size,'sha256':sha(path)})
    for path in [r/'terminal_policy_rules_v2.dataset.json',r/'terminal_evaluation_rules_v2.dataset.json',r/'high_r_test_labels_rules_v2.json']:
        record(path)
    compact=['qwen_validation_aggregate.json','qwen_test_aggregate.json','qwen_test_coverage_aggregate.json',
        'final_artifact_validation.json','high_r_analysis.json','high_r_label_validation.json','final_analysis.json',
        'evaluation_matrix_status.json','tests_final.xml','pytest_final.log','baseline_fit_validation.json',
        'installed_wheel_smoke.json','package_build_final.log','package_install_final.log',
        'cluster_matrix_status.json','calibration_ensemble_bins.json']
    compact.extend(p.name for p in r.glob('canonical_*.timing.json'))
    compact.extend(p.name for p in r.glob('canonical_*.progress.json'))
    for name in compact:shutil.copy2(r/name,out/name)
    reports=[r/f'qwen_{phase}_seed{s}.json' for phase in ('validation','test','test_coverage') for s in (42,123,456)]
    reports.extend([r/'direct_exposure_test_seed42.json',r/'direct_exposure_test_coverage_seed42.json',
                    r/'qwen_test_baselines.json',r/'qwen_coverage_baselines.json'])
    for path in reports:
        value=read_json(path);value['published_source_sha256']=sha(path)
        value['prediction_rows_in_full_local_artifact']=len(value.pop('rows'))
        write_json(out/path.name,value);record(path)
    for path in (r/'seals').glob('*.json'):shutil.copy2(path,out/f'seal_{path.name}')
    for folder in [*(f'qwen_warmup_seed{s}' for s in (42,123,456)),*(f'qwen_familiar_seed{s}' for s in (42,123,456)),'qwen_direct_exposure_seed42']:
        record(r/folder/'best.pt')
    for path in (r/'padded_batch_diagnostics').glob('*.json'):record(path)
    for path in r.glob('canonical_*.cache.json'):record(path)
    write_json(out/'raw_artifact_manifest.json',{'artifacts':raw,'notes':'SHA256 hashes are exact file bytes; dataset semantic digests in reports use canonical JSON. Full prediction rows, traces, checkpoint binaries and diagnostic padded outputs are in the local ZIP.'})
    write_json(out/'publication_provenance.json',provenance({'stage':'completed_exploratory_test','source_head_before_publication_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()}))
    (out/'README.md').write_text('''# Completed exploratory fixed-policy W/D/L test snapshot

All three primary model seeds and the one-seed all-rule reference are complete.
The final test uses canonical unpadded batch1 inference of the exact saved inputs,
with frozen weights and identical results across overlapping cohorts. Validation
retains the original padded procedure. The final audit and 51 tests pass.

The publication contains compact metrics, intervals, independent R256 sensitivity,
all exclusions, numerical diagnostics and exact raw-artifact/checkpoint hashes.
Read `docs/EXPLORATORY_STUDY.md` for methods, results and limitations, with CSV and
LaTeX exports under `tables/`. The older validated-core snapshot is unchanged.

Full prediction rows, all R32/R256 continuation traces, legacy/failed-stage evidence
and trainable checkpoint binaries remain in the separately generated local ZIP.
Pinned Qwen base weights are downloaded separately and are not bundled. No claim
of optimal play, confirmatory significance, scaling laws or contamination-free
rule learning is made. Preserve the negative controls and development incidents.
''')
    print('Published final snapshot',len(list(out.iterdir())),'files',flush=True)


def bundle():
    root=Path(__file__).resolve().parents[1];target=root.parent/'chess_icl_research_artifacts.zip'
    sources=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=root).decode().split('\0')
    paths={root/name for name in sources if name and (root/name).is_file()}
    for directory in (root/'results',root/'figures',root/'tables'):
        paths.update(p for p in directory.rglob('*') if p.is_file() and not p.name.endswith('.tmp') and '__pycache__' not in p.parts)
    manifest=[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(paths)]
    write_json(root.parent/'chess_icl_research_artifacts.manifest.json',{'files':manifest})
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(paths):z.write(path,arcname='cs182-chess/'+str(path.relative_to(root)))
    (root.parent/'chess_icl_research_artifacts.sha256').write_text(sha(target)+'  '+target.name+'\n')
    print('Bundled',len(paths),'files;',target.stat().st_size,'bytes;',sha(target),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--bundle-only',action='store_true');args=parser.parse_args()
    if args.bundle_only:bundle()
    else:main()
