"""Copy only completed, compact records into an immutable Git publication snapshot."""
from pathlib import Path
from collections import Counter
import hashlib,json,shutil
from chess_icl_v2.common import read_json,write_json,digest


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1];r=root/'results';out=r/'published/validated-core-v2'
    out.mkdir(parents=True,exist_ok=True)
    for name in ('h100_profile.json','cluster_environment.json','simulator_audit_rules_v2.json',
                 'rules_repair_validation.json','high_r_label_validation.json','capture_repair_summary.json',
                 'tests.xml','pytest.log','environment_cluster.txt','final_exploratory_quality_gates.json'):
        shutil.copy2(r/name,out/name)
    for folder in [*(f'qwen_warmup_seed{s}' for s in (42,123,456)),*(f'qwen_familiar_seed{s}' for s in (42,123,456)),'qwen_direct_exposure_seed42']:
        for name in ('manifest.json','best.json','training_log.json'):
            path=r/folder/name
            if not path.exists():raise ValueError(f'Not complete: {path}')
            value=read_json(path)
            if name=='manifest.json':
                value['published_source_sha256']=sha(path)
                if value.get('lineage'):value['lineage']['path']='results/'+Path(value['lineage']['path']).parent.name+'/best.pt'
                value['publication_note']='Completed run; filesystem path normalized. Binary checkpoint retained locally; its exact hash is unchanged.'
            write_json(out/f'{folder}.{name}',value)
    d=read_json(r/'terminal_policy_rules_v2.dataset.json');rows=[]
    for v in d['provenance']['config']['variants']:
        qs=[q for q in d['queries'] if q['variant']==v]
        rows.append({'variant':v,'rules_versions':sorted({q['state']['rules_version'] for q in qs}),
            'roots':len(qs),'roots_by_fold':dict(Counter(q['split'] for q in qs)),
            'continuations':sum(q['label']['replicates'] for q in qs),
            'counts_wdl':[sum(q['label']['counts_wdl'][i] for q in qs) for i in range(3)],
            'unresolved':sum(q['label']['unresolved'] for q in qs)})
    write_json(out/'dataset_summary.json',{'policy':'terminal_aware_v1','R':32,'rows':rows,'integrity':d['integrity'],
        'note':'Fold quotas8 families per rule; hash eligibility70/15/15 is not the final fold ratio.'})
    files=['terminal_policy_rules_v2.dataset.json','terminal_evaluation_rules_v2.dataset.json','high_r_test_labels_rules_v2.json']
    write_json(out/'raw_artifact_manifest.json',{'artifacts':[{'path':'results/'+name,'bytes':(r/name).stat().st_size,'sha256':sha(r/name)} for name in files],
        'storage':'Local deliverable archive; raw traces and binary checkpoints deliberately excluded from Git.'})
    (out/'README.md').write_text('''# Validated core and completed training snapshot

This commit contains the completed corrected simulator/data pipeline, audits,
actual H100 profile, exact dataset hashes and all completed Qwen training records.
The three familiar-rule runs use valid four-rule/standard gradient subsets; the
capture-conversion repair regenerated only that held-out rule. The all-rule
reference was retrained on corrected data. See docs/ARTIFACT_STATUS.md.

The final multi-seed test aggregate is **not part of this snapshot**. Evaluations
were still running when this snapshot was created. No provisional test result is
called final. The failed positive-pilot gate is preserved; these runs are explicitly
exploratory, with all data/simulator/hardware gates passing.

Raw data, individual continuation traces, live logs and binary checkpoints remain
in the local deliverable archive. Their file/checkpoint hashes and reproduction
scripts are provided. Published manifest paths are normalized to repository-relative
paths; the original manifest file hash is recorded alongside each copy.
''')
    print('Published compact completed-stage snapshot',len(list(out.iterdir())),'files')
