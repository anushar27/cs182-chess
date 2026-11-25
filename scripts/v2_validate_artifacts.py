"""Validate every saved prediction against pool, perspective and cohort contracts."""
from pathlib import Path
import numpy as np
from chess_icl_v2.common import digest,read_json,write_json
from chess_icl_v2.metrics import validate_probabilities


def main():
    root=Path(__file__).resolve().parents[1];r=root/'results'
    d=read_json(r/'terminal_evaluation_rules_v2.dataset.json');datahash=digest(d)
    queries={q['id']:q for q in d['queries']};demos={g['family_id']:g for g in d['demonstrations']}
    errors=[];reports={};cross={}
    for q in d['queries']:
        white=(q['state']['initial_fen'].split()[1]=='w') == (len(q['state']['moves'])%2==0)
        for run in q['label']['runs']:
            result=run['white_result']
            if not white and result in ('W','L'):result={'W':'L','L':'W'}[result]
            if result!=run['root_result']:errors.append('Label perspective mismatch')
    patterns=['qwen_validation_seed*.json','qwen_test_seed*.json','qwen_test_coverage_seed*.json',
              'direct_exposure_test_seed*.json','direct_exposure_test_coverage_seed*.json']
    for path in sorted({p for pattern in patterns for p in r.glob(pattern)}):
        ev=read_json(path)
        if ev['dataset_sha256']!=datahash or not ev['frozen_weights']:errors.append(path.name+': wrong dataset/frozen flag')
        zeros={(row['query_id'],row['model_seed']):row for row in ev['rows'] if row['k']==0}
        relevant={(row['query_id'],row['model_seed'],row['draw']):row for row in ev['rows'] if row['condition']=='relevant'}
        seen=set()
        for row in ev['rows']:
            key=(row['query_id'],row['model_seed'],row['condition'],row['k'],row['draw'])
            if key in seen:errors.append(path.name+': duplicate row')
            seen.add(key);validate_probabilities(row['prediction']);q=queries[row['query_id']]
            if row['target']!=q['label']['target_wdl'] or row['label_replicates']!=32:errors.append(path.name+': target mismatch')
            if len(row['demo_families'])!=row['k']:errors.append(path.name+': K mismatch')
            for family in row['demo_families']:
                g=demos[family]
                if g['split']!=q['split'] or g['status']!='terminal' or g['policy_id']!=q['label']['policy_id'] or family==q['family_id']:errors.append(path.name+': demo pool mismatch')
                if (g['variant']==q['variant'])!=(row['condition']!='mismatched'):errors.append(path.name+': rule relevance mismatch')
            if row['condition']=='mismatched':
                rel=relevant[(row['query_id'],row['model_seed'],row['draw'])];zero=zeros[(row['query_id'],row['model_seed'])]
                rl=rel['tokens']-zero['tokens'];ml=row['tokens']-zero['tokens']
                if abs(rl-ml)>.1*max(1,rl):errors.append(path.name+': control length mismatch')
            condition='direct' if path.name.startswith('direct') else 'familiar'
            crosskey=(ev['split'],ev['cohort'],row['query_id'],row['condition'],row['k'],row['draw'])
            if crosskey in cross and cross[crosskey]!=row['input_sha256']:errors.append('Different inputs across model seeds')
            cross[crosskey]=row['input_sha256']
        reports[path.name]={'predictions':len(ev['rows']),'unique_roots':len({row['query_id'] for row in ev['rows']}),
                            'excluded_roots':len(ev['exclusions']),'cohort':ev['cohort']}
    # Canonical outputs must be identical across cohorts for the exact same input.
    numerical={}
    for seed in (42,123,456):
        primary=r/f'qwen_test_seed{seed}.json';coverage=r/f'qwen_test_coverage_seed{seed}.json'
        if primary.exists() and coverage.exists():
            a={tuple(row[k] for k in ('query_id','condition','k','draw')):row for row in read_json(primary)['rows'] if row['condition'] in ('zero','relevant')}
            diffs=[]
            for row in read_json(coverage)['rows']:
                key=tuple(row[k] for k in ('query_id','condition','k','draw'))
                if key in a:
                    if a[key]['input_sha256']!=row['input_sha256']:errors.append('Coverage input mismatch')
                    diffs.extend(np.abs(np.array(a[key]['prediction'])-row['prediction']).tolist())
            numerical[str(seed)]={'max_probability_difference':max(diffs,default=0.),'mean_probability_difference':float(np.mean(diffs)) if diffs else None}
            if max(diffs,default=0.)!=0:errors.append('Canonical cross-cohort prediction mismatch')
    expected={f'qwen_{phase}_seed{seed}.json' for phase in ('validation','test','test_coverage') for seed in (42,123,456)} | {'direct_exposure_test_seed42.json','direct_exposure_test_coverage_seed42.json'}
    complete=expected.issubset(reports)
    for name in expected:
        if name in reports and 'validation' not in name:
            if not read_json(r/name).get('inference_policy','').startswith('canonical_unpadded'):
                errors.append(name+': canonical inference missing')
    report={'complete_matrix':complete,'missing_reports':sorted(expected-set(reports)),'passed':not errors and complete,'errors':errors,'dataset_sha256':datahash,'reports':reports,'cross_cohort_bfloat16_batch_sensitivity':numerical}
    write_json(r/'final_artifact_validation.json',report)
    if errors:raise ValueError(sorted(set(errors)))
    print('Validated prediction artifacts',len(reports),flush=True)


if __name__=='__main__':main()
