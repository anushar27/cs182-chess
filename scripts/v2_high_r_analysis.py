"""Score frozen saved predictions against independent, hash-selected R256 labels."""
from collections import defaultdict
from pathlib import Path
import math
import numpy as np
from chess_icl_v2.common import read_json,write_json,digest
from chess_icl_v2.metrics import aggregate,paired_gains,paired_context_difference,hierarchical_ci


def wilson(count,n):
    z=1.959963984540054;p=count/n;denom=1+z*z/n
    center=(p+z*z/(2*n))/denom
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [max(0.,center-radius),min(1.,center+radius)]


def summarize(rows,bootstrap=True):
    groups=defaultdict(list)
    for row in rows:groups[(row['exposure'],row['condition'],row['k'])].append(row)
    gains={};relevance={}
    for exposure in sorted({r['exposure'] for r in rows}) if bootstrap else []:
        cohort=[r for r in rows if r['exposure']==exposure]
        for cond in ('relevant','mismatched'):
            gains[f'{exposure}|{cond}|1']=hierarchical_ci(paired_gains(cohort,cond,1))
        if any(r['condition']=='mismatched' for r in cohort):
            relevance[f'{exposure}|1']=hierarchical_ci(paired_context_difference(cohort,1))
    counts={exposure:{'roots':len({r['query_id'] for r in rows if r['exposure']==exposure}),
                       'families':len({r['family_id'] for r in rows if r['exposure']==exposure})}
            for exposure in {r['exposure'] for r in rows}}
    return {'metrics':{'|'.join(map(str,key)):aggregate(value) for key,value in groups.items()},'cohort_counts':counts,
            'gains':gains,'relevance_gains':relevance,'prediction_rows':len(rows),
            'roots':len({r['query_id'] for r in rows}),'families':len({r['family_id'] for r in rows}),
            'variants':sorted({r['variant'] for r in rows})}


def main():
    root=Path(__file__).resolve().parents[1];r=root/'results'
    data=read_json(r/'terminal_evaluation_rules_v2.dataset.json')
    low={q['id']:q for q in data['queries']};high=read_json(r/'high_r_test_labels_rules_v2.json')
    if not read_json(r/'high_r_label_validation.json')['passed']:raise ValueError('High-R audit failed')
    labels={row['query_id']:row for row in high['rows']};noise=[]
    for qid,row in labels.items():
        a=low[qid]['label'];b=row['label']
        if b['replicates']!=256 or b['unresolved'] or a['replicates']!=32:raise ValueError('Unexpected label precision')
        delta=np.array(a['target_wdl'])-b['target_wdl']
        noise.append({'query_id':qid,'family_id':row['family_id'],'variant':row['variant'],
            'R32':a['target_wdl'],'R256':b['target_wdl'],'R256_counts':b['counts_wdl'],
            'R256_marginal_wilson95':[wilson(c,256) for c in b['counts_wdl']],
            'R32_mc_se':a['mc_standard_error'],'R256_mc_se':b['mc_standard_error'],
            'l1_difference':float(np.abs(delta).sum()),'max_class_difference':float(np.abs(delta).max()),
            'squared_l2_difference':float(np.square(delta).sum())})
    results={}
    for name,paths in {
        'familiar_matched':[r/f'qwen_test_seed{s}.json' for s in (42,123,456)],
        'familiar_coverage':[r/f'qwen_test_coverage_seed{s}.json' for s in (42,123,456)],
        'direct_matched':[r/'direct_exposure_test_seed42.json'],
        'direct_coverage':[r/'direct_exposure_test_coverage_seed42.json']}.items():
        rows=[]
        for path in paths:
            report=read_json(path)
            if not report.get('inference_policy','').startswith('canonical_unpadded'):raise ValueError('Canonical evaluation required')
            for row in report['rows']:
                if row['query_id'] in labels:
                    label=labels[row['query_id']]['label']
                    rows.append({**row,'target':label['target_wdl'],'label_replicates':256,
                                 'label_mc_se':label['mc_standard_error']})
        results[name]=summarize(rows)
    for filename in ('qwen_test_baselines.json','qwen_coverage_baselines.json'):
        baseline=read_json(r/filename);by_name=defaultdict(list)
        for row in baseline['rows']:
            if row['query_id'] in labels:
                by_name[row['baseline']].append({**row,'target':labels[row['query_id']]['label']['target_wdl']})
        results[filename.removesuffix('.json')]={name:summarize(rows,False) for name,rows in by_name.items()}
    write_json(r/'high_r_analysis.json',{'evaluation_dataset_sha256':digest(data),'high_r_sha256':digest(high),
        'selection':high['design']['selection'],'roots':len(labels),'families':len({v['family_id'] for v in labels.values()}),
        'independent_continuations':sum(v['label']['replicates'] for v in labels.values()),
        'mean_R32_R256_l1':float(np.mean([v['l1_difference'] for v in noise])),
        'mean_R32_R256_squared_l2':float(np.mean([v['squared_l2_difference'] for v in noise])),
        'max_R32_R256_class_difference':max(v['max_class_difference'] for v in noise),
        'noise_rows':noise,'results':results,
        'uncertainty_note':'Bootstrap intervals condition on observed labels. Wilson intervals are marginal class intervals, not simultaneous multinomial intervals. Independent R256 scoring is a sensitivity analysis; only two source families per rule.'})
    print('Independent high-R analysis complete',len(labels),'roots',flush=True)


if __name__=='__main__':main()
