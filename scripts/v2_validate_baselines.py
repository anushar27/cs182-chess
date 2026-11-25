"""Refit declared baselines and verify saved predictions and training-only exposure."""
from collections import Counter
from pathlib import Path
import numpy as np
from chess_icl_v2.baselines import SoftRegression,frequency,histogram_features,demo_prior,knn
from chess_icl_v2.common import read_json,write_json,digest,provenance


def main():
    r=Path(__file__).resolve().parents[1]/'results';data=read_json(r/'terminal_evaluation_rules_v2.dataset.json')
    manifest=read_json(r/'qwen_familiar_seed42/manifest.json');exposed=set(manifest['exposed_variants'])
    training=[q for q in data['queries'] if q['split']=='train' and q['variant'] in exposed and q['label']['target_wdl'] is not None]
    pairs=[]
    for q in training:
        demos=[d for d in data['demonstrations'] if d['split']=='train' and d['variant']==q['variant'] and d['status']=='terminal']
        pairs.append((q,[]));pairs.extend((q,[d]) for d in demos[:4])
    material=SoftRegression().fit([(q,[]) for q in training]);hist=SoftRegression(histogram_features).fit(pairs)
    prior=frequency(training);majority=np.ones(3);majority[np.argmax(prior)]+=1;majority=(majority/majority.sum()).tolist()
    by_q={q['id']:q for q in data['queries']};by_d={d['family_id']:d for d in data['demonstrations']}
    checked=Counter();knn_seen=set()
    for filename in ('qwen_test_baselines.json','qwen_coverage_baselines.json'):
        baseline=read_json(r/filename)
        if baseline['evaluation_cohort_sha256']!=digest(data) or baseline['training_roots']!=len(training):raise ValueError('Baseline data mismatch')
        for row in baseline['rows']:
            q=by_q[row['query_id']];demos=[by_d[f] for f in row['demo_families']];name=row['baseline']
            if row['target']!=q['label']['target_wdl']:raise ValueError('Baseline target mismatch')
            if name=='training_frequency':expected=prior
            elif name=='training_majority':expected=majority
            elif name=='query_material':expected=material.predict(q,demos)
            elif name=='length_histogram':expected=hist.predict(q,demos)
            elif name=='demo_frequency':expected=demo_prior(q,demos)
            elif name=='demo_majority':expected=demo_prior(q,demos,True)
            elif name=='knn_demo_states':
                key=(filename,q['variant'],row['condition'],row['side'])
                if key in knn_seen:continue
                knn_seen.add(key);expected=knn(q,demos)
            else:raise ValueError('Unknown baseline')
            if not np.allclose(expected,row['prediction'],rtol=0,atol=1e-12):raise ValueError('Refit differs: '+name)
            checked[name]+=1
    write_json(r/'baseline_fit_validation.json',{'passed':True,'dataset_sha256':digest(data),
        'provenance':provenance({'steps':500,'lr':.1,'l2':.001,'histogram_training':'K0 plus first four valid K1 demos per training root'}),
        'training_query_ids':[q['id'] for q in training], 'training_rules':sorted(exposed),
        'training_roots_by_rule':dict(Counter(q['variant'] for q in training)),
        'material_weights':material.weights.tolist(),'histogram_weights':hist.weights.tolist(),
        'training_frequency':prior,'training_majority':majority,'verified_predictions':dict(checked),
        'knn_check_scope':'One saved episode per file/rule/condition/side; all other baseline predictions checked by exact deterministic refit within1e-12.'})
    print('Verified baseline refits',dict(checked),flush=True)


if __name__=='__main__':main()
