"""Fit on training folds only; score the exact context-evaluation cohort."""
import argparse
from collections import defaultdict
import numpy as np
from chess_icl_v2.baselines import SoftRegression,frequency,histogram_features,demo_prior,knn,material
from chess_icl_v2.common import read_json,write_json
from chess_icl_v2.metrics import aggregate
from chess_icl_v2.state import State


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--evaluation',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();data=read_json(args.data);ev=read_json(args.evaluation);manifest=read_json(args.manifest)
    training=[q for q in data['queries'] if q['split']=='train' and q['variant'] in manifest.get('exposed_variants',manifest['config']['train_variants']) and q['label']['target_wdl'] is not None]
    if not training: raise ValueError('No complete-label training roots for baselines')
    demo_by_id={d['family_id']:d for d in data['demonstrations']}
    query_by_id={q['id']:q for q in data['queries']}
    # Learn the histogram model with relevant K=1 and K=0 examples, sampled without test access.
    pairs=[]
    for q in training:
        demos=[d for d in data['demonstrations'] if d['split']=='train' and d['variant']==q['variant'] and d['status']=='terminal']
        pairs.append((q,[]))
        for d in demos[:4]:pairs.append((q,[d]))
    mat=SoftRegression().fit([(q,[]) for q in training]);hist=SoftRegression(histogram_features).fit(pairs)
    prior=frequency(training);majority=np.ones(3);majority[np.argmax(prior)]+=1;majority=(majority/majority.sum()).tolist()
    rows=[]
    for row in ev['rows']:
        if row['condition'] not in ('zero','relevant','mismatched'): continue
        q=query_by_id[row['query_id']];demos=[demo_by_id[f] for f in row['demo_families']]
        predictions={'training_frequency':prior,'training_majority':majority,'query_material':mat.predict(q,demos),
            'length_histogram':hist.predict(q,demos),'demo_frequency':demo_prior(q,demos),
            'demo_majority':demo_prior(q,demos,True),'knn_demo_states':knn(q,demos)}
        for name,prob in predictions.items():rows.append({**row,'prediction':prob,'baseline':name})
    groups=defaultdict(list)
    for row in rows:groups[(row['baseline'],row['exposure'],row['condition'],row['k'])].append(row)
    report={'rows':rows,'metrics':{'|'.join(map(str,k)):aggregate(v) for k,v in groups.items()},
            'training_roots':len(training),'evaluation_cohort_sha256':ev['dataset_sha256'],
            'fit_scope':'training folds and training-exposed rules only; no variant IDs as input'}
    write_json(args.output,report)


if __name__=='__main__': main()
