"""Construct the pilot evidence record consumed by the main-run gates."""
import argparse
from chess_icl_v2.common import digest,read_json,write_json
from chess_icl_v2.metrics import aggregate


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',required=True);p.add_argument('--baselines',required=True);p.add_argument('--output',required=True);p.add_argument('--k',type=int,default=1);a=p.parse_args()
    ev=read_json(a.evaluation);base=read_json(a.baselines)
    relevant=[r for r in ev['rows'] if r['condition']=='relevant' and r['k']==a.k and r['exposure']=='familiar_rules']
    mismatched=[r for r in ev['rows'] if r['condition']=='mismatched' and r['k']==a.k and r['exposure']=='familiar_rules']
    knn=[r for r in base['rows'] if r['baseline']=='knn_demo_states' and r['condition']=='relevant' and r['k']==a.k and r['exposure']=='familiar_rules']
    if not relevant or not mismatched or not knn:raise ValueError('Need paired familiar-rule model and baseline evidence')
    keys=lambda rows:{(r['query_id'],r['model_seed'],r['draw']) for r in rows}
    if keys(relevant)!=keys(mismatched) or keys(knn)!=keys(relevant):raise ValueError('Pilot cohorts differ')
    r,m,n=[aggregate(x)['macro']['nll'] for x in (relevant,mismatched,knn)]
    zero=[row for row in ev['rows'] if row['condition']=='zero' and row['exposure']=='familiar_rules']
    k0_gain=aggregate(zero)['macro']['nll']-r
    write_json(a.output,{'context_gain_over_knn':n-r,'k0_context_gain':k0_gain,'relevant_gain_over_mismatched':m-r,
                        'familiar_variants':sorted({r['variant'] for r in relevant}),
                        'source_families':len({r['family_id'] for r in relevant}),
                        'evaluation_sha256':digest(ev),'baselines_sha256':digest(base),
                        'dataset_sha256':ev['dataset_sha256'],'k':a.k,'split':ev['split']})


if __name__=='__main__':main()
