import argparse
from collections import defaultdict
from chess_icl_v2.common import read_json,write_json
from chess_icl_v2.metrics import aggregate,paired_gains,hierarchical_ci,paired_context_difference


def main():
    p=argparse.ArgumentParser();p.add_argument('reports',nargs='+');p.add_argument('--output',required=True);a=p.parse_args()
    reports=[read_json(f) for f in a.reports]
    if len({r['dataset_sha256'] for r in reports})!=1:raise ValueError('Different evaluation datasets')
    if len({r.get('cohort','length_matched_controls') for r in reports})!=1:raise ValueError('Different evaluation cohorts')
    rows=[r for report in reports for r in report['rows']]
    keys=[(r['query_id'],r['model_seed'],r['condition'],r['k'],r['draw']) for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate prediction rows/seeds')
    # Every seed must use the same roots for a given control and K.
    groups=defaultdict(list)
    for row in rows:groups[(row['exposure'],row['condition'],row['k'])].append(row)
    for key,g in groups.items():
        cohorts=[{r['query_id'] for r in g if r['model_seed']==s} for s in {r['model_seed'] for r in g}]
        if any(c!=cohorts[0] for c in cohorts):raise ValueError(f'Unequal root cohorts across seeds for {key}')
    gain={};relevance={}
    for exposure in {r['exposure'] for r in rows}:
        for k in sorted({r['k'] for r in rows if r['k']}):
            cohort=[r for r in rows if r['exposure']==exposure]
            if any(r['condition']=='mismatched' for r in cohort):
                relevance[f'{exposure}|{k}']=hierarchical_ci(paired_context_difference(cohort,k))
            for cond in ('relevant','mismatched'):
                gain[f'{exposure}|{cond}|{k}']=hierarchical_ci(paired_gains([r for r in rows if r['exposure']==exposure],cond,k))
    write_json(a.output,{'metrics':{'|'.join(map(str,k)):aggregate(g) for k,g in groups.items()},'gains':gain,'relevance_gains':relevance,
                         'label_noise':'Report stored multinomial counts and independent high-R re-estimates separately'})


if __name__=='__main__':main()
