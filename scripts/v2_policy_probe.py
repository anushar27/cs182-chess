"""Development-only policy comparison on identical roots; not a mixed-policy dataset."""
import argparse
from collections import Counter
from chess_icl_v2.common import read_json,write_json,provenance
from chess_icl_v2.rollout import estimate
from chess_icl_v2.state import State


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    d=read_json(a.data);rows=[]
    for q in d['queries']:
        if q['split']!='train':continue
        label=estimate(State.from_dict(q['state']),'terminal_aware_v1',2,320,2026112401)
        rows.append({'query_id':q['id'],'family_id':q['family_id'],'variant':q['variant'],'label':label})
        write_json(a.output,{'provenance':provenance({'policy':'terminal_aware_v1','R':2,'budget':320}),
                            'exploratory_only':True,'rows':rows,'complete':False})
        print(q['variant'],label['counts_wdl'],label['unresolved'],flush=True)
    summary=[]
    for v in sorted({r['variant'] for r in rows}):
        runs=[run for r in rows if r['variant']==v for run in r['label']['runs']]
        summary.append({'variant':v,'continuations':len(runs),
           'counts':dict(Counter(r['root_result'] or 'unresolved' for r in runs)),
           'seconds':sum(r['seconds'] for r in runs),'plies':sum(len(r['moves']) for r in runs)})
    write_json(a.output,{'provenance':provenance({'policy':'terminal_aware_v1','R':2,'budget':320}),
                        'exploratory_only':True,'rows':rows,'summary':summary,'complete':True})


if __name__=='__main__':main()
