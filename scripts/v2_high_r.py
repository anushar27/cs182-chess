"""Independent high-sample labels on a preselected sealed-test subset."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from chess_icl_v2.common import digest,read_json,write_json,provenance
from chess_icl_v2.rollout import estimate
from chess_icl_v2.state import State


def job(q):
    return {'query_id':q['id'],'family_id':q['family_id'],'variant':q['variant'],
            'label':estimate(State.from_dict(q['state']),'terminal_aware_v1',256,2560,2026112410)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    data=read_json(a.data);selected=[]
    for variant in data['provenance']['config']['variants']:
        qs=[q for q in data['queries'] if q['variant']==variant and q['split']=='test']
        families=sorted({q['family_id'] for q in qs})[:2]
        selected.extend(q for q in qs if q['family_id'] in families)
    design={'provenance':provenance({'seed':2026112410,'R':256,'budget':2560,'policy':'terminal_aware_v1'}),
        'dataset_sha256':digest(data),'query_ids':[q['id'] for q in selected],
        'selection':'First two family hashes per rule, all retained siblings; no outcome selection'}
    write_json(a.output+'.design.json',design)
    with ProcessPoolExecutor(max_workers=48,mp_context=multiprocessing.get_context('spawn')) as pool:
        rows=list(pool.map(job,selected))
    write_json(a.output,{'design':design,'rows':rows,'unresolved':sum(r['label']['unresolved'] for r in rows)})
    print('High-R roots completed',len(rows),flush=True)


if __name__=='__main__':main()
