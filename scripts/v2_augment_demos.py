"""Add independent evaluation demos; hold all query roots/labels/checkpoints fixed."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import argparse
import time
from chess_icl_v2.common import digest,provenance,read_json,write_json
from chess_icl_v2.data import family_seeds
from chess_icl_v2.near import NearIndex,geometry,game_geometries
from chess_icl_v2.scalable import _game,fast_audit
from chess_icl_v2.state import State

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',default='results/terminal_policy_rules_v2.dataset.json')
    parser.add_argument('--output',default='results/terminal_evaluation_rules_v2.dataset.json')
    parser.add_argument('--design-output',default='results/extra_demo_design.json');args=parser.parse_args()
    data=read_json(args.data);parent=digest(data);start=time.perf_counter()
    design={'seed':2026112403,'extra_per_evaluation_fold':24,'policy':'terminal_aware_v1','max_plies':2560,
            'scope':['validation','test'],'selection':'fixed number of independently seeded full games; reject entire near-conflicting or unfinished games'}
    write_json(args.design_output,{'parent_dataset_sha256':parent,'provenance':provenance(design)})
    roots=NearIndex(2)
    for q in data['queries']:roots.add(geometry(State.from_dict(q['state']).board()))
    jobs=[(v,'terminal_aware_v1',seed,fam,split,'demonstration',2560)
          for v in data['provenance']['config']['variants'] for split in ('validation','test')
          for fam,seed in family_seeds(2026112403,v,'demonstration_extra_v1',split,24)]
    added=0
    with ProcessPoolExecutor(max_workers=48,mp_context=multiprocessing.get_context('spawn')) as pool:
        for i,g in enumerate(pool.map(_game,jobs,chunksize=1)):
            conflict=g['status']!='terminal' or any(roots.has_near(item) for item in game_geometries(g))
            if conflict:
                data.setdefault('excluded_demonstrations',[]).append(g)
                data['exclusions'].append({'family_id':g['family_id'],'reason':'extra_demo_unfinished_or_near_query'})
            else:data['demonstrations'].append(g);added+=1
            if (i+1)%32==0:print('extra demos',i+1,'/',len(jobs),'accepted',added,flush=True)
    data.setdefault('extensions',[]).append({'kind':'independent_evaluation_demos','parent_dataset_sha256':parent,
        'provenance':provenance(design),'accepted':added,'attempted':len(jobs),'seconds':time.perf_counter()-start})
    data['integrity']=fast_audit(data)
    write_json(args.output,data)
    print('saved evaluation data',added,'extra demos',flush=True)
