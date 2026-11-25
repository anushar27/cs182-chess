"""Persistent CPU worker pool; same pool/fold/seed semantics as the reference generator."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import random
import time
from pathlib import Path
from .common import digest,provenance,seed_for,write_json
from .data import SPLITS,family_seeds,game
from .near import NearIndex,geometry,game_geometries
from .rollout import estimate
from .state import State,capture,position_signature


def _game(args):return game(*args)
def _label(args):
    state,policy,r,budget,seed=args
    return estimate(state,policy,r,budget,seed,workers=1)


def generate(config,output):
    start=time.perf_counter()
    data={'schema':'chess_icl_v2.1','provenance':provenance(config),'demonstrations':[],
          'source_games':[],'queries':[],'exclusions':[]}
    ctx=multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=config['workers'],mp_context=ctx) as pool:
        for kind,key in [('demonstration','demo_families_per_split'),('query_source','query_families_per_split')]:
            jobs=[(v,config['policy_id'],seed,fam,split,kind,config['max_plies'])
                  for v in config['variants'] for split in SPLITS
                  for fam,seed in family_seeds(config['seed'],v,kind,split,config[key])]
            dest=data['demonstrations'] if kind=='demonstration' else data['source_games']
            for index,g in enumerate(pool.map(_game,jobs,chunksize=1)):
                dest.append(g)
                if (index+1)%16==0:print(kind,index+1,'/',len(jobs),flush=True)
            write_json(str(output)+'.games_checkpoint.json',data)
        demo_index=NearIndex(config['near_distance'])
        for g in data['demonstrations']:
            for item in game_geometries(g):demo_index.add(item)
        folds={split:NearIndex(config['near_distance']) for split in SPLITS};signatures=set();label_jobs=[]
        for g in data['source_games']:
            count=len(g['moves']);choices=list(range(8,count-1))
            if not choices:
                data['exclusions'].append({'family_id':g['family_id'],'reason':'no_root_pair'});continue
            ply=random.Random(seed_for(g['seed'],'root_selection')).choice(choices)
            b=State.from_dict(g['initial']).board()
            for uci in g['moves'][:ply]:b.push_uci(uci)
            for j in range(2):
                if j:b.push_uci(g['moves'][ply])
                state=capture(g['variant'],b);item=geometry(b);signature=position_signature(state)
                if signature in signatures or demo_index.has_near(item) or any(index.has_near(item) for fold,index in folds.items() if fold!=g['split']):
                    data['exclusions'].append({'family_id':g['family_id'],'state_key':state.key,'reason':'near_duplicate'});continue
                signatures.add(signature);folds[g['split']].add(item)
                data['queries'].append({'id':digest([g['family_id'],state.key]),'family_id':g['family_id'],
                    'split':g['split'],'variant':g['variant'],'state':state.to_dict()})
                label_jobs.append((state,config['policy_id'],config['replicates'],config['max_plies'],config['seed']))
        for i,label in enumerate(pool.map(_label,label_jobs,chunksize=1)):
            data['queries'][i]['label']=label
            if (i+1)%16==0:
                print('labels',i+1,'/',len(label_jobs),flush=True)
                # Recover completed labels without rewriting every continuation on each root.
                write_json(str(output)+'.label_progress.json',{'complete':i+1,'total':len(label_jobs),'seconds':time.perf_counter()-start})
    data['generation_seconds']=time.perf_counter()-start
    data['integrity']=fast_audit(data)
    write_json(output,data)
    return data


def fast_audit(data):
    # Validate all game replay and root/pool geometry with compact indices.
    errors=[];families={};demos=NearIndex(data['provenance']['config']['near_distance'])
    for g in data['demonstrations']+data['source_games']:
        if g['family_id'] in families:errors.append('duplicate family')
        from .data import split_for
        if split_for(g['family_id'])!=g['split']:errors.append('family fold mismatch')
        families[g['family_id']]=(g['split'],g['pool'])
        b=State.from_dict(g['initial']).board()
        if g['pool']=='demonstration':demos.add(geometry(b))
        for uci in g['moves']:
            b.push_uci(uci)
            if g['pool']=='demonstration':demos.add(geometry(b))
        if b.fen(en_passant='fen')!=g['final_fen']:errors.append('game replay mismatch')
    folds={s:NearIndex(data['provenance']['config']['near_distance']) for s in SPLITS};keys=set()
    for q in data['queries']:
        state=State.from_dict(q['state']);b=state.board();item=geometry(b);key=position_signature(state)
        if q['variant']!=state.variant or families.get(q['family_id'])!=(q['split'],'query_source'):errors.append('query provenance mismatch')
        if key in keys or demos.has_near(item) or any(idx.has_near(item) for fold,idx in folds.items() if fold!=q['split']):errors.append('duplicate/near query')
        keys.add(key);folds[q['split']].add(item)
        label=q['label'];counts=[sum(r['root_result']==c for r in label['runs']) for c in ('W','D','L')]
        if counts!=label['counts_wdl'] or len(label['runs'])!=label['replicates'] or len({r['seed'] for r in label['runs']})!=label['replicates']:errors.append('label counts/seeds')
        u=label['replicates']-sum(counts)
        if label['unresolved']!=u or label['target_wdl']!=([n/label['replicates'] for n in counts] if not u else None):errors.append('censor/target mismatch')
    if errors:raise ValueError(sorted(set(errors)))
    return {'passed':True,'queries':len(data['queries']),'families':len(families),
            'near_distance':data['provenance']['config']['near_distance'],
            'scope':'all demo states vs roots; cross-fold query roots','implementation':'exact pigeonhole Hamming index'}
