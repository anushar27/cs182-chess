"""Versioned targeted rule repair. Retain all other query states and labels exactly."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
from chess_icl_v2.common import digest,read_json,write_json,provenance
from chess_icl_v2.data import family_seeds,position_signature
from chess_icl_v2.near import NearIndex,geometry,game_geometries
from chess_icl_v2.scalable import generate,fast_audit,_game
from chess_icl_v2.state import State
from chess_icl_v2.rollout import estimate


def high_job(q):
    return {'query_id':q['id'],'family_id':q['family_id'],'variant':q['variant'],
      'label':estimate(State.from_dict(q['state']),'terminal_aware_v1',256,2560,2026112410)}


def main():
    training=read_json('results/terminal_policy.dataset.json')
    evaluation=read_json('results/terminal_evaluation.dataset.json')
    cfg={**training['provenance']['config'],'variants':['piece_capture']}
    repaired=generate(cfg,'results/piece_capture_rules_v2.dataset.json')
    def retain(data):
        return {**data,'demonstrations':[g for g in data['demonstrations'] if g['variant']!='piece_capture'],
                'source_games':[g for g in data['source_games'] if g['variant']!='piece_capture'],
                'queries':[q for q in data['queries'] if q['variant']!='piece_capture'],
                'exclusions':list(data['exclusions'])}
    base_train,base_eval=retain(training),retain(evaluation)
    demo_index=NearIndex(2)
    for g in base_eval['demonstrations']:
        for item in game_geometries(g):demo_index.add(item)
    folds={fold:NearIndex(2) for fold in ('train','validation','test')};signatures=set()
    for q in base_eval['queries']:
        state=State.from_dict(q['state']);folds[q['split']].add(geometry(state.board()));signatures.add(position_signature(state))
    kept=[]
    for q in repaired['queries']:
        state=State.from_dict(q['state']);item=geometry(state.board());signature=position_signature(state)
        if signature in signatures or demo_index.has_near(item) or any(idx.has_near(item) for fold,idx in folds.items() if fold!=q['split']):
            base_eval['exclusions'].append({'query_id':q['id'],'family_id':q['family_id'],'reason':'repaired_root_near_existing_global_pool'})
        else:
            kept.append(q);signatures.add(signature);folds[q['split']].add(item)
    for data in (base_train,base_eval):
        data['queries'].extend(kept);data['source_games'].extend(repaired['source_games'])
    roots=NearIndex(2)
    for q in base_eval['queries']:roots.add(geometry(State.from_dict(q['state']).board()))
    extra_jobs=[('piece_capture','terminal_aware_v1',seed,fam,split,'demonstration',2560)
       for split in ('validation','test') for fam,seed in family_seeds(2026112403,'piece_capture','demonstration_extra_v1',split,24)]
    with ProcessPoolExecutor(max_workers=48,mp_context=multiprocessing.get_context('spawn')) as pool:
        extras=list(pool.map(_game,extra_jobs))
    usable=[];excluded=[]
    for g in repaired['demonstrations']+extras:
        if g['status']!='terminal' or any(roots.has_near(item) for item in game_geometries(g)):excluded.append(g)
        else:usable.append(g)
    original_families={g['family_id'] for g in repaired['demonstrations']}
    base_train['demonstrations'].extend(g for g in usable if g['family_id'] in original_families)
    base_eval['demonstrations'].extend(usable)
    record={'variant':'piece_capture','rules_version':'research_rules_v2_royal_conversion',
      'reason':'King capturing enemy rook bypassed conversion because castling target color was omitted',
      'provenance':provenance(cfg),'retained_repaired_queries':len(kept),'dropped_repaired_queries':len(repaired['queries'])-len(kept),
      'usable_repaired_demos':len(usable),'excluded_repaired_demos':len(excluded)}
    for data,parent,path in [(base_train,training,'results/terminal_policy_rules_v2.dataset.json'),
                             (base_eval,evaluation,'results/terminal_evaluation_rules_v2.dataset.json')]:
        data.setdefault('extensions',[]).append({**record,'parent_dataset_sha256':digest(parent)})
        data.setdefault('excluded_demonstrations',[]).extend(excluded)
        # Strictly assert the already-trained four-rule/standard inputs and labels were retained.
        before=[q for q in parent['queries'] if q['variant']!='piece_capture']
        after=[q for q in data['queries'] if q['variant']!='piece_capture']
        if digest(before)!=digest(after):raise ValueError('Unrelated queries changed')
        data['integrity']=fast_audit(data);write_json(path,data)
    old_high=read_json('results/high_r_test_labels.json')
    qs=[q for q in kept if q['split']=='test'];families=sorted({q['family_id'] for q in qs})[:2]
    selected=[q for q in qs if q['family_id'] in families]
    write_json('results/high_r_capture_rules_v2.design.json',{'query_ids':[q['id'] for q in selected],
        'selection':'First two retained family hashes, all siblings','seed':2026112410,'R':256,'rules_version':record['rules_version']})
    with ProcessPoolExecutor(max_workers=min(48,max(1,len(selected))),mp_context=multiprocessing.get_context('spawn')) as pool:
        new_high=list(pool.map(high_job,selected))
    combined=[r for r in old_high['rows'] if r['variant']!='piece_capture']+new_high
    write_json('results/high_r_test_labels_rules_v2.json',{'design':old_high['design'],'repair':record,'rows':combined,
        'unresolved':sum(r['label']['unresolved'] for r in combined)})
    write_json('results/capture_repair_summary.json',record)
    print('Capture repair complete',record['retained_repaired_queries'],flush=True)


if __name__=='__main__':main()
