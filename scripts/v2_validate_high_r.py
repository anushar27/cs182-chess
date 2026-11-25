"""Verify high-R selection, independent streams, counts and corrected rule states."""
from chess_icl_v2.common import read_json,write_json,digest,seed_for
from chess_icl_v2.state import State

if __name__=='__main__':
    h=read_json('results/high_r_test_labels_rules_v2.json');d=read_json('results/terminal_policy_rules_v2.dataset.json')
    qmap={q['id']:q for q in d['queries']};expected=set()
    for variant in d['provenance']['config']['variants']:
        qs=[q for q in d['queries'] if q['split']=='test' and q['variant']==variant]
        families=sorted({q['family_id'] for q in qs})[:2]
        expected.update(q['id'] for q in qs if q['family_id'] in families)
    if {r['query_id'] for r in h['rows']}!=expected:raise ValueError('High-R selection differs from frozen family rule')
    seen=set();main_seeds={run['seed'] for q in d['queries'] for run in q['label']['runs']}
    for row in h['rows']:
        q=qmap[row['query_id']];state=State.from_dict(q['state']);label=row['label']
        if row['variant']!=q['variant'] or row['family_id']!=q['family_id'] or label['replicates']!=256:raise ValueError('Root metadata mismatch')
        for i,run in enumerate(label['runs']):
            if run['seed']!=seed_for(2026112410,'label',state.key,i) or run['seed'] in seen or run['seed'] in main_seeds:
                raise ValueError('Seed stream mismatch/reuse')
            seen.add(run['seed'])
        counts=[sum(run['root_result']==c for run in label['runs']) for c in ('W','D','L')]
        if counts!=label['counts_wdl'] or sum(counts)!=256 or label['target_wdl']!=[n/256 for n in counts]:raise ValueError('Incomplete or inconsistent high-R label')
    parent=h.get('parent_design',h['design'])
    h['parent_design']=parent
    h['design']={**parent,'dataset_sha256':digest(d),'query_ids':[r['query_id'] for r in h['rows']],
        'version':'rules_v2','parent_design_sha256':digest(parent),'repair_design':'high_r_capture_rules_v2.design.json'}
    h['validation']={'passed':True,'seed_streams_match_corrected_states':True,'unique_continuations':len(seen),
        'independent_of_R32_streams':True,'family_selection_verified':True,'unresolved':0}
    write_json('results/high_r_test_labels_rules_v2.json',h);write_json('results/high_r_label_validation.json',h['validation'])
    print(h['validation'])
