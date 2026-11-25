from copy import deepcopy
from pathlib import Path
import random

import pytest

from chess_icl_v2.common import canonical, read_json
from chess_icl_v2.data import generate as reference_generate
from chess_icl_v2.large_data import (UniqueIndex, _run_board, audit, check_rows, export_legacy_json,
    file_sha, generate, iter_queries, iter_rows, iter_snapshot_queries, label_root, store_rows)
from chess_icl_v2.policy import choose
from chess_icl_v2.rollout import estimate, continuation
from chess_icl_v2.rules import RULES, make_board, white_result
from chess_icl_v2.state import capture


@pytest.mark.parametrize('variant', list(RULES))
def test_complete_history_copy_matches_reference_all_rules(variant):
    board = make_board(variant, seed=71)
    rng = random.Random(11)
    for _ in range(18):
        if white_result(board) is not None:break
        board.push(choose(board, rng, 'terminal_aware_v1'))
    state = capture(variant, board)
    query = {'id':'fixture','family_id':'fixture','variant':variant,'split':'train','state':state.to_dict()}
    config = {'policy_id':'terminal_aware_v1','seed':99,'replicates':2,'horizons':[8,24]}
    actual = label_root((query, config))['label']
    expected = estimate(state,'terminal_aware_v1',2,24,99)
    for key in expected:
        if key!='runs':assert actual[key]==expected[key]
    for a,b in zip(actual['runs'],expected['runs']):
        for key in ('seed','moves','white_result','root_result','final_fen','status','termination'):
            assert a[key]==b[key]
    assert capture(variant,board).to_dict()==state.to_dict()


def test_copy_retains_automatic_repetition_and_clock_state():
    board=make_board('standard')
    for move in ['g1f3','g8f6','f3g1','f6g8']*4:board.push_uci(move)
    state=capture('standard',board)
    assert board.is_fivefold_repetition()
    actual=_run_board(state.board(),'terminal_aware_v1',1,20)
    expected=continuation(state,'terminal_aware_v1',1,20)
    for key in expected:
        if key!='seconds':assert actual[key]==expected[key]
    assert actual['moves']==[] and actual['white_result']=='D'


def test_shard_atomic_checksum_and_identical_geometry_dedup(tmp_path):
    path=tmp_path/'a.jsonl.gz';store_rows(path,[{'a':[1,2]},{'b':3}])
    assert list(iter_rows(path))==[{'a':[1,2]},{'b':3}]
    assert check_rows(path)['rows']==2
    index=UniqueIndex(2);item=(True,b'.'*64)
    for _ in range(20):index.add(item)
    assert len(index.seen)==1 and index.total_seen==20 and index.has_near(item)
    changed=(True,b'P'+b'.'*63)
    assert index.has_near(changed)
    with path.open('ab') as handle:handle.write(b'x')
    with pytest.raises(ValueError,match='checksum'):check_rows(path)


def test_sharded_generation_reference_resume_audit_and_export(tmp_path):
    cfg={'seed':137,'variants':['standard','antichess'],'policy_id':'terminal_aware_v1',
         'source_families':dict.fromkeys(('train','validation','test'),1),
         'demo_families':dict.fromkeys(('train','validation','test'),1),
         'replicates':2,'horizons':[2560],'near_distance':2,'workers':2,'label_shard_roots':2,
         'reference_dataset':None,'reference_labels':[]}
    reference=reference_generate({'seed':137,'variants':cfg['variants'],'policy_id':cfg['policy_id'],
        'demo_families_per_split':1,'query_families_per_split':1,'replicates':2,
        'max_plies':2560,'near_distance':2,'workers':1})
    output=tmp_path/'corpus';generate(cfg,output)
    actual={q['id']:q for q in iter_queries(output)}
    assert set(actual)=={q['id'] for q in reference['queries']}
    for q in reference['queries']:
        a=actual[q['id']]
        assert canonical(q['state'])==canonical(a['state'])
        assert q['label']['target_wdl']==a['label']['target_wdl']
        assert [r['moves'] for r in q['label']['runs']]==[r['moves'] for r in a['label']['runs']]
    result=audit(output,2);assert result['passed'] and result['queries']==len(actual)
    hashes={p.name:file_sha(p) for p in (output/'labels').glob('*.gz')}
    manifest_hash=file_sha(output/'manifest.json')
    generate(cfg,output)
    assert manifest_hash==file_sha(output/'manifest.json')
    assert hashes=={p.name:file_sha(p) for p in (output/'labels').glob('*.gz')}
    first=sorted((output/'snapshots').glob('*.json'))[0]
    assert len(list(iter_snapshot_queries(first)))==2
    assert all(q['split']=='train' for q in iter_snapshot_queries(first))
    exported=tmp_path/'export.json';export_legacy_json(output,exported)
    legacy=read_json(exported)
    assert len(legacy['queries'])==len(actual) and legacy['integrity']['passed']
    bad=deepcopy(cfg);bad['seed']+=1
    with pytest.raises(ValueError,match='changed'):generate(bad,output)
