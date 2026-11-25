from argparse import Namespace

from chess_icl_v2.common import read_json
from chess_icl_v2.large_data import generate, iter_snapshot_queries
from scripts.v2_large_dataset_panel import build, verify


def test_fresh_panel_uses_identical_parent_streams_without_new_samples(tmp_path):
    cfg={'seed':137,'variants':['standard','antichess'],'policy_id':'terminal_aware_v1',
         'source_families':dict.fromkeys(('train','validation','test'),1),
         'demo_families':dict.fromkeys(('train','validation','test'),1),
         'replicates':2,'horizons':[2560],'near_distance':2,'workers':2,'label_shard_roots':2,
         'reference_dataset':None,'reference_labels':[]}
    corpus=tmp_path/'corpus';generate(cfg,corpus)
    args=Namespace(corpus=str(corpus),output=str(tmp_path/'panel'),families=1,workers=2)
    build(args)
    manifest=read_json(tmp_path/'panel/snapshots/validation-panel.json')
    rows=list(iter_snapshot_queries(tmp_path/'panel/snapshots/validation-panel.json'))
    assert manifest['queries']==len(rows) and rows
    assert manifest['independent_samples_in_addition_to_full_corpus']==0
    assert all(row['split']=='validation' for row in rows)
    assert {row['variant'] for row in rows}==set(cfg['variants'])
    verify(args)
    proof=read_json(tmp_path/'panel/full_corpus_equivalence.json')
    assert proof['passed'] and proof['independent_extra_samples']==0
    assert proof['identical_duplicate_continuations']==len(rows)*2
