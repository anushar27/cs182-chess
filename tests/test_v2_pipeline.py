from copy import deepcopy
import json
import numpy as np
import pytest
import torch
from chess_icl_v2.common import digest
from chess_icl_v2.data import split_for, audit_dataset
from chess_icl_v2.episodes import ByteTokenizer, build_episode, paired_episodes
from chess_icl_v2.metrics import score, aggregate, paired_gains, hierarchical_ci
from chess_icl_v2.models import TinyTransformer, Predictor, collate, soft_ce
from chess_icl_v2.rules import make_board
from chess_icl_v2.state import capture
from chess_icl_v2.tokenizer import SquareMoveTokenizer
from chess_icl_v2.train import weights,restore


@pytest.fixture
def sample():
    b=make_board('standard')
    state=capture('standard',b)
    moves=['f2f3','e7e5','g2g4','d8h4']
    q={'id':'query','variant':'standard','family_id':'queryfamily','split':'validation',
       'state':state.to_dict(),'label':{'target_wdl':[.2,.3,.5],'policy_id':'uniform_legal_v1'}}
    demos=[]
    for i,v in enumerate(['standard','standard','antichess','antichess']):
        demos.append({'family_id':f'demo{i}','split':'validation','variant':v,
          'initial':state.to_dict(),'moves':moves,'white_result':'L','status':'terminal','policy_id':'uniform_legal_v1'})
    return q,demos


def test_query_is_never_truncated_and_k_is_exact(sample):
    q,d=sample;tok=ByteTokenizer()
    z=build_episode(q,d,tok,0,10000,'zero')
    ep=build_episode(q,d,tok,2,10000)
    assert ep.text.endswith('</QUERY>\n<PREDICT_WDL>') and ep.text.count('<DEMO>')==2
    with pytest.raises(ValueError,match='never truncate'): build_episode(q,d,tok,0,10)
    with pytest.raises(ValueError,match='Full K=2'): build_episode(q,d,tok,2,len(z.input_ids)+10)
    with pytest.raises(ValueError,match='independent'): build_episode(q,d,tok,3,10000)


def test_no_identifiers_or_query_labels_in_input(sample):
    q,d=sample;tok=ByteTokenizer()
    ep=build_episode(q,d,tok,1,10000)
    for forbidden in ['variant','queryfamily','demo0','demo1','uniform_legal','target_wdl','0.2']:
        assert forbidden not in ep.text
    altered=deepcopy(q);altered['label']['target_wdl']=[1.,0.,0.]
    assert build_episode(altered,d,tok,1,10000).input_ids==ep.input_ids
    wrong=deepcopy(d)
    for g in wrong: g['split']='test'
    with pytest.raises(ValueError): build_episode(q,wrong,tok,1,10000)


def test_controls_pair_roots_and_lengths(sample):
    q,d=sample
    z,r,m=paired_episodes(q,d,ByteTokenizer(),1,10000)
    assert r.metadata['demo_families']!=m.metadata['demo_families']
    assert r.metadata['query_id']==m.metadata['query_id']==z.metadata['query_id']
    assert r.metadata['draw']==m.metadata['draw']
    corrupt=build_episode(q,d,ByteTokenizer(),1,10000,'corrupt_outcomes')
    assert corrupt.input_ids!=r.input_ids
    assert corrupt.metadata['demo_families']==r.metadata['demo_families']
    assert corrupt.text.split('<QUERY>')[1]==r.text.split('<QUERY>')[1]


def test_square_move_vocabulary(sample):
    q,d=sample
    ep=build_episode(q,d,SquareMoveTokenizer(),1,10000)
    assert ep.input_ids[-1]==3 and max(ep.input_ids)<337 and min(ep.input_ids)>0
    assert len(ep.input_ids)<len(build_episode(q,d,ByteTokenizer(),1,10000).input_ids)


def test_proper_scores_and_cluster_bootstrap():
    perfect=score([.2,.3,.5],[.2,.3,.5])
    assert perfect['brier']==0
    assert score([1,0,0],[.9,.05,.05])['nll'] < score([1,0,0],[.2,.4,.4])['nll']
    with pytest.raises(ValueError): score([1,0,0],[.5,.5,.5])
    rows=[]
    for seed in (1,2):
        for i in range(4):
            common={'query_id':str(i),'model_seed':seed,'family_id':str(i//2),'variant':'standard','target':[1.,0.,0.]}
            rows.append({**common,'prediction':[.4,.3,.3],'k':0,'condition':'zero','draw':0})
            for draw in range(2): rows.append({**common,'prediction':[.6,.2,.2],'k':1,'condition':'relevant','draw':draw})
    json.dumps(aggregate(rows))  # Research reports must serialize without NumPy scalars.
    gains=paired_gains(rows)
    report=hierarchical_ci(gains,replicates=50)
    assert report['macro_gain']==pytest.approx(np.log(1.5))
    assert report['ci95'][0]==pytest.approx(report['ci95'][1])
    rows[0]['query_id']='not-paired'
    with pytest.raises(ValueError,match='identical roots'): paired_gains(rows)


def test_training_update_padding_reload_and_frozen_eval(sample,tmp_path):
    torch.set_num_threads(2);torch.manual_seed(2)
    q,d=sample;tok=SquareMoveTokenizer()
    a=build_episode(q,d,tok,0,10000,'zero');b=build_episode(q,d,tok,1,10000)
    model=TinyTransformer(vocab_size=337,width=16,layers=1,heads=2)
    model.eval()
    single=model(**collate([a]))
    padded=model(**collate([a,b]))[:1]
    assert torch.allclose(single,padded,atol=1e-6)
    before={k:v.clone() for k,v in weights(model).items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    loss=soft_ce(model(**collate([a,b])),torch.tensor([a.target,b.target]))
    loss.backward();optimizer.step()
    assert any(not torch.equal(before[k],v) for k,v in weights(model).items())
    path=tmp_path/'state.pt';torch.save(weights(model),path)
    restored=TinyTransformer(vocab_size=337,width=16,layers=1,heads=2);restore(restored,path)
    one=Predictor(model);two=Predictor(restored)
    assert one.predict_proba(a)==two.predict_proba(a)==two.predict_proba(a)
    assert not any(p.requires_grad for p in two.model.parameters())
    assert sum(two.predict_proba(a))==pytest.approx(1.)


def test_lora_injection_last_four_layers_without_download():
    from transformers import Qwen2Config,Qwen2Model
    from peft import LoraConfig,get_peft_model
    cfg=Qwen2Config(vocab_size=337,hidden_size=32,intermediate_size=64,num_hidden_layers=5,
                    num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=512)
    base=Qwen2Model(cfg)
    from chess_icl_v2.models import lora_config
    model=get_peft_model(base,lora_config(5))
    trainable=[n for n,p in model.named_parameters() if p.requires_grad]
    assert trainable and all('lora_' in n and '.layers.0.' not in n for n in trainable)
    output=model(input_ids=torch.tensor([[2,3,4]]),attention_mask=torch.ones(1,3,dtype=torch.long)).last_hidden_state
    assert output.shape==(1,3,32)
    output.float().sum().backward()


def test_qwen_wrapper_local_forward_checkpoint_and_frozen_base(tmp_path):
    from transformers import Qwen2Config,Qwen2Model
    from chess_icl_v2.models import QwenProbabilityModel
    torch.set_num_threads(2)
    path=tmp_path/'local_qwen'
    Qwen2Model(Qwen2Config(vocab_size=337,hidden_size=32,intermediate_size=64,num_hidden_layers=5,
       num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=512)).save_pretrained(path)
    cfg={'model_id':str(path),'revision':'a'*40,'gradient_checkpointing':False}
    model=QwenProbabilityModel(**cfg)
    frozen={n:p.clone() for n,p in model.named_parameters() if not p.requires_grad}
    batch={'input_ids':torch.tensor([[1,2,3]]),'attention_mask':torch.ones(1,3,dtype=torch.long)}
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    soft_ce(model(**batch),torch.tensor([[.3,.3,.4]])).backward();opt.step()
    assert all(torch.equal(v,dict(model.named_parameters())[n]) for n,v in frozen.items())
    checkpoint=tmp_path/'qwen.pt';torch.save(weights(model),checkpoint)
    loaded=QwenProbabilityModel(**cfg);restore(loaded,checkpoint)
    model.eval();loaded.eval()
    assert torch.equal(model(**batch),loaded(**batch))


def test_extension_prunes_whole_conflicting_demo_and_preserves_roots():
    from chess_icl_v2.data import split_for
    from chess_icl_v2.diagnostics import extend
    from chess_icl_v2.rollout import continuation,estimate
    root=capture('standard',make_board('standard'))
    short=continuation(root,'uniform_legal_v1',19,1)
    longer=continuation(root,'uniform_legal_v1',19,4)
    b=root.board()
    for move in longer['moves'][:2]:b.push_uci(move)
    query_state=capture('standard',b)
    families=[f'family{i}' for i in range(100) if split_for(f'family{i}')=='train']
    demo={'family_id':families[0],'split':'train','pool':'demonstration','variant':'standard','initial':root.to_dict(),**short}
    source={'family_id':families[1],'split':'train','pool':'query_source','variant':'standard','initial':query_state.to_dict(),
            **continuation(query_state,'uniform_legal_v1',99,0)}
    query={'id':'fixed','family_id':families[1],'split':'train','variant':'standard','state':query_state.to_dict(),
           'label':estimate(query_state,'uniform_legal_v1',1,0,57)}
    data={'provenance':{'config':{'seed':57,'near_distance':0}},'demonstrations':[demo],'source_games':[source],
          'queries':[query],'exclusions':[]}
    result=extend(data,4)
    assert result['queries'][0]['state']==query['state']
    assert result['queries'][0]['label']['runs'][0]['seed']==query['label']['runs'][0]['seed']
    assert not result['demonstrations'] and len(result['excluded_demonstrations'])==1
    assert result['exclusions'][0]['reason']=='extended_demo_near_query'
    assert len(data['demonstrations'])==1 and data['queries'][0]['label']['unresolved']==1
    assert result['integrity']['passed']


def test_baseline_cache_preserves_outcomes_and_root_perspective(sample):
    from chess_icl_v2.baselines import knn,SoftRegression
    q,d=sample
    assert knn(q,d[:1],1)==[.25,.25,.5]
    assert knn(q,d[:1],1)==[.25,.25,.5]
    changed=deepcopy(d[:1]);changed[0]['white_result']='W'
    assert knn(q,changed,1)==[.5,.25,.25]
    black=deepcopy(q);black['state']=capture('standard',make_board('standard')).mirror().to_dict()
    assert knn(black,d[:1],1)==[.5,.25,.25]
    with pytest.raises(ValueError,match='training data only'):
        SoftRegression().fit([(q,[])])


def test_common_warmup_lineage_exposure_and_hash_validation(tmp_path):
    from chess_icl_v2.train import train
    from chess_icl_v2.common import read_json
    config={'track':'tiny','mode':'pilot','tokenizer':'square_move_v1',
      'architecture':{'vocab_size':337,'width':16,'layers':1,'heads':2},'seed':3,
      'device':'cpu','steps':1,'eval_interval':1,'learning_rate':.01,'max_tokens':1024,
      'k_values':[0],'train_variants':['standard'],'strict_episode_coverage':True}
    def dataset(variant):
        return {'demonstrations':[],'queries':[
            {'id':split,'family_id':split,'split':split,'variant':variant,
             'state':capture(variant,make_board(variant)).to_dict(),
             'label':{'target_wdl':[.1,.8,.1],'policy_id':'uniform_legal_v1'}} for split in ('train','validation')]}
    parent=tmp_path/'warmup';child=tmp_path/'child'
    train(dataset('standard'),config,parent)
    child_config={**config,'train_variants':['diagonal_rook']}
    train(dataset('diagonal_rook'),child_config,child,parent/'best.pt')
    manifest=read_json(child/'manifest.json')
    assert manifest['exposed_variants']==['diagonal_rook','standard']
    assert manifest['lineage']['sha256']==read_json(parent/'manifest.json')['checkpoint_sha256']
    with (parent/'best.pt').open('ab') as f:f.write(b'x')
    with pytest.raises(ValueError,match='Warmup checkpoint hash mismatch'):
        train(dataset('diagonal_rook'),child_config,tmp_path/'bad',parent/'best.pt')


def test_compact_near_index_matches_exhaustive_hamming():
    import random
    from chess_icl_v2.near import NearIndex
    rng=random.Random(31);items=[(bool(i%2),bytes(rng.choice(b'.PNBRQKpnbrqk') for _ in range(64))) for i in range(120)]
    idx=NearIndex(2)
    for item in items:idx.add(item)
    for side,original in items[::3]:
        for changes in range(5):
            changed=bytearray(original)
            for square in rng.sample(range(64),changes):changed[square]=ord('.') if changed[square]!=ord('.') else ord('P')
            query=(side,bytes(changed))
            expected=any(s==side and sum(a!=b for a,b in zip(board,query[1]))<=2 for s,board in items)
            assert idx.has_near(query)==expected


def test_parallel_generator_matches_reference(tmp_path):
    from chess_icl_v2.data import generate
    from chess_icl_v2.scalable import generate as parallel
    cfg={'seed':101,'variants':['standard','antichess'],'policy_id':'uniform_legal_v1','max_plies':12,
         'replicates':2,'demo_families_per_split':1,'query_families_per_split':1,'near_distance':2,'workers':1}
    a=generate(cfg);b=parallel(cfg,tmp_path/'parallel.json')
    assert [q['id'] for q in a['queries']]==[q['id'] for q in b['queries']]
    assert a['exclusions']==b['exclusions']
    for x,y in zip(a['queries'],b['queries']):
        assert x['state']==y['state']
        assert x['label']['counts_wdl']==y['label']['counts_wdl']
        assert [r['moves'] for r in x['label']['runs']]==[r['moves'] for r in y['label']['runs']]


def test_exploratory_seal_preserves_data_quality_gates():
    from chess_icl_v2.protocol import can_seal
    required=['all_requested_rules_present','simulator_audits','dataset_integrity',
      'no_censored_targets','minimum_replicates','both_sides_all_folds','measured_h100_profile','outcome_variation_two_familiar_rules']
    gate={'passed':False,'checks':{key:True for key in required}}
    gate['checks']['pilot_context_gain']=False
    assert not can_seal(gate) and can_seal(gate,exploratory=True)
    gate['checks']['no_censored_targets']=False
    assert not can_seal(gate,exploratory=True)
    assert not can_seal({},exploratory=True)


def test_relevance_gain_is_paired_against_mismatch():
    from chess_icl_v2.metrics import paired_context_difference
    root={'query_id':'r','model_seed':42,'draw':0,'k':1,'variant':'standard','family_id':'f','target':[1.,0.,0.]}
    rows=[{**root,'condition':'relevant','prediction':[.8,.1,.1]},
          {**root,'condition':'mismatched','prediction':[.4,.3,.3]}]
    assert paired_context_difference(rows)[0]['gain']==pytest.approx(np.log(2))
    rows[1]['draw']=1
    with pytest.raises(ValueError):paired_context_difference(rows)
