from types import SimpleNamespace
import pytest
from chess_icl_v2.canonical import CanonicalCache
from chess_icl_v2.common import read_json, write_json


class FrozenPredictor:
    def __init__(self):self.calls=0
    def predict_proba(self,episode):
        self.calls+=1
        return [.2,.3,.5]


def test_same_input_across_targets_and_reload_uses_identical_prediction(tmp_path):
    predictor=FrozenPredictor();path=tmp_path/'cache.json'
    cache=CanonicalCache(predictor,'model-A',path)
    episode=SimpleNamespace(input_ids=[1,2,3],target=[1.,0.,0.])
    assert cache.predict_proba(episode)==[.2,.3,.5]
    episode.target=[0.,0.,1.]
    value=cache.predict_proba(episode);value[0]=9
    assert cache.predict_proba(episode)==[.2,.3,.5]
    cache.save();loaded=CanonicalCache(predictor,'model-A',path)
    assert loaded.predict_proba(episode)==[.2,.3,.5]
    assert predictor.calls==1
    loaded.predict_proba(SimpleNamespace(input_ids=[1,3,2]))
    assert predictor.calls==2
    with pytest.raises(ValueError,match='mismatch'):CanonicalCache(predictor,'model-B',path)


def test_corrupt_cache_is_rejected(tmp_path):
    path=tmp_path/'cache.json';cache=CanonicalCache(FrozenPredictor(),'A',path)
    cache.predict_proba(SimpleNamespace(input_ids=[1]));cache.save()
    data=read_json(path);data['entries'][next(iter(data['entries']))]=[.2,.3,.7]
    write_json(path,data)
    with pytest.raises(ValueError,match='Invalid WDL'):CanonicalCache(FrozenPredictor(),'A',path)


def test_evaluator_reuses_input_without_using_target(tmp_path):
    import torch
    from chess_icl_v2.evaluate import evaluate
    from chess_icl_v2.models import TinyTransformer
    from chess_icl_v2.rules import make_board
    from chess_icl_v2.state import capture
    from chess_icl_v2.train import weights,file_hash
    config={'track':'tiny','architecture':{'vocab_size':337,'width':16,'layers':1,'heads':2},
            'seed':3,'device':'cpu','max_tokens':1024,'train_variants':['standard']}
    checkpoint=tmp_path/'best.pt';torch.save(weights(TinyTransformer(**config['architecture'])),checkpoint)
    write_json(tmp_path/'manifest.json',{'config':config,'checkpoint_sha256':file_hash(checkpoint)})
    state=capture('standard',make_board('standard')).to_dict()
    data={'provenance':{'config':{'policy_id':'uniform_legal_v1','variants':['standard']}},
          'demonstrations':[],'queries':[
              {'id':str(i),'family_id':str(i),'variant':'standard','split':'validation','state':state,
               'label':{'target_wdl':target,'policy_id':'uniform_legal_v1','replicates':32,
                        'mc_standard_error':[0.,0.,0.]}}
              for i,target in enumerate(([1.,0.,0.],[0.,0.,1.]))]}
    report=evaluate(data,checkpoint,tmp_path/'evaluation.json',k_values=(0,),draws=1)
    assert report['frozen_weights'] and report['inference_policy'].startswith('canonical_unpadded')
    assert report['rows'][0]['prediction']==report['rows'][1]['prediction']
    assert report['rows'][0]['target']!=report['rows'][1]['target']
    cache=read_json(tmp_path/'evaluation.inference_cache.json')
    assert len(cache['entries'])==1
