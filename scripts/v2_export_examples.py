from pathlib import Path
from transformers import AutoTokenizer
from chess_icl_v2.common import read_json,write_json
from chess_icl_v2.episodes import build_episode
from chess_icl_v2.state import State

if __name__=='__main__':
    d=read_json('results/terminal_evaluation_rules_v2.dataset.json');c=read_json('configs/qwen_revision.json')
    t=AutoTokenizer.from_pretrained(c['model_id'],revision=c['revision']);out=Path('examples');out.mkdir(exist_ok=True)
    for variant in d['provenance']['config']['variants']:
        q=sorted([q for q in d['queries'] if q['variant']==variant and q['split']=='validation'],key=lambda q:q['id'])[0]
        ep=build_episode(q,d['demonstrations'],t,1,32768,'relevant',0)
        (out/f'{variant}.input.txt').write_text(ep.text+'\n')
        write_json(out/f'{variant}.audit.json',{'query_id':q['id'],'policy':'terminal_aware_v1','rules_version':q['state']['rules_version'],
            'root_player':'white' if State.from_dict(q['state']).geometry[0] else 'black','counts_wdl':q['label']['counts_wdl'],
            'target_wdl':q['label']['target_wdl'],'R':32,'tokens':len(ep.input_ids),'private_metadata_not_in_input':ep.metadata})
    print('Exported',len(d['provenance']['config']['variants']),'complete examples')
