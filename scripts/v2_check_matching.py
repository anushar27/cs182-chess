import argparse
from collections import Counter
from transformers import AutoTokenizer
from chess_icl_v2.common import read_json,write_json
from chess_icl_v2.episodes import paired_episodes

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--split',default='validation');p.add_argument('--output',required=True);a=p.parse_args()
    cfg=read_json('configs/qwen_revision.json');tok=AutoTokenizer.from_pretrained(cfg['model_id'],revision=cfg['revision'])
    d=read_json(a.data);rows=[]
    for q in d['queries']:
        if q['split']!=a.split:continue
        try:
            for draw in range(5):paired_episodes(q,d['demonstrations'],tok,1,32768,draw)
            rows.append({'query_id':q['id'],'variant':q['variant'],'passed':True})
        except ValueError as e:rows.append({'query_id':q['id'],'variant':q['variant'],'passed':False,'reason':str(e)})
    write_json(a.output,{'split':a.split,'rows':rows})
    print(Counter((r['variant'],r['passed']) for r in rows))
