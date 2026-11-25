import argparse
from chess_icl_v2.common import read_json,write_json
from chess_icl_v2.data import summarize
from chess_icl_v2.scalable import generate

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    d=generate(read_json(a.config),a.output);write_json(a.output+'.summary.json',summarize(d))
