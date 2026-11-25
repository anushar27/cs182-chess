"""Freeze three trained checkpoints, then run the declared evaluation cohorts."""
import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from chess_icl_v2.common import read_json,write_json


def main():
    root=Path(__file__).resolve().parents[1];mapping=read_json(root/'results/cluster_matrix_status.json')['gpu_mapping']
    status={'data':'terminal_evaluation_rules_v2.dataset.json','runs':{}};lock=threading.Lock()
    def update(key,value):
        with lock:status['runs'][key]=value;write_json(root/'results/evaluation_matrix_status.json',status)
    def run(seed):
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(mapping[str(seed)]),CUBLAS_WORKSPACE_CONFIG=':4096:8',
            HF_HOME=os.environ.get('HF_HOME',str(root/'work/hf-cache')),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
            OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(root))
        data='results/terminal_evaluation_rules_v2.dataset.json';checkpoint=f'results/qwen_familiar_seed{seed}/best.pt'
        seal=f'results/seals/qwen_seed{seed}.json'
        if not (root/seal).exists():
            subprocess.run([sys.executable,'-m','chess_icl_v2.cli','seal','--exploratory','--data',data,'--checkpoint',checkpoint,
              '--protocol','configs/v2_frozen_test_protocol.json','--gates','results/final_exploratory_quality_gates.json',
              '--output',seal],cwd=root,env=env,check=True)
        for name,split,coverage in [('validation','validation',False),('test','test',False),('test_coverage','test',True)]:
            key=f'{name}_seed{seed}';output=f'results/qwen_{name}_seed{seed}.json'
            update(key,{'status':'running','started_unix':time.time(),'gpu':mapping[str(seed)]})
            cmd=[sys.executable,'-u','-m','chess_icl_v2.cli','evaluate','--data',data,'--checkpoint',checkpoint,
                 '--split',split,'--k','0','1','--draws','5','--output',output]
            if split=='test':cmd+=['--seal',seal]
            if coverage:cmd+=['--coverage-only']
            start=time.perf_counter()
            with (root/f'results/qwen_{name}_seed{seed}.log').open('w') as f:
                result=subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
            if result.returncode:
                update(key,{'status':'failed','returncode':result.returncode});raise RuntimeError(key)
            report=read_json(root/output)
            update(key,{'status':'complete','seconds':time.perf_counter()-start,'predictions':len(report['rows']),
                        'excluded_roots':len(report['exclusions']),'gpu':mapping[str(seed)]})
        return seed
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for seed in pool.map(run,(42,123,456)):print('evaluations complete',seed,flush=True)
    status['complete']=True;write_json(root/'results/evaluation_matrix_status.json',status)


if __name__=='__main__':main()
