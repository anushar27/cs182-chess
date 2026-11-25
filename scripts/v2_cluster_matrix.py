"""One independent H100 process per seed; preserve all job logs and common parents."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from chess_icl_v2.common import read_json,write_json


def main():
    root=Path(__file__).resolve().parents[1]
    csv=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,memory.used','--format=csv,noheader,nounits'],text=True)
    devices=sorted((int(used),int(index),int(free)) for index,free,used in (line.split(',') for line in csv.strip().splitlines()))
    chosen=[index for used,index,free in devices if free>48000][:3]
    if len(chosen)<3:raise RuntimeError('Need three H100 devices with >48GB free each')
    status={'started_unix':time.time(),'gpu_mapping':dict(zip(map(str,(42,123,456)),chosen)),'runs':{}}
    lock=threading.Lock()
    def update(key,value):
        with lock:
            status['runs'][key]=value;write_json(root/'results/cluster_matrix_status.json',status)
    def run(seed,gpu):
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',
             OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
             HF_HOME=os.environ.get('HF_HOME',str(root/'work/hf-cache')),PYTHONPATH=str(root))
        parent=root/f'results/qwen_warmup_seed{seed}'
        for phase,config,out in [('warmup',f'configs/v2_qwen_cluster_warmup_seed{seed}.json',parent),
                                 ('familiar',f'configs/v2_qwen_exploratory_seed{seed}.json',root/f'results/qwen_familiar_seed{seed}')]:
            key=f'{phase}_seed{seed}'
            if (out/'manifest.json').exists():
                if read_json(out/'manifest.json')['config']!=read_json(root/config):raise ValueError('Existing config mismatch')
                update(key,{'status':'already_complete','gpu':gpu});continue
            update(key,{'status':'running','gpu':gpu,'started_unix':time.time()})
            cmd=[sys.executable,'-u','-m','chess_icl_v2.cli','train','--data','results/terminal_policy.dataset.json',
                 '--config',config,'--output',str(out)]
            if phase=='familiar':cmd+=['--initial-checkpoint',str(parent/'best.pt')]
            with (root/f'results/qwen_{phase}_seed{seed}.log').open('w') as log:
                result=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                update(key,{'status':'failed','gpu':gpu,'returncode':result.returncode});raise RuntimeError(key)
            manifest=read_json(out/'manifest.json')
            update(key,{'status':'complete','gpu':gpu,'seconds':manifest['seconds'],
                        'validation_nll':manifest['best_macro_validation_nll'],'checkpoint_sha256':manifest['checkpoint_sha256']})
        return seed
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for seed in pool.map(lambda pair:run(*pair),zip((42,123,456),chosen)):
            print('completed seed',seed,flush=True)
    status['complete']=True;write_json(root/'results/cluster_matrix_status.json',status)


if __name__=='__main__':main()
