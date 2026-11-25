import os,subprocess,sys
from pathlib import Path
from chess_icl_v2.common import write_json,read_json

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
    occupied=set(read_json(root/'results/cluster_matrix_status.json')['gpu_mapping'].values())
    choices=sorted((int(used),int(index)) for index,used,free in (s.split(',') for s in lines.strip().splitlines()) if int(index) not in occupied and int(free)>48000)
    gpu=choices[0][1]
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HOME=os.environ.get('HF_HOME',str(root/'work/hf-cache')),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(root))
    write_json(root/'results/direct_exposure_gpu.json',{'gpu':gpu,'seed':42,'condition':'all-rule exposure reference'})
    with (root/'results/qwen_direct_exposure_seed42.log').open('w') as f:
        subprocess.run([sys.executable,'-u','-m','chess_icl_v2.cli','train','--data','results/terminal_policy_rules_v2.dataset.json',
          '--config','configs/v2_qwen_direct_exposure_seed42.json','--output','results/qwen_direct_exposure_seed42',
          '--initial-checkpoint','results/qwen_warmup_seed42/best.pt'],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
