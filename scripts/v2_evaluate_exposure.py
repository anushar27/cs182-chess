import os,subprocess,sys,time
from pathlib import Path
from chess_icl_v2.common import read_json,write_json

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1];gpu=read_json(root/'results/direct_exposure_gpu.json')['gpu']
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HOME=os.environ.get('HF_HOME',str(root/'work/hf-cache')),
      HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(root))
    data='results/terminal_evaluation_rules_v2.dataset.json';checkpoint='results/qwen_direct_exposure_seed42/best.pt';seal='results/seals/direct_exposure_seed42.json'
    subprocess.run([sys.executable,'-m','chess_icl_v2.cli','seal','--exploratory','--data',data,'--checkpoint',checkpoint,
      '--protocol','configs/v2_direct_exposure_protocol.json','--gates','results/final_exploratory_quality_gates.json','--output',seal],cwd=root,env=env,check=True)
    for name,coverage in [('test',False),('test_coverage',True)]:
        cmd=[sys.executable,'-u','-m','chess_icl_v2.cli','evaluate','--data',data,'--checkpoint',checkpoint,'--split','test',
           '--k','0','1','--draws','5','--seal',seal,'--output',f'results/direct_exposure_{name}_seed42.json']
        if coverage:cmd+=['--coverage-only']
        start=time.perf_counter()
        with (root/f'results/direct_exposure_{name}_seed42.log').open('w') as f:subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        write_json(root/f'results/direct_exposure_{name}_timing.json',{'seconds':time.perf_counter()-start,'gpu':gpu})
