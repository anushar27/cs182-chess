"""H100 one-update, save/reload, deterministic prediction and length/memory probe."""
import argparse
import gc
from pathlib import Path
import tempfile
import time
import torch
from chess_icl_v2.common import read_json,write_json,provenance
from chess_icl_v2.models import QwenProbabilityModel,soft_ce
from chess_icl_v2.train import weights,restore


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',default='configs/qwen_revision.json');p.add_argument('--output',required=True);p.add_argument('--lengths',nargs='+',type=int,default=[4096,8192,16384,24576,32768]);a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('No CUDA available; no GPU profile fabricated')
    torch.manual_seed(42)
    cfg=read_json(a.model)
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(cfg['model_id'],revision=cfg['revision'],trust_remote_code=False)
    model=QwenProbabilityModel(**cfg).to('cuda')
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    results=[]
    for length in a.lengths:
        if length>model.context_limit:
            results.append({'tokens':length,'status':'exceeds_native_context'});continue
        try:
            torch.cuda.reset_peak_memory_stats();model.train()
            ids=torch.ones((1,length),device='cuda',dtype=torch.long)
            mask=torch.ones_like(ids);target=torch.tensor([[.2,.3,.5]],device='cuda')
            torch.cuda.synchronize();start=time.perf_counter()
            optimizer.zero_grad(set_to_none=True);loss=soft_ce(model(ids,mask),target);loss.backward();optimizer.step()
            torch.cuda.synchronize()
            results.append({'tokens':length,'status':'passed','seconds':time.perf_counter()-start,
                            'loss':loss.item(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                            'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
        except torch.cuda.OutOfMemoryError:
            results.append({'tokens':length,'status':'out_of_memory'})
            optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        write_json(a.output,{'provenance':provenance(cfg),'device_name':torch.cuda.get_device_name(),
                             'native_context':model.context_limit,'rows':results,'passed':False})
    model.eval()
    ids=torch.ones((1,64),device='cuda',dtype=torch.long);mask=torch.ones_like(ids)
    with torch.inference_mode():one=model(ids,mask).cpu();two=model(ids,mask).cpu()
    with tempfile.TemporaryDirectory() as temp:
        ckpt=Path(temp)/'adapter.pt';torch.save(weights(model),ckpt)
        del optimizer,model;gc.collect();torch.cuda.empty_cache()
        loaded=QwenProbabilityModel(**cfg).to('cuda');restore(loaded,ckpt);loaded.eval()
        with torch.inference_mode():three=loaded(ids,mask).cpu()
    deterministic=torch.equal(one,two) and torch.equal(one,three)
    report={'provenance':provenance(cfg),'device_name':torch.cuda.get_device_name(),
            'rows':results,'save_reload_deterministic':deterministic,
            'passed':deterministic and any(r['status']=='passed' for r in results)}
    write_json(a.output,report)


if __name__=='__main__': main()
