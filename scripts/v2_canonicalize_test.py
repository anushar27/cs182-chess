"""Re-score the exact saved test inputs with canonical batch1; retain padded diagnostics."""
import argparse
from collections import defaultdict
from pathlib import Path
import shutil,time
import torch
from chess_icl_v2.canonical import CanonicalCache
from chess_icl_v2.common import digest,read_json,write_json,provenance
from chess_icl_v2.episodes import build_episode
from chess_icl_v2.metrics import aggregate,paired_gains,hierarchical_ci,calibration
from chess_icl_v2.models import Predictor
from chess_icl_v2.train import model_and_tokenizer,restore,file_hash


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:groups[(row['exposure'],row['condition'],row['k'])].append(row)
    metrics={}
    for key,value in groups.items():
        metrics['|'.join(map(str,key))]={'all':aggregate(value),
            'equal_material':aggregate([r for r in value if abs(r['material'])<=2]),
            'white':aggregate([r for r in value if r['side']=='white']),
            'black':aggregate([r for r in value if r['side']=='black']),
            'calibration':calibration(value),
            'calibration_by_side':{side:calibration([r for r in value if r['side']==side]) for side in ('white','black')}}
    gains={}
    for exposure in ('familiar_rules','heldout_rules','standard_control'):
        cohort=[r for r in rows if r['exposure']==exposure]
        for condition in ('relevant','mismatched'):
            gains[f'{exposure}|{condition}|1']=hierarchical_ci(paired_gains(cohort,condition,1),replicates=300)
    return metrics,gains


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--direct',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[1];r=root/'results';start=time.perf_counter()
    prefix='direct_exposure' if a.direct else 'qwen';folder=f'qwen_direct_exposure_seed{a.seed}' if a.direct else f'qwen_familiar_seed{a.seed}'
    manifest=read_json(r/folder/'manifest.json');config=manifest['config'];torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    diag=r/'padded_batch_diagnostics';diag.mkdir(exist_ok=True)
    names=[f'{prefix}_test_seed{a.seed}.json',f'{prefix}_test_coverage_seed{a.seed}.json']
    reports=[]
    for name in names:
        source=diag/name
        if not source.exists():shutil.copy2(r/name,source)
        reports.append(read_json(source))
    data=read_json(r/'terminal_evaluation_rules_v2.dataset.json');queries={q['id']:q for q in data['queries']};games={g['family_id']:g for g in data['demonstrations']}
    if any(report['dataset_sha256']!=digest(data) for report in reports):raise ValueError('Saved input dataset changed')
    model,tok=model_and_tokenizer(config);checkpoint=r/folder/'best.pt'
    if file_hash(checkpoint)!=manifest['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    restore(model,checkpoint);predictor=Predictor(model,config['device'],tok.pad_token_id)
    versions={n:v._version for n,v in model.named_parameters()}
    fingerprint=digest({'checkpoint':manifest['checkpoint_sha256'],'architecture':config['architecture'],
        'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),'policy':'unpadded_batch1_v1'})
    cache=CanonicalCache(predictor,fingerprint,r/f'canonical_{prefix}_seed{a.seed}.cache.json')
    unique={row['input_sha256']:row for report in reports for row in report['rows']}
    values={};torch.cuda.reset_peak_memory_stats();forward_seconds=0.
    first_episode=None;first_probability=None
    for i,(sha,row) in enumerate(sorted(unique.items(),key=lambda pair:pair[1]['tokens'])):
        q=queries[row['query_id']]
        pool=[games[f] for f in row['demo_families']] if row['condition']=='mismatched' else data['demonstrations']
        ep=build_episode(q,pool,tok,row['k'],config['max_tokens'],row['condition'],row['draw'])
        if ep.metadata['input_sha256']!=sha or len(ep.input_ids)!=row['tokens']:raise ValueError('Reconstructed input differs from saved input')
        before=time.perf_counter();values[sha]=cache.predict_proba(ep);forward_seconds+=time.perf_counter()-before
        if first_episode is None:first_episode=ep;first_probability=values[sha]
        if (i+1)%128==0:
            cache.save();write_json(r/f'canonical_{prefix}_seed{a.seed}.progress.json',{'completed':i+1,'unique_inputs':len(unique),'seconds':time.perf_counter()-start})
            print(prefix,a.seed,i+1,'/',len(unique),flush=True)
    cache.save()
    if predictor.predict_proba(first_episode)!=first_probability:raise ValueError('Canonical single-episode prediction changed')
    if any(v.requires_grad or v._version!=versions[n] for n,v in model.named_parameters()):raise ValueError('Weights changed during canonical inference')
    max_delta=0.;sum_delta=0.;n_delta=0
    for name,report in zip(names,reports):
        for row in report['rows']:
            new=values[row['input_sha256']]
            differences=[abs(a-b) for a,b in zip(row['prediction'],new)]
            max_delta=max(max_delta,*differences);sum_delta+=sum(differences);n_delta+=3
            row['prediction']=new
        report['metrics'],report['context_gains']=summarize(report['rows'])
        report['inference_policy']='canonical_unpadded_batch1_cached_by_token_ids'
        report['canonicalization_provenance']=provenance({'fingerprint':fingerprint,'parent_padded_report_sha256':file_hash(diag/name)})
        write_json(r/name,report)
    timing={'unique_inputs':len(unique),'new_forward_calls':cache.new_predictions,'total_seconds':time.perf_counter()-start,
        'forward_and_transfer_seconds':forward_seconds,'max_change_from_padded_probability':max_delta,
        'mean_change_from_padded_probability':sum_delta/max(1,n_delta),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'weights_and_repeat_verified':True,'fingerprint':fingerprint}
    write_json(r/f'canonical_{prefix}_seed{a.seed}.timing.json',timing)
    write_json(r/f'canonical_{prefix}_seed{a.seed}.progress.json',{'status':'complete','completed':len(unique),
        'unique_inputs':len(unique),'seconds':timing['total_seconds']})
    print(timing,flush=True)


if __name__=='__main__':main()
