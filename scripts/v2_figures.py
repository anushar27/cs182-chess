"""Export paper figures from authoritative retained data, never from illustrative values."""
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from chess_icl_v2.common import read_json,write_json

ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'savefig.bbox':'tight','pdf.fonttype':42,'svg.fonttype':'none'})


def save(fig,name):
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    for suffix in ('png','pdf','svg'):fig.savefig(out/f'{name}.{suffix}',dpi=220)
    svg=out/f'{name}.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def main():
    r=ROOT/'results';d=read_json(r/'terminal_policy_rules_v2.dataset.json')
    variants=d['provenance']['config']['variants'];values=[]
    for variant in variants:
        qs=[q for q in d['queries'] if q['variant']==variant and q['split']!='test']
        values.append(np.mean([q['label']['target_wdl'] for q in qs],axis=0))
    values=np.array(values);fig,ax=plt.subplots(figsize=(8,7));left=np.zeros(len(values))
    for i,(name,color) in enumerate(zip(('Win','Draw','Loss'),('#1b9e77','#8d99ae','#d95f02'))):
        ax.barh(np.arange(len(values)),values[:,i],left=left,label=name,color=color,height=.7);left+=values[:,i]
    ax.set_yticks(np.arange(len(values)),[v.replace('_',' ') for v in variants]);ax.invert_yaxis();ax.set_xlim(0,1)
    ax.set_xlabel('Mean root-player probability, empirical R=32');ax.set_title('Development outcome diversity under terminal-aware play')
    ax.legend(ncol=3,loc='lower center',bbox_to_anchor=(.5,-.18),frameon=False);save(fig,'policy_outcome_diversity')
    profile=read_json(r/'h100_profile.json');rows=[x for x in profile['rows'] if x['status']=='passed']
    fig,ax=plt.subplots(figsize=(6.8,4))
    for key,label,style in [('peak_allocated_bytes','Allocated','o-'),('peak_reserved_bytes','Reserved (allocator cache)','s--')]:
        ax.plot([x['tokens']/1024 for x in rows],[x[key]/2**30 for x in rows],style,label=label)
    ax.set(xlabel='Context length (1,024-token units)',ylabel='Peak GPU memory (GiB)',title='Qwen-7B LoRA: measured H100 update profile')
    ax.set_xticks([4,8,16,24,32]);ax.legend(frameon=False);ax.grid(alpha=.2);save(fig,'h100_context_memory')
    path=r/'qwen_test_aggregate.json'
    if path.exists():
        result=read_json(path);groups=[g for g in ('familiar_rules','heldout_rules','standard_control') if f'{g}|relevant|1' in result['gains']]
        fig,ax=plt.subplots(figsize=(7,4.3))
        for offset,condition,color in [(-.10,'relevant','#1b9e77'),(.10,'mismatched','#7570b3')]:
            entries=[result['gains'][f'{g}|{condition}|1'] for g in groups]
            y=np.array([v['macro_gain'] for v in entries]);low=np.array([v['ci95'][0] for v in entries]);high=np.array([v['ci95'][1] for v in entries])
            ax.errorbar(np.arange(len(groups))+offset,y,yerr=[y-low,high-y],fmt='o',capsize=4,label=condition,color=color)
        ax.axhline(0,color='#555555',linewidth=1);ax.set_xticks(np.arange(len(groups)),[g.replace('_',' ') for g in groups])
        ax.set(ylabel='Macro paired NLL gain over K=0 (nats)',title='Frozen Qwen test: K=1 context gain')
        ax.legend(frameon=False);ax.grid(axis='y',alpha=.2);save(fig,'test_context_gain')
        if result.get('relevance_gains'):
            entries=[result['relevance_gains'][f'{g}|1'] for g in groups];fig,ax=plt.subplots(figsize=(7,4))
            y=np.array([v['macro_gain'] for v in entries]);lo=np.array([v['ci95'][0] for v in entries]);hi=np.array([v['ci95'][1] for v in entries])
            ax.errorbar(np.arange(len(groups)),y,yerr=[y-lo,hi-y],fmt='o',capsize=4,color='#1b9e77')
            ax.axhline(0,color='#555555',linewidth=1);ax.set_xticks(np.arange(len(groups)),[g.replace('_',' ') for g in groups])
            ax.set(ylabel='NLL(mismatched) − NLL(relevant)',title='Rule relevance beyond matched context length');ax.grid(axis='y',alpha=.2)
            save(fig,'test_relevance_gain')
    paths=[r/f'qwen_test_coverage_seed{s}.json' for s in (42,123,456)]
    if all(p.exists() for p in paths):
        bin_records=[]
        by_key=defaultdict(list);targets={};sides={}
        for p in paths:
            for row in read_json(p)['rows']:
                by_key[(row['query_id'],row['condition'])].append(row['prediction']);targets[row['query_id']]=row['target'];sides[row['query_id']]=row['side']
        fig,axes=plt.subplots(2,3,figsize=(10,6.6),sharex=True,sharey=True)
        for i,side in enumerate(('white','black')):
            for c,cl in enumerate(('Win','Draw','Loss')):
                ax=axes[i,c];ax.plot([0,1],[0,1],color='#999999',linestyle='--',linewidth=1)
                for condition,color in [('zero','#7570b3'),('relevant','#1b9e77')]:
                    x=[];y=[]
                    records=[(np.mean(pred,axis=0)[c],targets[qid][c]) for (qid,cond),pred in by_key.items() if cond==condition and sides[qid]==side]
                    for b in range(10):
                        selected=[(p,t) for p,t in records if min(int(p*10),9)==b]
                        if selected:
                            x.append(np.mean([p for p,t in selected]));y.append(np.mean([t for p,t in selected]))
                            bin_records.append({'side':side,'class':cl,'condition':condition,'bin':b,'roots':len(selected),
                                                'mean_prediction':float(x[-1]),'mean_target':float(y[-1])})
                    ax.plot(x,y,'o-',color=color,label='K=0' if condition=='zero' else 'Relevant K=1',markersize=4)
                ax.set_title(f'{side.title()} to move · {cl}');ax.set_xlim(0,1);ax.set_ylim(0,1);ax.grid(alpha=.15)
                if i==1:ax.set_xlabel('Mean predicted probability')
                if c==0:ax.set_ylabel('Mean empirical probability')
        axes[0,0].legend(frameon=False,fontsize=8);fig.suptitle('Descriptive calibration: three-seed/context-mean predictor, R=32 targets',y=1.02)
        fig.tight_layout();save(fig,'test_calibration_ensemble')
        write_json(r/'calibration_ensemble_bins.json',{'description':'Average probabilities over three model seeds and, for K1, five context draws per root; ten equal-width bins. Descriptive ensemble, not the primary mean-of-seed NLL. No uncertainty intervals; sparse-bin counts must accompany interpretation.','bins':bin_records})


if __name__=='__main__':main()
