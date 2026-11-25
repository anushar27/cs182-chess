"""Export all completed test comparisons and a manuscript-ready exploratory record."""
from collections import Counter,defaultdict
from pathlib import Path
import csv
from chess_icl_v2.common import read_json,write_json,digest
from chess_icl_v2.metrics import aggregate,score,hierarchical_ci,calibration

ROOT=Path(__file__).resolve().parents[1]
GROUPS=('familiar_rules','heldout_rules','standard_control')


def grouped(rows):
    groups=defaultdict(list)
    for row in rows:groups[(row['exposure'],row['condition'],row['k'])].append(row)
    return {'|'.join(map(str,k)):aggregate(v) for k,v in groups.items()}


def export_table(name,fields,rows):
    out=ROOT/'tables';out.mkdir(exist_ok=True)
    with (out/f'{name}.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    escape=lambda x:str(x).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
    lines=[r'\begin{tabular}{'+'l'*len(fields)+'}',r'\hline',' & '.join(map(escape,fields))+r' \\',r'\hline']
    for row in rows:lines.append(' & '.join(escape(f'{row[k]:.6f}' if isinstance(row[k],float) else row[k]) for k in fields)+r' \\')
    lines.extend([r'\hline',r'\end{tabular}'])
    (out/f'{name}.tex').write_text('\n'.join(lines)+'\n')


def main():
    r=ROOT/'results';data=read_json(r/'terminal_evaluation_rules_v2.dataset.json')
    qa=read_json(r/'final_artifact_validation.json')
    if not qa['passed']:raise ValueError('Final artifact audit required')
    reports=[read_json(r/f'qwen_test_seed{s}.json') for s in (42,123,456)]
    rows=[row for report in reports for row in report['rows']]
    coverage=[row for s in (42,123,456) for row in read_json(r/f'qwen_test_coverage_seed{s}.json')['rows']]
    if any(not report.get('inference_policy','').startswith('canonical_unpadded') for report in reports):raise ValueError('Canonical inference required')
    agg=read_json(r/'qwen_test_aggregate.json');high=read_json(r/'high_r_analysis.json')
    baselines=read_json(r/'qwen_test_baselines.json')
    metrics=[];per_rule=[]
    for cohort,values in [('matched',agg['metrics']),('coverage',read_json(r/'qwen_test_coverage_aggregate.json')['metrics'])]:
        for key,m in sorted(values.items()):
            exposure,condition,k=key.split('|')
            metrics.append({'cohort':cohort,'exposure':exposure,'condition':condition,'K':int(k),
                            'rows':m['n'],**m['macro']})
            for rule,measured in m['per_variant'].items():
                per_rule.append({'cohort':cohort,'rule':rule,'exposure':exposure,'condition':condition,'K':int(k),**measured})
    export_table('test_metrics',list(metrics[0]),metrics);export_table('test_per_rule',list(per_rule[0]),per_rule)
    gain_rows=[]
    for kind,keyname in [('over_K0','gains'),('relevance','relevance_gains')]:
        for key,g in sorted(agg[keyname].items()):
            if 'macro_gain' not in g:continue
            gain_rows.append({'comparison':kind,'group_condition':key,'gain_nats':g['macro_gain'],'ci95_low':g['ci95'][0],'ci95_high':g['ci95'][1]})
    export_table('test_paired_gains',list(gain_rows[0]),gain_rows)
    baseline_rows=[]
    for key,m in sorted(baselines['metrics'].items()):
        name,exposure,condition,k=key.split('|')
        baseline_rows.append({'baseline':name,'exposure':exposure,'condition':condition,'K':int(k),**m['macro']})
    export_table('test_baselines',list(baseline_rows[0]),baseline_rows)
    costs=[]
    for variant in data['provenance']['config']['variants']:
        queries=[q for q in data['queries'] if q['variant']==variant]
        runs=[run for q in queries for run in q['label']['runs']]
        seconds=sum(run['seconds'] for run in runs);plies=sum(len(run['moves']) for run in runs)
        costs.append({'rule':variant,'roots':len(queries),'continuations':len(runs),
            'mean_plies':plies/len(runs),'mean_worker_elapsed_seconds':seconds/len(runs),
            'sum_worker_elapsed_seconds':seconds,
            'worker_hours_per_million_at_observed_mean':1e6*seconds/len(runs)/3600})
    export_table('generation_cost',list(costs[0]),costs)
    # Repartition the exposed reference by the primary model's rule sets only for comparison.
    exposure_by_query={row['query_id']:row['exposure'] for row in rows}
    direct=read_json(r/'direct_exposure_test_seed42.json')
    refrows=[{**row,'exposure':exposure_by_query[row['query_id']]} for row in direct['rows']]
    reference=grouped(refrows);reference_table=[]
    for key,m in sorted(reference.items()):
        exposure,condition,k=key.split('|')
        reference_table.append({'primary_model_rule_group':exposure,'condition':condition,'K':int(k),**m['macro']})
    export_table('all_rule_reference',list(reference_table[0]),reference_table)
    baseline_by_key={(row['baseline'],row['query_id'],row['condition'],row['k'],row['draw']):row for row in baselines['rows']}
    baseline_gains={}
    for name in ('query_material','knn_demo_states'):
        for exposure in GROUPS[:2]:
            gains=[]
            for row in rows:
                if row['condition']!='relevant' or row['exposure']!=exposure:continue
                b=baseline_by_key[(name,row['query_id'],row['condition'],row['k'],row['draw'])]
                gains.append({**row,'gain':score(row['target'],b['prediction'])['nll']-score(row['target'],row['prediction'])['nll']})
            baseline_gains[f'{exposure}|{name}']=hierarchical_ci(gains)
    interventions={}
    for exposure in GROUPS:
        relevant={(row['query_id'],row['model_seed'],row['draw']):row for row in rows if row['condition']=='relevant' and row['exposure']==exposure}
        for condition in ('corrupt_outcomes','corrupt_trajectories'):
            gains=[]
            for row in rows:
                if row['condition']!=condition or row['exposure']!=exposure:continue
                rel=relevant[(row['query_id'],row['model_seed'],row['draw'])]
                gains.append({**row,'gain':score(row['target'],row['prediction'])['nll']-score(row['target'],rel['prediction'])['nll']})
            interventions[f'{exposure}|{condition}']=hierarchical_ci(gains)
    # Store every slice, retaining counts rather than silently comparing unequal populations.
    slices={}
    for exposure in GROUPS:
        for condition in ('zero','relevant','mismatched','corrupt_outcomes','corrupt_trajectories'):
            group=[row for row in rows if row['exposure']==exposure and row['condition']==condition]
            slices[f'{exposure}|{condition}']={
                'white':aggregate([row for row in group if row['side']=='white']),
                'black':aggregate([row for row in group if row['side']=='black']),
                'abs_material_le2':aggregate([row for row in group if abs(row['material'])<=2]),
                'calibration_by_side':{side:calibration([row for row in group if row['side']==side]) for side in ('white','black')}}
    seed_metrics={str(s):grouped(report['rows']) for s,report in zip((42,123,456),reports)}
    cohort_counts={}
    for exposure in GROUPS:
        for cohort,rr in [('matched',rows),('coverage',coverage)]:
            q=[row for row in rr if row['exposure']==exposure]
            cohort_counts[f'{exposure}|{cohort}']={'roots':len({row['query_id'] for row in q}),
                'families':len({row['family_id'] for row in q}),'rules':len({row['variant'] for row in q})}
    primary=[]
    for exposure,label in zip(GROUPS,('Familiar (4)','Held-out (9)','Standard')):
        g=agg['gains'][f'{exposure}|relevant|1']
        primary.append({'Rule set':label,'Roots':cohort_counts[f'{exposure}|matched']['roots'],
            'K0 NLL':agg['metrics'][f'{exposure}|zero|0']['macro']['nll'],
            'K1 NLL':agg['metrics'][f'{exposure}|relevant|1']['macro']['nll'],
            'Gain':g['macro_gain'],'95% interval':f"[{g['ci95'][0]:.6f}, {g['ci95'][1]:.6f}]"})
    export_table('primary_results',list(primary[0]),primary)
    analysis={'dataset_sha256':digest(data),'cohort_counts':cohort_counts,'per_seed_metrics':seed_metrics,
        'reference_repartitioned_by_primary_rule_groups':reference,
        'reference_note':'All thirteen variant rules are gradient-exposed in this reference; the group names describe only the primary model.',
        'baseline_paired_advantage':baseline_gains,'intervention_degradation':interventions,'slices':slices,
        'excluded_roots':reports[0]['exclusions'],'baselines_training_roots':baselines['training_roots'],
        'continuation_costs':costs,'cost_note':'Worker elapsed seconds include scheduling and sharing. They are not measured CPU core time. Extrapolation excludes source/demo generation, audit, deduplication and startup. Original generation wall time excludes later repair and demo extensions.'}
    write_json(r/'final_analysis.json',analysis)
    text='''# Fixed-policy chess-variant probability prediction: exploratory study

## Objective and scope

The target is the player-to-move W/D/L distribution when both players follow
`terminal_aware_v1`: choose uniformly among legal immediate wins if any exist,
otherwise uniformly among all legal moves, sorted by UCI before seeded sampling.
The same policy is used for full demonstrations, source games and independent
label continuations. No material evaluation or optimal-play labels enter targets.
The experiment covers thirteen project variants and a separate standard control.

This is a completed small exploratory experiment, not a confirmatory paper result.
The negative scratch pilot was retained; the subsequent H100 experiment explicitly
bypassed only its positive-gain gate under the user's instruction to run the study.
All simulator, split, complete-target, R32, side and hardware gates passed. The
test protocol and selected checkpoints were sealed before test-model comparisons.
The structural test folds had already appeared in development feasibility records.

## Methods and reproducibility

See [exact rules](RULES.md), [implementation contracts](IMPLEMENTATION.md),
[decisions and incidents](DECISIONS.md), [cluster measurements](CLUSTER_RESULTS.md),
[artifact validity](ARTIFACT_STATUS.md), and [runbook](REPRODUCING.md).
Rules were reconstructed from the original repository and the design attachment;
ambiguous choices were frozen and recorded, including royal king behavior,
castling, promotion/capture conversion, color-dependent immunity and draw claims.
The long-board proposal is explicitly an 8×8 initial-pawn-extension rule.

Full replay states retain initial position, every past move, castling, en passant,
clocks, promotion information and fixed barriers. Inputs contain complete terminal
demonstration trajectories and their White outcomes, then the full query and a
prediction marker. No rule description/name or private family/label information is
included. Overflow rejects the episode; neither the query nor a demo is trimmed.
Query history and visible geometry may themselves reveal a rule at K=0.

Training uses pinned Qwen2.5-7B-Instruct, last-four-layer attention LoRA r8/alpha16,
and a three-way FP32 head: 731,651 trainable parameters. Three seeds (42,123,456)
receive 30 standard warmup updates, then 200 updates on diagonal rook, straight
bishop, weakened queen and extended pawn, with K=0/1, accumulation4, LR1e-4 and
20-update main LR warmup. Checkpoints minimize macro validation soft NLL over
the four training rules; no checkpoint was selected using test scores. The other
nine variant rules receive no gradients in these models. The one-seed all-rule
reference starts from the same standard warmup and receives 200 variant updates.
Its exposure and number of distinct labeled roots differ; it is not a causal
estimate of exposure alone or an upper bound.

The corrected pool contains 614 roots (206 train,197 validation,211 test), each
with 32 independent continuations, and 1,000 full demonstration games after
evaluation-only additions. The pools and query-source families are separate;
cross-fold roots and query-vs-demo states pass the stated Hamming-distance audit.
All 19,648 R32 continuations terminated within 2,560 plies. Unfinished runs would
produce null targets and marginal identification bounds, never automatic draws.
Eight source/demo families were requested per fold/rule; hash eligibility70/15/15
does not imply that the final quotas have a70/15/15 ratio.

Primary test comparisons use the same206 roots across K0, relevantK1, foreign-rule
K1 and both corruption controls, with five draws per context. Foreign demos must
match complete encoded demonstration length within10%. Five roots fail matching;
the separate coverage cohort retains all211 roots for K0/relevantK1. Weights stay
frozen. After discovering batch-dependent BF16 probabilities (up to about0.044),
we rescored exactly the saved inputs individually without padding. Original input
hashes, root cohorts, matching, targets and selected weights are unchanged; old
padded outputs remain diagnostics. Identical token sequences use the same output.
Validation checkpoint selection retains the original padded procedure. This
numerical analysis amendment was recorded after the first test pass, before final
comparative claims, and must not be described as preregistered.

Metrics are soft-target NLL and squared-vector Brier, averaged within rule then
equally across rules. Positive gain means lower NLL with relevant context. The
1,000-replicate, seed0 hierarchy resamples model seeds, demo draws and source
families within fixed rules. Its intervals condition on empirical labels; Monte
Carlo label uncertainty is assessed separately. Only three model seeds and few
families per rule limit inference. All intervals and ablations are exploratory;
no multiplicity-adjusted significance claim is made.

## Frozen matched-cohort results

| Rule set | Roots/families | K0 NLL | Relevant K1 NLL | Mismatched K1 NLL | Relevant Brier |
|---|---:|---:|---:|---:|---:|
'''
    for exposure in GROUPS:
        c=cohort_counts[f'{exposure}|matched'];z=agg['metrics'][f'{exposure}|zero|0']['macro'];rel=agg['metrics'][f'{exposure}|relevant|1']['macro'];mis=agg['metrics'][f'{exposure}|mismatched|1']['macro']
        text+=f"| {exposure} | {c['roots']}/{c['families']} | {z['nll']:.6f} | {rel['nll']:.6f} | {mis['nll']:.6f} | {rel['brier']:.6f} |\n"
    text+='\n| Rule set | Relevant gain over K0 [95% interval] | Relevance advantage over mismatch [95% interval] |\n|---|---|---|\n'
    for exposure in GROUPS:
        g=agg['gains'][f'{exposure}|relevant|1'];v=agg['relevance_gains'][f'{exposure}|1']
        text+=f"| {exposure} | {g['macro_gain']:.6f} [{g['ci95'][0]:.6f}, {g['ci95'][1]:.6f}] | {v['macro_gain']:.6f} [{v['ci95'][0]:.6f}, {v['ci95'][1]:.6f}] |\n"
    text+='\nThe primary group has four familiar rules; transfer has nine gradient-held-out rules. Standard is a separate warmup control. Raw per-seed, per-rule, side, equal-material and calibration records are retained in the machine-readable analysis.\n\n## Baselines and interventions\n\n'
    # This interpretation is tied to the fixed completed study; avoid silently reusing it for a different run.
    fg=agg['gains']['familiar_rules|relevant|1'];hg=agg['gains']['heldout_rules|relevant|1']
    if fg['ci95'][1]<0 and hg['ci95'][0]<0<hg['ci95'][1] and all(
        agg['relevance_gains'][f'{group}|1']['ci95'][0]<0<agg['relevance_gains'][f'{group}|1']['ci95'][1] for group in GROUPS[:2]):
        text+='The completed study does not demonstrate beneficial rule-specific ICL. Relevant context worsens familiar-rule predictions; the held-out estimate is negative but inconclusive. Neither rule group has a resolved relevance advantage over matched foreign-rule context. This is a negative result for this training/data scale, not evidence that the task is impossible.\n\n'
    text+=f"Learned baselines use only {baselines['training_roots']} training roots from exposed rules, including standard warmup. They receive no rule IDs. Baselines cover frequency/majority, material-count regression, length/outcome histograms, demo frequency/majority and demo-state kNN. K1 demo frequency and majority coincide because there is only one game.\n\n"
    text+='| Baseline, relevant K1 | Familiar macro NLL | Held-out macro NLL |\n|---|---:|---:|\n'
    for name in sorted({row['baseline'] for row in baselines['rows']}):
        text+=f"| {name} | {baselines['metrics'][f'{name}|familiar_rules|relevant|1']['macro']['nll']:.6f} | {baselines['metrics'][f'{name}|heldout_rules|relevant|1']['macro']['nll']:.6f} |\n"
    text+='\n| Rule set | Corruption | NLL(corrupt) − NLL(relevant) [95% interval] |\n|---|---|---|\n'
    for key,v in interventions.items():
        exposure,cond=key.split('|');text+=f"| {exposure} | {cond} | {v['macro_gain']:.6f} [{v['ci95'][0]:.6f}, {v['ci95'][1]:.6f}] |\n"
    text+='\nShuffled trajectories are generally illegal and shift the input distribution; they are a sensitivity intervention, not a valid alternate policy.\n\n## Independent high-R sensitivity analysis\n\n'
    text+=f"The first two retained test-family hashes per rule, all siblings, select {high['roots']} roots from {high['families']} families for {high['independent_continuations']:,} fresh R256 continuations. None is unresolved and no R32 random stream is reused. Mean R32–R256 vector L1 difference is {high['mean_R32_R256_l1']:.6f}; maximum class difference is {high['max_R32_R256_class_difference']:.6f}. Per-root counts, plug-in SE and marginal Wilson intervals are retained. R256 still estimates policy probabilities; it is not ground truth.\n\n"
    text+='| Rule set | High-R matched roots/families | Relevant NLL at R256 | Relevant gain over K0 [95% interval] |\n|---|---|---:|---|\n'
    hr=high['results']['familiar_matched']
    for exposure in GROUPS:
        m=hr['metrics'][f'{exposure}|relevant|1']['macro'];g=hr['gains'][f'{exposure}|relevant|1']
        c=hr['cohort_counts'][exposure]
        text+=f"| {exposure} | {c['roots']}/{c['families']} | {m['nll']:.6f} | {g['macro_gain']:.6f} [{g['ci95'][0]:.6f}, {g['ci95'][1]:.6f}] |\n"
    text+='\nThe all-root high-R analysis and baseline/reference re-scores are retained separately. Bootstrap intervals do not propagate Monte Carlo label uncertainty; the independent re-estimate is a sensitivity check on only two families per rule.\n\n## All-rule exposure reference\n\n'
    text+='| Primary model rule group | Seed42 primary relevant NLL | All-rule reference relevant NLL |\n|---|---:|---:|\n'
    for exposure in GROUPS:
        key=f'{exposure}|relevant|1';text+=f"| {exposure} | {seed_metrics['42'][key]['macro']['nll']:.6f} | {reference[key]['macro']['nll']:.6f} |\n"
    text+='\nThe reference has gradients from all thirteen variants; its rows on the primary model\'s held-out rule set do not measure held-out transfer. One reference seed does not support an across-seed exposure-effect claim.\n\n## Numerical inference audit\n'
    numerical=[read_json(path) for path in sorted(r.glob('canonical_*.timing.json'))]
    text+=f"\nThe maximum probability change from the original padded test outputs to the canonical policy was {max(v['max_change_from_padded_probability'] for v in numerical):.6f}; mean absolute changes per model were "+', '.join(f"{v['mean_change_from_padded_probability']:.6f}" for v in numerical)+'. These are floating-point inference-policy differences, not changes to labels or weights.\n'
    text+='''
## Interpretation and limitations

Context improvement over K0 alone cannot establish inference of a rule: compare
the matched foreign-rule context, shortcut baselines and corruptions. The tables
report all controls irrespective of direction. A gain interval spanning zero is
inconclusive at this sample size; it does not prove no effect. Even a positive
relevance advantage requires larger independent families and a newly frozen
confirmatory test before a broad rule-learning claim. Named variants may have
appeared in Qwen pretraining. Simulator audits are sampled, with independent
engine comparisons for ten rule/control adapters and golden/property tests for
four in-house rules; these are not correctness proofs.

The study measures neither engine strength nor optimal-play outcomes. No larger-K
learning curve, scratch scaling law, μP validation, zero-context-only training
ablation or fully powered confirmatory study was run. Those are future experiments,
not missing rows to fill with inferred numbers. Retained failed stages and the
capture-rule correction are part of the scientific record.

Generation cost is reported per variant in `tables/generation_cost.csv`: actual
continuation lengths and worker elapsed seconds, with a clearly labeled linear
worker-hour extrapolation. Elapsed time includes sharing/scheduling and is not a
measurement of CPU core time. The original217.97-second48-worker generation phase
precedes audit, capture repair and evaluation-demo additions; it is not the total
end-to-end cost of this corrected study. GPU loop and canonical-rescore times are
retained per run rather than interpreted as dedicated-device throughput.

## Tables, figures and retained evidence

CSV and LaTeX tables are under `tables/`; SVG, PNG and PDF figures under `figures/`.
`results/final_analysis.json` includes per-seed metrics, all slices, intervention
intervals, paired baseline comparisons and the complete exclusion list.
`results/high_r_analysis.json` carries all independent-noise records and re-scores.
The final publication snapshot contains compact results and exact raw-artifact
hashes. Full datasets, continuation traces and checkpoint binaries are stored in
the local deliverable archive rather than Git. Reproduction uses the pinned model
revision and environments; numerical kernels can change exact floating-point
values across different hardware/software versions.
'''
    (ROOT/'docs/EXPLORATORY_STUDY.md').write_text(text)
    print('Final manuscript tables and study record exported',flush=True)


if __name__=='__main__':main()
