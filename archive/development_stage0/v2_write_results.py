"""Generate manuscript-facing tables directly from saved measured artifacts."""
import csv
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from chess_icl_v2.common import read_json,write_json


def main():
    root=Path(__file__).resolve().parents[1];r=root/'results'
    initial=read_json(r/'stage0_all_rules.dataset.json.summary.json')
    extended=read_json(r/'stage0_extended.dataset.json.summary.json')
    data=read_json(r/'stage0_extended.dataset.json')
    audit=read_json(r/'simulator_audit_final.json')
    ev=read_json(r/'tiny_validation.json');base=read_json(r/'tiny_baselines.json')
    model=read_json(r/'tiny_pilot/manifest.json');gate=read_json(r/'main_gates.json')
    tokens=read_json(r/'qwen_extended_token_lengths.json')
    tests=ET.parse(r/'tests.xml').getroot()
    suites=list(tests) if tests.tag=='testsuites' else [tests]
    ntests=sum(int(s.get('tests',0)) for s in suites)
    failures=sum(int(s.get('failures',0))+int(s.get('errors',0)) for s in suites)
    a={x['variant']:x for x in initial['rows']}
    rows=[]
    for x in extended['rows']:
        before=a[x['variant']];c=x['root_outcome_counts']
        rows.append({'variant':x['variant'],'roots':x['queries'],'R':4,
            'unresolved_320':before['root_outcome_counts'].get('unresolved',0),
            'W_1280':c.get('W',0),'D_1280':c.get('D',0),'L_1280':c.get('L',0),
            'unresolved_1280':c.get('unresolved',0),'terminal_demos':x['terminal_demo_games'],
            'seconds_per_ply_320':before['seconds_per_ply'],
            'projected_core_hours_per_million_320':before['projected_core_hours_per_million_continuations']})
    with (r/'stage0_table.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (r/'stage0_table.tex').open('w') as f:
        f.write('\\begin{tabular}{lrrrrr}\nRule & Unresolved (320) & W & D & L & Unresolved (1280) \\\\\n\\hline\n')
        for x in rows:
            f.write(x['variant'].replace('_',r'\_')+f" & {x['unresolved_320']} & {x['W_1280']} & {x['D_1280']} & {x['L_1280']} & {x['unresolved_1280']} \\\\\n")
        f.write('\\end{tabular}\n')
    total_unresolved=sum(x['unresolved_320'] for x in rows)
    query_by_id={q['id']:q for q in data['queries']}
    excluded=Counter(query_by_id[e['query_id']]['variant'] for e in ev['exclusions'])
    npositions=sum(v['reachable_positions'] for v in audit['variants'].values())
    ncomparisons=sum(v['independent_move_comparisons'] for v in audit['variants'].values())
    nterminal=sum(v['independent_terminal_comparisons'] for v in audit['variants'].values())
    text=f'''# Measured results and remaining work

These are **exploratory local results**, not completed 7B/H100 or confirmatory
paper results. Client calendar date: November 24, 2026. Raw machine timestamps
are UTC. All tables below are generated from retained artifacts by
`scripts/v2_write_results.py`.

## Verification

- V2 regression/integration suite: **{ntests} tests, {failures} failures**. Includes rule attacks,
special moves, state/repetition round trips, deterministic independent workers,
query retention/leakage, paired controls, proper scores, checkpoint reload/frozen
inference, tiny local Qwen LoRA injection/backprop and warmup lineage.
- Simulator audit: **{npositions} reachable positions**, **{ncomparisons} independent
Fairy-Stockfish legal-move comparisons**, and **{nterminal} encountered independent
no-move terminal result comparisons**; all passed. This is limited sampled coverage,
not proof or comprehensive terminal equivalence. Four rules use golden/property
coverage only. [Audit artifact](../results/simulator_audit_final.json).
- Qwen tokenizer/config downloaded at immutable revision
`a09a35458c702b33eeacc393d103063234e8bc28`; native configured context: **32,768**.
No 7B weights were downloaded or trained locally. The local tiny Qwen architecture
test is a mechanics test, not a 7B experiment.
- Python wheel build and cluster shell syntax checked. Full environment and test
logs: [environment](../results/environment.txt), [tests](../results/pytest.log).

## Stage0, all 13 project rules plus standard

Seed20261124; uniform_legal_v1; one query-source family per fold per rule, two
adjacent opposite-side roots/family; R=4. Original pools:84 demonstration games,
42 query-source games,84 roots and336 independent label continuations.
Initial administrative budget:320 plies. Generation wall time:
**{initial['generation_seconds']:.2f} seconds** on this host; process/root replay,
source generation and deduplication are included in that wall time.

**{total_unresolved}/336 continuations were unresolved at320 plies.** Extending the same
streams to1280 resolved all336; none were reclassified by material or silently
counted as draws. All84 query states and replicate seeds were retained. One whole
demonstration was excluded because its newly visible suffix was near a query;
its raw trace is retained.83 terminal demonstrations remain. Main-pool integrity
and both-side coverage passed. First extension failure/recomputation cost was not
separately instrumented; do not call225.82 seconds the total cost of all development.

Each row contains24 continuations (6 roots ×4); W/D/L are root-player outcomes.
The original320-ply rollout core-hour projections exclude startup, root replay,
deduplication and extension costs; they are not measured192-core throughput.

| Rule | Unresolved at320 | W at1280 | D at1280 | L at1280 | Unresolved at1280 |
|---|---:|---:|---:|---:|---:|
'''
    for x in rows:text+=f"| {x['variant']} | {x['unresolved_320']} | {x['W_1280']} | {x['D_1280']} | {x['L_1280']} | {x['unresolved_1280']} |\n"
    text+='''
Raw data and tables: [320-ply dataset](../results/stage0_all_rules.dataset.json),
[extended dataset](../results/stage0_extended.dataset.json),
[CSV](../results/stage0_table.csv), [LaTeX](../results/stage0_table.tex).
R=4 is far too small to establish fine probability differences; label uncertainty
is saved [separately](../results/label_uncertainty.json). Stage0 structural test
folds were inspected as development diagnostics and are not sealed paper tests.

## Terminal-aware development probe

Same two development/train roots per rule, two fresh continuations per root,
320-ply budget (56 continuations total). This is a separate policy experiment;
its labels are never merged into the uniform-policy dataset. Some movement/capture
rules show more decisive outcomes, while other rules remain draw-dominated or
censored. R=2 does not justify selecting a final policy or claiming improvement.
Every trace and cost is in [the probe](../results/terminal_aware_probe.json).

| Rule | W | D | L | Unresolved | Measured rollout seconds |
|---|---:|---:|---:|---:|---:|
'''
    for v in read_json(r/'terminal_aware_probe.json')['summary']:
        c=v['counts'];text+=f"| {v['variant']} | {c.get('W',0)} | {c.get('D',0)} | {c.get('L',0)} | {c.get('unresolved',0)} | {v['seconds']:.4f} |\n"
    text+='''
## Actual Qwen token measurements

Move and complete-episode counts use the downloaded Qwen tokenizer. Full games
are retained. SAN comparison is restricted to standard/antichess. The choice of
UCI for custom rules is a reproducibility choice, not evidence that it is globally
shorter than SAN. The tiny scratch vocabulary is measured separately from Qwen.

| K | Available full episodes | Median Qwen tokens | Maximum Qwen tokens |
|---|---:|---:|---:|
'''
    for k in (0,1,2,4,8,16):
        lengths=[e['tokens'] for e in tokens['episodes'] if e['k']==k and 'tokens' in e]
        text+=f"| {k} | {len(lengths)} | {np.median(lengths) if lengths else 'unavailable'} | {max(lengths) if lengths else 'unavailable'} |\n"
    text+='''
K≥4 is unavailable because Stage0 has only two demo games/fold/rule, not because
this establishes a tokenizer or GPU context limit. Counts at larger K require a
larger demo pool. [Complete measurements](../results/qwen_extended_token_lengths.json).
No H100 memory/throughput measurement exists.

## Scratch pipeline pilot and frozen evaluation

'''
    text+=f'''Single seed42, {model['trainable_parameters']:,} parameters, one32-wide causal RoPE
block,337-token square/move vocabulary,30 optimizer updates,16 training episodes
and16 validation episodes over the four familiar rules, K=0/1. Training measured
{model['seconds']:.3f} seconds after episode construction on CPU; this excludes model/data
setup, evaluation and baseline time. No warmup was used for this debugging run;
main Qwen runs require a common standard warmup. Checkpoint selection used macro
validation NLL on all16 validation episodes: **{model['best_macro_validation_nll']:.6f}**.
[Manifest](../results/tiny_pilot/manifest.json), [training log](../results/tiny_pilot/training_log.json).

Frozen evaluation: **{len(ev['rows'])} predictions**,5 demo draws/K>0, K=0 once/root,
and **{len(ev['exclusions'])} excluded roots** because a full relevant/mismatched context
pair could not meet10% demo-length tolerance. Exclusions were applied to every
condition for that root. Excluded rule counts: `{dict(excluded)}`.
These exclusions affect representativeness; do not extrapolate to all held-outs.

| Exposure | Condition | K | Macro NLL | Vector Brier |
|---|---|---:|---:|---:|
'''
    for key,value in ev['metrics'].items():
        exposure,condition,k=key.split('|');m=value['all']['macro']
        text+=f"| {exposure} | {condition} | {k} | {m['nll']:.6f} | {m['brier']:.6f} |\n"
    text+='''
The main familiar-rule comparison is negative: relevant context does not improve
NLL over K=0, and mismatched context is about as effective. The small positive
held-out gain also occurs with mismatched context; it does not establish transfer.
There is no demonstrated useful rule ICL in this pilot. Single-seed bootstrap
intervals below omit seed uncertainty and all Monte Carlo target noise; each rule
has only one validation source family, so they are especially inadequate for
scientific significance claims.

| Exposure | Context | Paired NLL gain | Conditional bootstrap95% interval |
|---|---|---:|---|
'''
    for key,v in ev['context_gains'].items():
        if 'macro_gain' in v:
            exposure,condition,_=key.split('|');text+=f"| {exposure} | {condition} | {v['macro_gain']:.6f} | [{v['ci95'][0]:.6f}, {v['ci95'][1]:.6f}] |\n"
    text+='''
Calibration bins/by-side scores, equal-material slices, class counts, corruptions,
all per-root probabilities and exact input/demo hashes are in
[the evaluation artifact](../results/tiny_validation.json).

## Baselines on the same familiar-rule relevant-context cohort

| Baseline | Macro NLL | Vector Brier |
|---|---:|---:|
'''
    for key,m in base['metrics'].items():
        name,exposure,cond,k=key.split('|')
        if exposure=='familiar_rules' and cond=='relevant' and k=='1':
            text+=f"| {name} | {m['macro']['nll']:.6f} | {m['macro']['brier']:.6f} |\n"
    text+='''
Every baseline prediction and all other exposure/K groups are retained in
[the baseline artifact](../results/tiny_baselines.json). Learned baselines use
training folds only. These are small linear/retrieval controls, not optimized
state-of-the-art chess predictors. Beating kNN alone is insufficient: relevant
context must also beat K=0 and mismatched context.

## Main gates and unexecuted work

'''
    for name,passed in gate['checks'].items():text+=f"- `{name}`: **{'pass' if passed else 'FAIL / unavailable'}**.\n"
    text+='''
The main run is blocked. Current R=4 is below the required32; the pilot lacks useful
relevant-context gain; no H100 profile or usable remote desktop/terminal exists
in this chat. The CPU host's nvidia-smi cannot communicate with a driver. GPU
scripts and pinned Qwen training code are provided, but **7B training, H100 timing,
8-GPU packing, three-model-seed results, high-R confirmatory labels and sealed-test
results have not been produced**. No old repository accuracy plot is reused.

Next work is development of a useful policy/root distribution, a larger independent
dataset and multi-seed pilot, followed by actual H100 profiling and gated training.
Do not launch the expensive matrix solely because the implementation runs.

Known limitations: sampled simulator coverage; four in-house rule implementations;
rule conventions that differ from a team's unstated interpretation; conservative
near-duplicate filtering; query histories can reveal rules even at K=0; missing
long-K demo pools; low sample sizes; untested7B/CUDA path; no exact mid-run optimizer
resume or validated μP scaling track. [Decisions and incidents](DECISIONS.md).
'''
    (root/'docs/RESULTS.md').write_text(text)
    write_json(r/'results_index.json',{'tests':ntests,'test_failures':failures,'reachable_positions':npositions,
              'independent_move_comparisons':ncomparisons,'initial_unresolved':total_unresolved,
              'final_unresolved':0,'roots':84,'continuations':336,'main_gates_passed':gate['passed']})


if __name__=='__main__':main()
