"""Measured cluster report; missing artifacts stay explicitly pending."""
from collections import Counter
from pathlib import Path
import numpy as np
from chess_icl_v2.common import read_json


def main():
    root=Path(__file__).resolve().parents[1];r=root/'results'
    env=read_json(r/'cluster_environment.json');profile=read_json(r/'h100_profile.json')
    data=read_json(r/'terminal_policy_rules_v2.dataset.json')
    text=f'''# H100 fixed-policy experiment results

This report supersedes the earlier hardware-access conclusion in the CPU/Stage0
snapshot. Restricted execution hid the GPU devices. Approved host execution on
`{env['host']}` confirmed **{env['gpu_count']} H10080GB GPUs and{env['logical_cpus']} logical CPUs**,
driver580.95.05, PyTorch{env['torch']} and CUDA{env['cuda']}. The earlier negative
scratch result remains a result; it was not evidence that the host lacked GPUs.
[Environment](../results/published/validated-core-v2/cluster_environment.json), [decisions](DECISIONS.md).

## Actual Qwen-7B CUDA profile

Pinned Qwen2.5-7B-Instruct backbone, last-four-layer LoRA r8/alpha16 and FP32 W/D/L
head. All lengths below completed a real optimizer update. Exact repeated
predictions and checkpoint save/reload passed. These repeated-token probes measure
memory/mechanics, not chess performance. Read cold-start and sharing effects into
timing rather than inferring a smooth throughput curve from five updates.

| Tokens | Update seconds | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|---:|
'''
    for row in profile['rows']:
        if row['status']=='passed':text+=f"| {row['tokens']} | {row['seconds']:.3f} | {row['peak_allocated_bytes']/2**30:.3f} | {row['peak_reserved_bytes']/2**30:.3f} |\n"
    text+='''
[Raw profile](../results/published/validated-core-v2/h100_profile.json); full warnings/log in the local artifact ZIP.
The SDPA sliding-window warning was retained and inspected: use_sliding_window
is false; configured sliding_window131072 exceeds native context32768. No context
extension or custom attention mask was introduced.

## Larger development dataset under terminal_aware_v1

'''
    text+=f'''Seed2026112402; R32; administrative budget2560;8 demonstration and8 query-source
families per fold per rule; three disjoint folds. **{len(data['queries'])} roots and
{sum(q['label']['replicates'] for q in data['queries']):,} label continuations** retained.
All continuations and demonstration games terminated; no censoring was converted
to draws. The original generation phase measured **{data['generation_seconds']:.2f}s** with48
persistent workers, before the final audit. The capture-rule repair and evaluation-demo
extension are additional retained phases, not included in that number. Near-duplicate and replay audits passed. There are
{len(data['exclusions'])} outcome-independent source/root exclusions, retained with reasons.
The exact compact Hamming index is regression-tested against exhaustive distance;
the parallel generator is tested against the reference seed/trajectory semantics.

Unlike the first uniform-policy sample, development roots show substantial decisive
outcomes. Root distributions also changed because source games use the declared
policy; do not interpret the table as a randomized paired policy effect. The earlier
same-root small policy probe is the paired diagnostic.

| Rule | Train+validation roots | W continuations | D | L | Unresolved |
|---|---:|---:|---:|---:|---:|
'''
    for variant in data['provenance']['config']['variants']:
        qs=[q for q in data['queries'] if q['variant']==variant and q['split']!='test']
        counts=[sum(q['label']['counts_wdl'][i] for q in qs) for i in range(3)]
        text+=f"| {variant} | {len(qs)} | {counts[0]} | {counts[1]} | {counts[2]} | {sum(q['label']['unresolved'] for q in qs)} |\n"
    text+='''
[Corrected-data/evaluation manifest](../results/published/final-test-v2/raw_artifact_manifest.json),
[artifact status](ARTIFACT_STATUS.md),
[config](../configs/v2_terminal_pilot.json); full generation log in the artifact ZIP.
Labels are root-player W/D/L under this policy. They are not engine strength,
material adjudications or optimal-play outcomes. Test labels were not used to
train or select checkpoints; the high-R subset selection uses family hashes only.

## Training and checkpoint lineage

Exploratory matrix fixed before opening its test-model scores: seeds42/123/456,
30-update same-seed standard warmup,200-update familiar-rule training, K mixture0/1,
accumulation4, LR1e-4,20-update LR warmup and validation every40 updates. All base
weights stay frozen; **731,651** LoRA/head parameters train. All selected checkpoints
use macro validation NLL. This is a measured small-scale experiment, not the
previously illustrative2,000-update full matrix. No positive-result stopping rule.

| Seed | Condition | Updates | Best validation NLL | Training seconds | Peak allocated GiB |
|---|---|---:|---:|---:|---:|
'''
    for seed in (42,123,456):
        for phase,folder in [('standard warmup',f'qwen_warmup_seed{seed}'),('four familiar rules',f'qwen_familiar_seed{seed}')]:
            path=r/folder/'manifest.json'
            if path.exists():
                m=read_json(path);text+=f"| {seed} | {phase} | {m['config']['steps']} | {m['best_macro_validation_nll']:.6f} | {m['seconds']:.2f} | {m['peak_cuda_allocated_bytes']/2**30:.3f} |\n"
            else:text+=f'| {seed} | {phase} | pending | pending | pending | pending |\n'
    path=r/'qwen_direct_exposure_seed42/manifest.json'
    if path.exists():
        m=read_json(path);text+=f"| 42 | all-rule exposure reference | {m['config']['steps']} | {m['best_macro_validation_nll']:.6f} | {m['seconds']:.2f} | {m['peak_cuda_allocated_bytes']/2**30:.3f} |\n"
    else:text+='| 42 | all-rule exposure reference | pending | pending | pending | pending |\n'
    text+='''
Training time is the instrumented optimization/validation loop, excluding model
loading and episode construction. GPU sharing and other users' workloads can
change timing. No unrelated processes were stopped. The actual GPU selection and
checkpoint hashes are in [the matrix status](../results/published/final-test-v2/cluster_matrix_status.json).
Parent manifests record the union of all gradient-exposed rules, so an exposed
parent cannot accidentally count as held-out transfer.

## Frozen evaluation

The four training rules are diagonal_rook, straight_bishop, weakened_queen and
extended_pawn; the remaining nine project rules are held out from gradients.
Standard is reported separately. The all-rule condition is an exposure reference,
not a theoretical upper bound or a held-out transfer result. Qwen pretraining may
already contain named variants; no contamination-free claim is made.

Test access binds dataset/checkpoint/protocol hashes and explicitly marks these
experiments exploratory. Data/simulator/R/hardware gates all pass; a positive-pilot
gain is not fabricated. K0, relevantK1, mismatchedK1 and both corruption controls
use common roots, five demo draws and exact length matching; all exclusions are
reported. All weights remain frozen. Larger-K experiments are separate from the
registered primaryK1 analysis.
'''
    for split in ('validation','test','test_coverage'):
        summary=r/f'qwen_{split}_aggregate.json'
        if summary.exists():
            a=read_json(summary);text+=f'\n### {split.capitalize()} results across three seeds\n\n| Exposure | Condition | K | Macro NLL | Vector Brier |\n|---|---|---:|---:|---:|\n'
            for key,metric in a['metrics'].items():
                exposure,condition,k=key.split('|');m=metric['macro'];text+=f"| {exposure} | {condition} | {k} | {m['nll']:.6f} | {m['brier']:.6f} |\n"
            text+='\n| Exposure | Context | Paired NLL gain | Hierarchical95% interval |\n|---|---|---:|---|\n'
            for key,v in a['gains'].items():
                if 'macro_gain' in v:
                    exposure,condition,k=key.split('|');text+=f"| {exposure} | {condition}K{k} | {v['macro_gain']:.6f} | [{v['ci95'][0]:.6f}, {v['ci95'][1]:.6f}] |\n"
            text+=f'\n[Full aggregate](../results/published/final-test-v2/qwen_{split}_aggregate.json). The hierarchy covers model seeds, demo draws and source families; Monte Carlo target noise is separate.\n'
        else:text+=f'\n**{split.capitalize()} aggregate: pending completion of all frozen evaluations.**\n'
    timings=list(r.glob('canonical_*.timing.json'))
    if timings:
        text+='\n## Numerical inference audit\n\nOriginal batch8 test outputs are retained as diagnostics. Final test inputs were reconstructed and hash-checked, then evaluated individually without padding; validation checkpoint selection is unchanged. No weights/targets/cohorts were changed.\n\n| Run | Unique inputs | Rescore wall seconds | Max probability change from padded | Mean absolute change |\n|---|---:|---:|---:|---:|\n'
        for path in sorted(timings):
            t=read_json(path);text+=f"| {path.name.removeprefix('canonical_').removesuffix('.timing.json')} | {t['unique_inputs']} | {t['total_seconds']:.2f} | {t['max_change_from_padded_probability']:.6f} | {t['mean_change_from_padded_probability']:.6f} |\n"
        text+='\nEvery run verified unchanged parameter versions and an uncached repeated output. The final eleven-report audit passes; overlapping primary/coverage inputs now have exactly identical probabilities. See the [complete study](EXPLORATORY_STUDY.md) for negative results, baselines, exclusions and independent R256 sensitivity.\n'
    text+='''
## Scope and limitations

This is the full retained record of the implemented small-scale pipeline and
measured cluster experiment. It does not establish optimal play, a scaling law,
a full-powered confirmatory result or a validated μP study. Simulator checks are
sampled; four in-house rules lack an independent-engine equivalence proof. Query
histories may reveal rules already atK0, and setup/barrier geometry is observable.
The independent high-R analysis covers56 roots/28 families (53 roots/27 families in the matched-control cohort), with its actual counts and uncertainty reported separately.
The manuscript should distinguish this exploratory study from a newly preregistered,
larger confirmatory experiment. Every positive/negative condition belongs in the
results, including failed feasibility stages and discarded controls.
'''
    (root/'docs/CLUSTER_RESULTS.md').write_text(text)


if __name__=='__main__':main()
