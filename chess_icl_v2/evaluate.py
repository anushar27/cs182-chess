from pathlib import Path
from collections import defaultdict
import torch
from .baselines import material
from .common import digest, read_json, write_json, provenance
from .episodes import build_episode, paired_episodes
from .metrics import aggregate, paired_gains, hierarchical_ci, calibration
from .models import Predictor
from .state import State
from .train import model_and_tokenizer, restore, file_hash


def evaluate(data, checkpoint, output, split='validation', k_values=(0,1), draws=5, seal=None, coverage_only=False):
    evaluation_provenance=provenance({'split':split,'k_values':list(k_values),'demo_draws':draws,'coverage_only':coverage_only,'matching':'exhaustive_K1_exact10pct_v2'})
    checkpoint = Path(checkpoint)
    manifest = read_json(checkpoint.parent / 'manifest.json')
    if file_hash(checkpoint) != manifest['checkpoint_sha256']:
        raise ValueError('Checkpoint hash mismatch')
    if split == 'test':
        if seal is None or seal['checkpoint_sha256'] != file_hash(checkpoint) or seal['dataset_sha256'] != digest(data):
            raise ValueError('Test evaluation requires a matching frozen seal')
        if coverage_only and 'all_root_prediction' not in seal['protocol'].get('cohorts',[]):
            raise ValueError('Coverage cohort not declared in the frozen protocol')
        if sorted(k_values)!=sorted(seal['protocol']['k_values']) or draws!=seal['protocol']['demo_draws']:
            raise ValueError('Evaluation matrix differs from the frozen protocol')
    config = manifest['config']
    torch.set_num_threads(config.get('cpu_threads', 2))
    torch.use_deterministic_algorithms(True)
    model, tokenizer = model_and_tokenizer(config)
    restore(model, checkpoint)
    predictor = Predictor(model, config['device'], tokenizer.pad_token_id)
    rows, excluded, inference_episodes = [], [], []
    for q in data['queries']:
        if q['split'] != split:
            continue
        if q['label']['target_wdl'] is None:
            excluded.append({'query_id': q['id'], 'reason': 'unresolved_rollout'})
            continue
        episodes = []
        try:
            # Enforce one common root cohort across every requested K/control/draw.
            episodes.append(build_episode(q, data['demonstrations'], tokenizer, 0, config['max_tokens'], 'zero'))
            for k in k_values:
                if not k: continue
                for draw in range(draws):
                    if coverage_only:
                        episodes.append(build_episode(q,data['demonstrations'],tokenizer,k,config['max_tokens'],'relevant',draw))
                    else:
                        _, relevant, mismatch = paired_episodes(q, data['demonstrations'], tokenizer, k, config['max_tokens'], draw)
                        episodes.extend([relevant, mismatch])
                        for condition in ('corrupt_outcomes', 'corrupt_trajectories'):
                            episodes.append(build_episode(q, data['demonstrations'], tokenizer, k,
                                                          config['max_tokens'], condition, draw))
        except ValueError as e:
            excluded.append({'query_id': q['id'], 'reason': str(e)})
            continue
        query_state=State.from_dict(q['state'])
        query_side='white' if query_state.geometry[0] else 'black'
        query_material=material(query_state)
        for ep in episodes:
            inference_episodes.append(ep)
            rows.append({**ep.metadata, 'prediction': None, 'target': ep.target,
                         'model_seed': config['seed'], 'side': query_side,
                         'material': query_material,
                         'exposure': ('standard_control' if q['variant'] == 'standard' else
                                      ('familiar_rules' if q['variant'] in manifest.get('exposed_variants', config['train_variants']) else 'heldout_rules')),
                         'checkpoint_sha256': manifest['checkpoint_sha256'],
                         'label_replicates': q['label']['replicates'], 'label_mc_se': q['label']['mc_standard_error']})
    if rows:
        probabilities=predictor.predict_many(inference_episodes,config.get('inference_batch_size',8))
        for row,p in zip(rows,probabilities):row['prediction']=p
        # Identical batch repeats; a different padding/batch shape need not be bitwise equivalent.
        first=sorted(inference_episodes,key=lambda ep:len(ep.input_ids))[:config.get('inference_batch_size',8)]
        if predictor.predict_many(first,config.get('inference_batch_size',8))!=predictor.predict_many(first,config.get('inference_batch_size',8)):
            raise ValueError('Nondeterministic frozen predictions')
    groups = defaultdict(list)
    for row in rows:
        groups[(row['exposure'], row['condition'], row['k'])].append(row)
    summary = {'|'.join(map(str, key)): {'all': aggregate(value),
               'equal_material': aggregate([r for r in value if abs(r['material']) <= 2]),
               'white': aggregate([r for r in value if r['side'] == 'white']),
               'black': aggregate([r for r in value if r['side'] == 'black']),
               'calibration': calibration(value),
               'calibration_by_side': {side:calibration([r for r in value if r['side']==side]) for side in ('white','black')}}
               for key, value in groups.items()}
    gains = {}
    for exposure in ('familiar_rules', 'heldout_rules', 'standard_control'):
        cohort = [r for r in rows if r['exposure'] == exposure]
        for k in k_values:
            if k:
                for condition in ('relevant', 'mismatched'):
                    gains[f'{exposure}|{condition}|{k}'] = hierarchical_ci(paired_gains(cohort, condition, k), replicates=300)
    report = {'evaluation_provenance':evaluation_provenance,'policy_id':data['provenance']['config']['policy_id'],
              'rules_versions':{v:sorted({q['state']['rules_version'] for q in data['queries'] if q['variant']==v}) for v in data['provenance']['config']['variants']},
              'cohort': 'all_root_prediction' if coverage_only else 'length_matched_controls', 'split': split, 'dataset_sha256': digest(data), 'checkpoint_sha256': file_hash(checkpoint),
              'rows': rows, 'exclusions': excluded, 'metrics': summary, 'context_gains': gains,
              'frozen_weights': True, 'pretraining_contamination': config['track'] == 'qwen'}
    write_json(output, report)
    return report
