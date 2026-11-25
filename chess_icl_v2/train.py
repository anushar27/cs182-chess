from __future__ import annotations

import hashlib
from pathlib import Path
import random
import time
import numpy as np
import torch
from .common import digest, provenance, read_json, write_json
from .episodes import ByteTokenizer, build_episode
from .tokenizer import SquareMoveTokenizer
from .metrics import aggregate
from .models import TinyTransformer, QwenProbabilityModel, collate, soft_ce


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def model_and_tokenizer(config):
    if config['track'] == 'tiny':
        return TinyTransformer(**config['architecture']), (ByteTokenizer() if config.get('tokenizer') == 'byte' else SquareMoveTokenizer())
    if config['track'] != 'qwen':
        raise ValueError('Unknown model track')
    from transformers import AutoTokenizer
    q = config['architecture']
    model = QwenProbabilityModel(**q)
    tokenizer = AutoTokenizer.from_pretrained(q['model_id'], revision=q['revision'], trust_remote_code=False)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if config['max_tokens'] > model.context_limit:
        raise ValueError('Requested context exceeds pinned model configuration')
    return model, tokenizer


def weights(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def restore(model, path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    if set(saved) != expected:
        raise ValueError('Checkpoint trainable parameter set mismatch')
    missing, unexpected = model.load_state_dict(saved, strict=False)
    if unexpected or expected.intersection(missing):
        raise ValueError('Checkpoint load failed')


def episodes_for(data, split, config):
    episodes, exclusions = [], []
    for q in data['queries']:
        if q['split'] != split or q['variant'] not in config['train_variants']:
            continue
        if q['label']['target_wdl'] is None:
            exclusions.append({'query_id': q['id'], 'reason': 'unresolved_rollout'})
            continue
        for k in config['k_values']:
            try:
                episodes.append(build_episode(q, data['demonstrations'], config['_tokenizer'], k,
                                               config['max_tokens'], 'relevant' if k else 'zero'))
            except ValueError as e:
                exclusions.append({'query_id': q['id'], 'k': k, 'reason': str(e)})
    return episodes, exclusions


@torch.inference_mode()
def validate(model, episodes, device, pad, batch_size=1):
    model.eval()
    rows = []
    ordered=sorted(episodes,key=lambda ep:len(ep.input_ids))
    for start in range(0,len(ordered),batch_size):
        group=ordered[start:start+batch_size]
        probabilities=model(**collate(group,pad,device)).float().softmax(-1).cpu().tolist()
        for ep,pred in zip(group,probabilities):
            rows.append({**ep.metadata,'prediction':pred,'target':ep.target})
    return aggregate(rows)


def train(data, config, output, initial_checkpoint=None):
    run_provenance = provenance(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config.get('cpu_threads', 2))
    torch.use_deterministic_algorithms(True)
    device = config['device']
    if config['track'] == 'qwen' and not device.startswith('cuda'):
        raise ValueError('The 7B main track requires CUDA; use tiny for a CPU smoke run')
    model, tokenizer = model_and_tokenizer(config)
    lineage = None
    exposed = set(config["train_variants"])
    if initial_checkpoint:
        initial_checkpoint = Path(initial_checkpoint)
        parent = read_json(initial_checkpoint.parent / 'manifest.json')
        if parent['config']['architecture'] != config['architecture'] or parent['config']['track'] != config['track']:
            raise ValueError('Warmup architecture mismatch')
        if file_hash(initial_checkpoint) != parent['checkpoint_sha256']:
            raise ValueError('Warmup checkpoint hash mismatch')
        if parent['config']['seed'] != config['seed']:
            raise ValueError('Use the declared common warmup for this model seed')
        exposed.update(parent.get('exposed_variants', parent['config']['train_variants']))
        restore(model, initial_checkpoint)
        lineage = {'path': str(initial_checkpoint), 'sha256': file_hash(initial_checkpoint),
                   'parent_manifest_sha256': digest(parent)}
    model.to(device)
    epconfig = {**config, '_tokenizer': tokenizer}
    training, train_exclusions = episodes_for(data, 'train', epconfig)
    validation, validation_exclusions = episodes_for(data, 'validation', epconfig)
    if not training or not validation:
        raise ValueError('Need complete-label training AND validation episodes')
    write_json(output / 'exclusions.json', {'train': train_exclusions, 'validation': validation_exclusions})
    if config.get('strict_episode_coverage', True) and (train_exclusions or validation_exclusions):
        raise ValueError('Incomplete episode coverage; inspect exclusions instead of silently selecting easy roots')
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config['learning_rate'], betas=(.9, .95), weight_decay=.01)
    best, log, start = float('inf'), [], time.perf_counter()
    rng = random.Random(seed)
    accumulation = config.get('gradient_accumulation', 1)
    if config['steps'] < 1 or accumulation < 1:
        raise ValueError('Invalid step or accumulation count')
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    for step in range(1, config['steps'] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(accumulation):
            ep = training[rng.randrange(len(training))]
            batch = collate([ep], tokenizer.pad_token_id, device)
            logits = model(**batch)
            target = torch.tensor([ep.target], dtype=torch.float32, device=device)
            loss = soft_ce(logits, target)
            (loss / accumulation).backward()
            losses.append(loss.item())
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        # Linear warmup, cosine decay; logged LR is the value used for this update.
        warmup = config.get('warmup_steps', 0)
        factor = min(1., step / max(1, warmup)) if step <= warmup else (
            .5 * (1 + np.cos(np.pi * (step - warmup - 1) / max(1, config['steps'] - warmup))))
        for group in optimizer.param_groups:
            group['lr'] = config['learning_rate'] * factor
        optimizer.step()
        record = {'step': step, 'loss': float(np.mean(losses)), 'learning_rate': optimizer.param_groups[0]['lr']}
        if step % config['eval_interval'] == 0 or step == config['steps']:
            report = validate(model, validation, device, tokenizer.pad_token_id, config.get('validation_batch_size',1))
            record['validation'] = report
            if report['macro']['nll'] < best:
                best = report['macro']['nll']
                torch.save(weights(model), output / 'best.pt')
                write_json(output / 'best.json', {'step': step, 'validation': report})
        log.append(record)
        write_json(output / 'training_log.json', log)
    manifest = {'provenance': run_provenance, 'config': config, 'lineage': lineage,
                'exposed_variants': sorted(exposed),
                'dataset_sha256': digest(data), 'checkpoint_sha256': file_hash(output / 'best.pt'),
                'training_episodes': len(training), 'validation_episodes': len(validation),
                'seconds': time.perf_counter() - start, 'best_macro_validation_nll': best,
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if device.startswith('cuda') else None}
    write_json(output / 'manifest.json', manifest)
    return manifest
