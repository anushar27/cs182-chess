import argparse
from pathlib import Path
from .common import digest, read_json, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description='Auditable fixed-policy WDL research pipeline')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('audit'); p.add_argument('--output', required=True); p.add_argument('--games',type=int,default=3);p.add_argument('--plies',type=int,default=80)
    p = sub.add_parser('generate');p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p = sub.add_parser('extend');p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--max-plies',type=int,required=True);p.add_argument('--workers',type=int,default=1)
    p = sub.add_parser('train');p.add_argument('--data',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--initial-checkpoint');p.add_argument('--gates')
    p = sub.add_parser('evaluate');p.add_argument('--coverage-only',action='store_true');p.add_argument('--data',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--split',choices=['validation','test'],default='validation');p.add_argument('--k',type=int,nargs='+',default=[0,1]);p.add_argument('--draws',type=int,default=5);p.add_argument('--seal')
    p = sub.add_parser('gates');p.add_argument('--data',required=True);p.add_argument('--audits',required=True);p.add_argument('--pilot');p.add_argument('--profile');p.add_argument('--output',required=True)
    p = sub.add_parser('seal');p.add_argument('--exploratory',action='store_true');p.add_argument('--data',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--protocol',required=True);p.add_argument('--gates',required=True);p.add_argument('--output',required=True)
    p = sub.add_parser('tokens');p.add_argument('--data',required=True);p.add_argument('--tokenizer',required=True);p.add_argument('--revision',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    if args.command == 'audit':
        from .audit import run_audits
        report=run_audits(games=args.games,plies=args.plies);write_json(args.output,report)
        print(f"audit passed={report['passed']}")
        return 0 if report['passed'] else 1
    if args.command == 'generate':
        from .data import generate,summarize
        data=generate(read_json(args.config));write_json(args.output,data);write_json(str(args.output)+'.summary.json',summarize(data))
        print(f"Generated {len(data['queries'])} roots");return 0
    data=read_json(args.data)
    if args.command == 'extend':
        from .diagnostics import extend
        from .data import summarize
        result=extend(data,args.max_plies,args.workers);write_json(args.output,result);write_json(str(args.output)+'.summary.json',summarize(result));return 0
    if args.command == 'train':
        from .train import train
        config=read_json(args.config)
        if config.get('mode') == 'main':
            gate=read_json(args.gates) if args.gates else {}
            if not gate.get('passed') or gate.get('dataset_sha256') != digest(data):
                raise ValueError('Main training blocked by feasibility/audit gates')
            if not args.initial_checkpoint:
                raise ValueError('Main conditions must clone a common warmup checkpoint')
        train(data,config,args.output,args.initial_checkpoint);return 0
    if args.command == 'evaluate':
        from .evaluate import evaluate
        result=evaluate(data,args.checkpoint,args.output,args.split,args.k,args.draws,read_json(args.seal) if args.seal else None,args.coverage_only)
        print(f"Predictions={len(result['rows'])}; excluded roots={len(result['exclusions'])}");return 0
    if args.command == 'gates':
        from .protocol import gates
        result=gates(data,read_json(args.audits),read_json(args.pilot) if args.pilot else None,read_json(args.profile) if args.profile else None)
        write_json(args.output,result);print(result);return 0 if result['passed'] else 1
    if args.command == 'seal':
        from .train import file_hash
        gate=read_json(args.gates);protocol=read_json(args.protocol)
        from .protocol import can_seal
        if not can_seal(gate,args.exploratory) or gate['dataset_sha256'] != digest(data):
            raise ValueError('Cannot freeze test access before the declared data/hardware gates pass')
        manifest=read_json(Path(args.checkpoint).parent/'manifest.json')
        if set(manifest.get('exposed_variants', manifest['config']['train_variants'])) & set(protocol['heldout_variants']):
            raise ValueError('Held-out rule exposure in checkpoint')
        if file_hash(args.checkpoint) != manifest['checkpoint_sha256']:
            raise ValueError('Checkpoint mismatch')
        if Path(args.output).exists(): raise ValueError('Refusing to overwrite a test seal')
        write_json(args.output,{'dataset_sha256':digest(data),'checkpoint_sha256':file_hash(args.checkpoint),
                              'protocol':protocol,'protocol_sha256':digest(protocol),'gates_sha256':digest(gate),
                              'exploratory':args.exploratory,'confirmatory_gates_passed':gate['passed']})
        return 0
    if args.command == 'tokens':
        import re
        if not re.fullmatch('[0-9a-f]{40}',args.revision): raise ValueError('Pin tokenizer revision')
        from transformers import AutoTokenizer,AutoConfig
        from .diagnostics import token_lengths
        tok=AutoTokenizer.from_pretrained(args.tokenizer,revision=args.revision,trust_remote_code=False)
        config=AutoConfig.from_pretrained(args.tokenizer,revision=args.revision,trust_remote_code=False)
        write_json(args.output,token_lengths(data,tok,{'revision':args.revision,'max_position_embeddings':config.max_position_embeddings}))
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
