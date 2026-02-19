import argparse
import json
import os

import numpy as np

from relate_compile import (
    compile_schema,
    compile_solver,
    evaluate_filtered,
    evaluate_candidates,
    evaluate_hybrid,
    parse_anyburl_rules,
    read_dict,
    read_triples,
)


def parse_args():
    parser = argparse.ArgumentParser(description='RELATE-Compile pipeline (training-free)')
    parser.add_argument('--data_path', required=True, help='Path to dataset folder')
    parser.add_argument('--output_dir', required=True, help='Directory to write compiled artifacts')
    parser.add_argument('--phase_dim', type=int, default=128, help='Phase/modulus dimension')
    parser.add_argument('--symmetry_threshold', type=float, default=0.8)
    parser.add_argument('--inverse_threshold', type=float, default=0.8)
    parser.add_argument('--composition_threshold', type=float, default=0.5)
    parser.add_argument('--composition_topk', type=int, default=5)
    parser.add_argument('--composition_max_pairs', type=int, default=200000)
    parser.add_argument('--compile_only', action='store_true')
    parser.add_argument('--eval_split', choices=['valid', 'test'], default=None)
    parser.add_argument('--eval_mode', choices=['full', 'candidates', 'hybrid_rules'], default='full',
                        help='Evaluation mode: full, candidates, or hybrid_rules (AnyBURL + RELATE)')
    parser.add_argument('--anyburl_rules', type=str, default=None,
                        help='Path to AnyBURL rules file (required for hybrid_rules)')
    parser.add_argument('--candidate_max', type=int, default=15000,
                        help='Maximum candidate set size for hybrid rule evaluation')
    parser.add_argument('--alpha', type=float, default=0.9,
                        help='Fusion weight for rule score (alpha) in hybrid mode')
    parser.add_argument('--fallback_topk', type=int, default=1000,
                        help='Fallback top-k for hybrid rules (range/tails/2-hop)')
    parser.add_argument('--rule_topk', type=int, default=None,
                        help='Rule-only top-k cap before fallbacks in hybrid rules')
    return parser.parse_args()


def main():
    args = parse_args()

    thresholds = {
        'symmetry': args.symmetry_threshold,
        'inverse': args.inverse_threshold,
        'composition': args.composition_threshold,
        'comp_topk': args.composition_topk,
        'comp_max_pairs': args.composition_max_pairs,
    }

    schema_path = os.path.join(args.output_dir, 'schema.json')
    schema = compile_schema(args.data_path, thresholds, schema_path)

    solver_out = compile_solver(args.data_path, schema, args.output_dir, dim=args.phase_dim)

    if args.compile_only:
        print(f'Compiled artifacts written to {args.output_dir}')
        return

    if args.eval_split:
        entity2id = read_dict(os.path.join(args.data_path, 'entities.dict'))
        relation2id = read_dict(os.path.join(args.data_path, 'relations.dict'))
        train = read_triples(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
        valid = read_triples(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
        test = read_triples(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)

        all_true = train + valid + test
        split = valid if args.eval_split == 'valid' else test

        if args.eval_mode == 'hybrid_rules':
            if not args.anyburl_rules:
                raise ValueError('--anyburl_rules is required for hybrid_rules mode')
            rules = parse_anyburl_rules(args.anyburl_rules, entity2id, relation2id)
            metrics = evaluate_hybrid(
                split,
                all_true,
                solver_out['entity_phase'],
                solver_out['relation_phase'],
                solver_out['entity_modulus'],
                solver_out['relation_modulus'],
                nentity=len(entity2id),
                rules=rules,
                train_triples=train,
                alpha=args.alpha,
                candidate_max=args.candidate_max,
                fallback_topk=args.fallback_topk,
                rule_topk=args.rule_topk
            )
        elif args.eval_mode == 'candidates':
            metrics = evaluate_candidates(
                split,
                all_true,
                solver_out['entity_phase'],
                solver_out['relation_phase'],
                solver_out['entity_modulus'],
                solver_out['relation_modulus'],
                nentity=len(entity2id),
                schema=schema,
                train_triples=train
            )
        else:
            metrics = evaluate_filtered(
                split,
                all_true,
                solver_out['entity_phase'],
                solver_out['relation_phase'],
                solver_out['entity_modulus'],
                solver_out['relation_modulus'],
                nentity=len(entity2id)
            )
        out_path = os.path.join(args.output_dir, f'metrics_{args.eval_split}.json')
        with open(out_path, 'w', encoding='utf-8') as fout:
            json.dump(metrics, fout, indent=2)
        print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
