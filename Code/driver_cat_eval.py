#!/usr/bin/python3

import argparse
import json
import logging
import os

import torch
from torch.utils.data import DataLoader

from model import KGEModel
from dataloader import TrainDataset, TestDataset, BidirectionalOneShotIterator
from utils import categorize_relations, filter_test_triples, read_triple

def parse_args():
    parser = argparse.ArgumentParser(description='Knowledge Graph Embedding Training and Evaluation')

    parser.add_argument('--cuda', action='store_true', help='Use GPU for training')
    parser.add_argument('--do_train', action='store_true', help='Enable training mode')
    parser.add_argument('--do_valid', action='store_true', help='Enable validation mode')
    parser.add_argument('--do_test', action='store_true', help='Enable test mode')

    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--model', type=str, default='TransE', help='Model to use (e.g., TransE, DistMult)')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save trained model')
    parser.add_argument('--init_checkpoint', type=str, default=None, help='Path to load checkpoint')

    parser.add_argument('-de', '--double_entity_embedding', action='store_true', help='Double entity embedding')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true', help='Double relation embedding')
    parser.add_argument('-n', '--negative_sample_size', type=int, default=128, help='Number of negative samples')
    parser.add_argument('-d', '--hidden_dim', type=int, default=500, help='Embedding dimension')
    parser.add_argument('-g', '--gamma', type=float, default=12.0, help='Margin for scoring')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('-b', '--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--test_batch_size', type=int, default=4, help='Batch size for testing')
    parser.add_argument('--max_steps', type=int, default=100000, help='Maximum training steps')
    parser.add_argument('--save_checkpoint_steps', type=int, default=10000, help='Checkpoint saving interval')
    parser.add_argument('--log_steps', type=int, default=100, help='Logging interval')

    return parser.parse_args()

def set_logger(log_path):
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def evaluate_by_category(model, test_triples, all_true_triples, train_triples, args):
    relation_mapping = categorize_relations(train_triples)
    results = {}

    for category, relations in relation_mapping.items():
        logging.info(f"Evaluating category: {category}")

        filtered_test_triples = filter_test_triples(test_triples, relations)

        head_dataloader = DataLoader(
            TestDataset(
                filtered_test_triples, all_true_triples, args.nentity, args.nrelation, mode='head-batch'
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, os.cpu_count() // 2),
            collate_fn=TestDataset.collate_fn
        )

        tail_dataloader = DataLoader(
            TestDataset(
                filtered_test_triples, all_true_triples, args.nentity, args.nrelation, mode='tail-batch'
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, os.cpu_count() // 2),
            collate_fn=TestDataset.collate_fn
        )

        head_metrics = model.test_step(model, head_dataloader, all_true_triples, args)
        tail_metrics = model.test_step(model, tail_dataloader, all_true_triples, args)

        results[category] = {'head': head_metrics, 'tail': tail_metrics}

    return results

def load_config(init_checkpoint, args):
    """
    Load configuration from a saved checkpoint.
    """
    config_path = os.path.join(init_checkpoint, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config.json found in {init_checkpoint}")
    
    with open(config_path, 'r') as f:
        loaded_config = json.load(f)

    # Override args with loaded configuration
    for key, value in loaded_config.items():
        setattr(args, key, value)
    return args


def main(args):

    if not any([args.do_train, args.do_valid, args.do_test]):
        raise ValueError('At least one of --do_train, --do_valid, or --do_test must be set.')

    if args.init_checkpoint is None and args.data_path is None:
        raise ValueError('Either --init_checkpoint or --data_path must be specified.')

    if args.do_train and args.save_path is None:
        raise ValueError('Please specify --save_path to save the trained model.')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.init_checkpoint:
        args = load_config(args.init_checkpoint, args)

    set_logger(os.path.join(args.save_path or args.init_checkpoint, 'log.txt'))

    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = {line.split('\t')[1]: int(line.split('\t')[0]) for line in fin}

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = {line.split('\t')[1]: int(line.split('\t')[0]) for line in fin}

    nentity, nrelation = len(entity2id), len(relation2id)
    args.nentity, args.nrelation = nentity, nrelation

    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    all_true_triples = train_triples + valid_triples + test_triples

    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding
    )

    if args.cuda:
        kge_model = kge_model.cuda()

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        results = evaluate_by_category(kge_model, test_triples, all_true_triples, train_triples, args)
        for category, metrics in results.items():
            logging.info(f"{category} Head Metrics: {metrics['head']}")
            logging.info(f"{category} Tail Metrics: {metrics['tail']}")

if __name__ == '__main__':
    args = parse_args()
    main(args)
