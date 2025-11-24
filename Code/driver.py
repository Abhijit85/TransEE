
#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random
from collections import defaultdict

import numpy as np
import torch
from dotenv import load_dotenv

from torch.utils.data import DataLoader

from model import KGEModel

from dataloader import TrainDataset, PathDataset
from dataloader import BidirectionalOneShotIterator

from torch.optim.lr_scheduler import CosineAnnealingLR

load_dotenv()

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    
    parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions', type=int, nargs='+', default=None, 
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')
    
    parser.add_argument('--data_path', type=str, default=None)
    default_model = 'TransE'
    parser.add_argument('--model', default=default_model, type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
    
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.00005, type=float) # change it to 5e-5 for FB15k237,YAGO310 and 1e-5 for WNR18RR
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')
    
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('--lr_t_max', default=None, type=int,
                        help='Optional cosine scheduler period; defaults to max_steps')
    parser.add_argument('--lr_eta_min', default=1e-5, type=float,
                        help='Minimum learning rate for cosine scheduler decay')
    parser.add_argument('--lr_drop_steps', type=int, nargs='+', default=None,
                        help='Optional manual LR drop steps (training iterations)')
    parser.add_argument('--lr_drop_gamma', type=float, default=0.5,
                        help='Multiplicative factor applied at each manual drop step')
    parser.add_argument('--secondary_warmup_step', type=int, default=None,
                        help='Optional step to apply a secondary warm-up LR bump')
    parser.add_argument('--secondary_warmup_gamma', type=float, default=1.0,
                        help='Multiplicative factor for the secondary warm-up bump')
    parser.add_argument('--stop_at_first_peak', action='store_true',
                        help='Stop training once validation MRR falls below its best value')
    parser.add_argument('--early_stop_patience', type=int, default=None,
                        help='Number of consecutive non-improving validations before stopping (default: 5)')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0,
                        help='Minimum MRR improvement required to reset patience')
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of micro-steps to accumulate before an optimizer update')
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=5000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')


    parser.add_argument('-eras','--use_eras', action='store_true', help='Enable ERAS for RelatE')
    parser.add_argument('--k_prototypes', default=4, type=int, help='Number of ERAS prototypes')

    parser.add_argument('--init_modulus_weight', type=float, default=2.5,help='Initial value for RelatE modulus weight (default: 3.5)')



    # Type constraints
    parser.add_argument('--type_map_path', type=str, default=None, help='Path to entity-type map JSON file')
    parser.add_argument('--type_lambda', type=float, default=1.0,help='Scaling factor for type bias injection (default 1.0)')
    parser.add_argument('--init_rel_width', type=float, default=0.1,help='Initial value for relation-specific slope (default: 0.1)')

    # Multi-hop / phase extensions
    parser.add_argument('--path_loss_weight', type=float, default=0.0, help='Weight of multi-hop path ranking loss')
    parser.add_argument('--path_negative_size', type=int, default=8, help='Number of negative tails per path sample')
    parser.add_argument('--path_batch_size', type=int, default=64)
    parser.add_argument('--path_hops', type=int, nargs='+', default=[2, 3], help='Hop lengths to enumerate for path training')
    parser.add_argument('--path_max_per_hop', type=int, default=5000, help='Maximum number of sampled paths per hop length')
    parser.add_argument('--path_consistency_weight', type=float, default=0.0, help='Optional consistency loss weight between composed relation and explicit path')
    parser.add_argument('--path_curriculum_steps', type=int, nargs=2, default=None, help='Start/stop steps for enabling path loss')
    parser.add_argument('--path_margin', type=float, default=1.0, help='Margin for path ranking loss')
    parser.add_argument('--path_consistency_margin', type=float, default=1.0, help='Margin for path consistency regularizer')
    parser.add_argument('--phase_harmonics', type=int, default=2, help='Number of phase harmonics for multi-frequency scoring')
    parser.add_argument('--inverse_map_path', type=str, default=None, help='Optional JSON mapping of relation -> inverse relation for phase tying')
    
    parsed_args = parser.parse_args(args)

    env_data_path = os.getenv('DATA_PATH')
    if parsed_args.data_path is None and env_data_path:
        parsed_args.data_path = env_data_path

    env_model = os.getenv('MODEL_NAME')
    if env_model and parsed_args.model == default_model:
        parsed_args.model = env_model

    env_patience = os.getenv('EARLY_STOP_PATIENCE')
    if parsed_args.early_stop_patience is None:
        parsed_args.early_stop_patience = int(env_patience) if env_patience else 5

    env_accum = os.getenv('GRADIENT_ACCUMULATION_STEPS')
    if env_accum and parsed_args.gradient_accumulation_steps == 1:
        parsed_args.gradient_accumulation_steps = int(env_accum)
    if parsed_args.gradient_accumulation_steps < 1:
        parsed_args.gradient_accumulation_steps = 1

    return parsed_args

def load_entity_types(type_map_path, entity2id):
    if not type_map_path or not os.path.exists(type_map_path):
        return {}
    with open(type_map_path, 'r') as fin:
        type_map = json.load(fin)
    entity_types = {}
    for entity, etype in type_map.items():
        if entity in entity2id:
            entity_types[entity2id[entity]] = etype
    return entity_types

def load_inverse_relations(inverse_map_path, relation2id):
    if not inverse_map_path or not os.path.exists(inverse_map_path):
        return {}
    with open(inverse_map_path, 'r') as fin:
        inverse_map = json.load(fin)
    inverse_id_map = {}
    for rel_name, inv_name in inverse_map.items():
        if rel_name not in relation2id:
            logging.warning('Inverse map key %s not present in relations.dict', rel_name)
            continue
        if inv_name not in relation2id:
            relation2id[inv_name] = len(relation2id)
        inverse_id_map[relation2id[rel_name]] = relation2id[inv_name]
    logging.info('Loaded %d asymmetric inverse mappings.', len(inverse_id_map))
    return inverse_id_map

def add_inverse_triples(triples, inverse_id_map):
    if not inverse_id_map:
        return triples
    augmented = list(triples)
    for h, r, t in triples:
        inv_r = inverse_id_map.get(r)
        if inv_r is not None:
            augmented.append((t, inv_r, h))
    return augmented

def build_adjacency(triples):
    adjacency = defaultdict(list)
    for h, r, t in triples:
        adjacency[h].append((r, t))
    return adjacency

def enumerate_paths(adjacency, hops, max_paths_per_hop=None, seed=0):
    rng = random.Random(seed)
    all_paths = []
    max_hop = max(hops) if hops else 0
    if max_hop < 2:
        return all_paths

    for hop in hops:
        if hop < 2:
            continue
        hop_paths = []
        for head, neighbors in adjacency.items():
            partial = [(head, [rel], tail) for rel, tail in neighbors]
            depth = 1
            current = partial
            while depth < hop:
                next_paths = []
                for _, rels, tail in current:
                    for rel_next, tail_next in adjacency.get(tail, []):
                        next_paths.append((head, rels + [rel_next], tail_next))
                current = next_paths
                depth += 1
                if not current:
                    break
            hop_paths.extend(current)
        rng.shuffle(hop_paths)
        if max_paths_per_hop is not None:
            hop_paths = hop_paths[:max_paths_per_hop]
        all_paths.extend(hop_paths)
    return all_paths

def build_two_hop_cache(adjacency, max_candidates=256, seed=0):
    rng = random.Random(seed)
    cache = {}
    for head, neighbors in adjacency.items():
        candidates = set()
        for _, tail in neighbors:
            candidates.add(tail)
            for _, t2 in adjacency.get(tail, []):
                candidates.add(t2)
        if not candidates:
            continue
        cand_list = list(candidates)
        rng.shuffle(cand_list)
        cache[head] = cand_list[:max_candidates]
    return cache

def should_enable_path_loss(step, args):
    if args.path_loss_weight <= 0:
        return False
    if args.path_curriculum_steps is None:
        return True
    start, end = args.path_curriculum_steps
    return (step >= start) and (end <= 0 or step <= end)

def override_config(args):
    '''
    Override model and data configuration
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']
    args.init_modulus_weight = argparse_dict.get('init_modulus_weight', 3.0) # adding the new modulus weight parameter


def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    model_to_save = model.module if hasattr(model, 'module') else model

    torch.save({
        **save_variable_list,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )
    
    entity_embedding = model_to_save.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'), 
        entity_embedding
    )
    
    relation_embedding = model_to_save.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'), 
        relation_embedding
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

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

# Function to adjust learning rate

# def adjust_learning_rate(optimizer, step, max_steps, initial_lr, final_lr):
#     """
#     Linearly decays learning rate from initial_lr to final_lr based on current training step.
#     """
#     # progress = step / max_steps
#     # new_lr = initial_lr - (initial_lr - final_lr) * progress
#     # new_lr = max(new_lr, final_lr)  # Clamp to final_lr if needed

#     # Cosine decay, polynomial decay, or simple linear decay (easy one here)
#     decay_ratio = step / max_steps
#     new_lr = initial_lr * (1.0 - decay_ratio) + final_lr * decay_ratio

#     for param_group in optimizer.param_groups:
#         param_group['lr'] = new_lr

# Function to add reciprocal triples

def add_reciprocal_triples(triples, nrelation):
    '''
    Function  to add reciprocal triples
    This function takes a list of triples and the number of relations
    It creates reciprocal triples by swapping the head and tail entities
    '''
    reciprocal_triples = []
    for h, r, t in triples:
        reciprocal_triples.append((t, r + nrelation, h))
    return triples + reciprocal_triples

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
        
        
def main(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')
    
    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')
    
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # Write logs to checkpoint and console
    set_logger(args)
    
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    
    inverse_id_map = load_inverse_relations(args.inverse_map_path, relation2id)
    
    entity_types = load_entity_types(args.type_map_path, entity2id)
    
    # Read regions for Countries S* datasets
    if args.countries:
        regions = list()
        with open(os.path.join(args.data_path, 'regions.list')) as fin:
            for line in fin:
                region = line.strip()
                regions.append(entity2id[region])
        args.regions = regions

    nentity = len(entity2id)
    nrelation = len(relation2id)
    
    args.nentity = nentity
    args.nrelation = nrelation
    
    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    
    # train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    # logging.info('#train: %d' % len(train_triples))
    # valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    # logging.info('#valid: %d' % len(valid_triples))
    # test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    # logging.info('#test: %d' % len(test_triples))

    #  Reading triples
    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    test_triples  = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)

    train_triples_raw = list(train_triples)
    train_triples = add_inverse_triples(train_triples, inverse_id_map)
    valid_triples = add_inverse_triples(valid_triples, inverse_id_map)
    test_triples  = add_inverse_triples(test_triples, inverse_id_map)
    
    #All true triples
    all_true_triples = train_triples + valid_triples + test_triples
    
    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        # nrelation=nrelation,
        nrelation=args.nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
        #ERAS variant
        use_eras=args.use_eras,
        k_prototypes=args.k_prototypes,
        # Type constraints
        type_map_path=args.type_map_path,
        entity2id=entity2id, 
        init_modulus_weight=args.init_modulus_weight,
        init_rel_width=args.init_rel_width,
        phase_harmonics=args.phase_harmonics

    )
    
    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    if args.cuda and not torch.cuda.is_available():
        logging.warning('CUDA requested but not available. Falling back to CPU execution.')
        args.cuda = False
        device = torch.device('cpu')

    kge_model = kge_model.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        logging.info('Multiple GPUs detected (%d). Enabling DataParallel.', torch.cuda.device_count())
        kge_model = torch.nn.DataParallel(kge_model)
    
    path_iterator = None
    if args.do_train and args.path_loss_weight > 0:
        adjacency = build_adjacency(train_triples_raw)
        two_hop_cache = build_two_hop_cache(adjacency, max_candidates=args.path_negative_size * 16)
        path_bank = enumerate_paths(adjacency, args.path_hops, args.path_max_per_hop)
        if not path_bank:
            logging.warning('Requested path loss but no admissible paths were found.')
        else:
            logging.info('Constructed %d multi-hop paths for training.', len(path_bank))
            path_dataset = PathDataset(
                path_bank,
                nentity,
                negative_sample_size=args.path_negative_size,
                entity_types=entity_types,
                two_hop_cache=two_hop_cache
            )
            path_dataloader = DataLoader(
                path_dataset,
                batch_size=args.path_batch_size,
                shuffle=True,
                num_workers=max(1, args.cpu_num//2),
                collate_fn=PathDataset.collate_fn
            )
            path_iterator = BidirectionalOneShotIterator.one_shot_iterator(path_dataloader)

    if args.do_train:
        train_samplers = []
        train_dataset_head = TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch')
        train_dataset_tail = TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch')
        train_sampler_head = None
        train_sampler_tail = None

        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            train_dataset_head,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        
        train_dataloader_tail = DataLoader(
            train_dataset_tail,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        
        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()), 
            lr=current_learning_rate,weight_decay=0.000001
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.lr_t_max or args.max_steps,
            eta_min=args.lr_eta_min
        )

        manual_lr_drop_steps = set(args.lr_drop_steps or [])
        applied_lr_drop_steps = set()
        secondary_warmup_applied = False

        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    init_step = 0
    best_val_mrr = 0.0
    best_step = 0
    patience_counter = 0

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        model_to_load = kge_model.module if hasattr(kge_model, 'module') else kge_model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)

    
    step = init_step
    
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)
    
    # Set valid dataloader as it would be evaluated during training
    
    if args.do_train:
        logging.info('learning_rate = %.6f' % current_learning_rate)

        training_logs = []
        
        grad_steps = max(1, args.gradient_accumulation_steps)
        #Training Loop
        stop_training = False
        for step in range(init_step, args.max_steps):
            micro_logs = []
            for accum_idx in range(grad_steps):
                path_batch = None
                path_weight = args.path_loss_weight
                if path_iterator and should_enable_path_loss(step, args):
                    path_batch = next(path_iterator)
                else:
                    path_weight = 0.0

                micro_log = KGEModel.train_step(
                    kge_model,
                    optimizer,
                    train_iterator,
                    args,
                    step=step,
                    path_batch=path_batch,
                    path_weight=path_weight,
                    consistency_weight=args.path_consistency_weight,
                    zero_grad=(accum_idx == 0),
                    optimizer_step=(accum_idx == grad_steps - 1),
                    accumulation_steps=grad_steps
                )
                micro_logs.append(micro_log)

            log = {}
            for key in micro_logs[0].keys():
                log[key] = sum(m[key] for m in micro_logs) / len(micro_logs)

            scheduler.step()   # Smooth cosine update
            current_learning_rate = optimizer.param_groups[0]['lr']

            if (not secondary_warmup_applied
                and args.secondary_warmup_step is not None
                and step == args.secondary_warmup_step):
                secondary_warmup_applied = True
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.secondary_warmup_gamma
                scheduler.base_lrs = [base * args.secondary_warmup_gamma for base in scheduler.base_lrs]
                if hasattr(scheduler, '_last_lr'):
                    scheduler._last_lr = [lr * args.secondary_warmup_gamma for lr in scheduler._last_lr]
                current_learning_rate = optimizer.param_groups[0]['lr']
                logging.info(
                    'Secondary LR warm-up applied at step %d: Learning Rate = %.6e',
                    step,
                    current_learning_rate
                )

            if manual_lr_drop_steps and step in manual_lr_drop_steps and step not in applied_lr_drop_steps:
                applied_lr_drop_steps.add(step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.lr_drop_gamma
                scheduler.base_lrs = [base * args.lr_drop_gamma for base in scheduler.base_lrs]
                if hasattr(scheduler, '_last_lr'):
                    scheduler._last_lr = [lr * args.lr_drop_gamma for lr in scheduler._last_lr]
                current_learning_rate = optimizer.param_groups[0]['lr']
                logging.info(
                    'Manual LR drop applied at step %d: Learning Rate = %.6e',
                    step,
                    current_learning_rate
                )

            
            training_logs.append(log)
            
            # if step >= warm_up_steps:
            #     current_learning_rate = current_learning_rate / 10
            #     logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
            #     optimizer = torch.optim.Adam(
            #         filter(lambda p: p.requires_grad, kge_model.parameters()), 
            #         lr=current_learning_rate
            #     )
            #     warm_up_steps = warm_up_steps * 3

        # Adjust learning rate  
        # Smooth LR decay every step
            # adjust_learning_rate(
            #     optimizer,
            #     step,
            #     max_steps=args.max_steps,
            #     initial_lr=args.learning_rate,
            #     final_lr=1e-5  # tune the final learning rate as needed
            # )
            # ✏️ Log the learning rate decay every 1000 steps (or any interval you want)
            if step % 1000 == 0:
                current_learning_rate = optimizer.param_groups[0]['lr']
                logging.info(f"Step {step}: Adjusted learning rate to {current_learning_rate:.6e}")


            # Saves checkpoint at every N steps
            # if step % args.save_checkpoint_steps == 0:
            #     save_variable_list = {
            #         'step': step, 
            #         'current_learning_rate': current_learning_rate,
            #         'warm_up_steps': warm_up_steps
            #     }
            #     save_model(kge_model, optimizer, save_variable_list, args)
                
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Training average', step, metrics)


                # 📝 Log LR too
                current_learning_rate = optimizer.param_groups[0]['lr']
                logging.info(f"Step {step}: Learning Rate = {current_learning_rate:.8f}")


                # training_logs = []

            # # Add this at the top of the training loop
            # best_val_mrr = 0.0
            # best_step = 0 
                
            if args.do_valid and step % args.valid_steps == 0:
                logging.info('Evaluating on Valid Dataset...')
                metrics = KGEModel.test_step(kge_model, valid_triples, all_true_triples, args)
                log_metrics('Valid', step, metrics)

                improved = metrics['MRR'] > (best_val_mrr + args.early_stop_min_delta)
                if improved:
                    if step != best_step:
                        logging.info(f'New best model at step {step}, MRR: {metrics["MRR"]:.4f}')
                    best_val_mrr = metrics['MRR']
                    best_step = step
                    patience_counter = 0

                    save_variable_list = {
                        'step': step,
                        'current_learning_rate': current_learning_rate,
                        'warm_up_steps': warm_up_steps
                    }
                    save_model(kge_model, optimizer, save_variable_list, args)
                else:
                    if args.stop_at_first_peak and best_val_mrr > 0:
                        logging.info(
                            'Validation MRR dropped from %.4f to %.4f at step %d, early stopping triggered.',
                            best_val_mrr,
                            metrics['MRR'],
                            step
                        )
                        stop_training = True
                        break
                    if args.early_stop_patience:
                        patience_counter += 1
                        logging.info(
                            'Validation did not improve best MRR %.4f (current %.4f). Patience %d/%d.',
                            best_val_mrr,
                            metrics['MRR'],
                            patience_counter,
                            args.early_stop_patience
                        )
                        if patience_counter >= args.early_stop_patience:
                            logging.info(
                                'Early stopping triggered after %d non-improving validations.',
                                args.early_stop_patience
                            )
                            stop_training = True
                            break


        if stop_training:
            logging.info('Stopping training loop due to early stopping condition.')

        # save_variable_list = {
        #     'step': step, 
        #     'current_learning_rate': current_learning_rate,
        #     'warm_up_steps': warm_up_steps
        # }
        # save_model(kge_model, optimizer, save_variable_list, args)
        
    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = KGEModel.test_step(kge_model, valid_triples, all_true_triples, args)
        log_metrics('Valid', step, metrics)
    
    # if args.do_test:
    #     logging.info('Evaluating on Test Dataset...')
    #     metrics = kge_model.test_step(kge_model, test_triples, all_true_triples, args)
    #     log_metrics('Test', step, metrics)

    # After training, load the best model before testing
    if args.do_test:
        # logging.info(f'Loading best model from step {best_step}...')
        logging.info(f" Using best validation model from step {best_step} for final test evaluation.")
        checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint'))
        model_to_load = kge_model.module if hasattr(kge_model, 'module') else kge_model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        metrics = KGEModel.test_step(kge_model, test_triples, all_true_triples, args)
        log_metrics('Test', best_step, metrics)
    
    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics = KGEModel.test_step(kge_model, train_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)
if __name__ == '__main__':
    main(parse_args())