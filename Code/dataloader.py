#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset

class PathDataset(Dataset):
    """
    Dataset of multi-hop paths (head, [relations], tail) for auxiliary training.
    Each item returns the path specification plus sampled negative tails that
    follow the same relation chain but lead to incorrect entities.
    """
    def __init__(
        self,
        paths,
        nentity,
        negative_sample_size=10,
        entity_types=None,
        two_hop_cache=None
    ):
        """
        paths: list of tuples (head_id, [relation_ids], tail_id)
        entity_types: dict mapping entity_id -> type string (optional)
        two_hop_cache: dict head_id -> set of candidate negative ids (optional)
        """
        self.paths = paths
        self.nentity = nentity
        self.negative_sample_size = negative_sample_size
        self.entity_types = entity_types or {}
        self.two_hop_cache = two_hop_cache or {}
        self.entities = np.arange(self.nentity)

        self.path_true_tails = {}
        for head, rels, tail in self.paths:
            key = (head, tuple(rels))
            self.path_true_tails.setdefault(key, set()).add(tail)

        self.type_to_entities = {}
        if self.entity_types:
            for ent, typ in self.entity_types.items():
                self.type_to_entities.setdefault(typ, []).append(ent)
            for typ, ents in self.type_to_entities.items():
                self.type_to_entities[typ] = np.array(ents, dtype=np.int64)

    def __len__(self):
        return len(self.paths)

    def sample_negative_tail(self, head, rels, tail):
        key = (head, tuple(rels))
        true_tails = self.path_true_tails.get(key, set())
        negatives = []
        attempts = 0
        max_attempts = self.negative_sample_size * 10

        while len(negatives) < self.negative_sample_size and attempts < max_attempts:
            attempts += 1
            candidate = None

            if self.entity_types and np.random.rand() < 0.5:
                tail_type = self.entity_types.get(tail)
                if tail_type in self.type_to_entities:
                    bucket = self.type_to_entities[tail_type]
                    candidate = int(np.random.choice(bucket))

            if candidate is None:
                candidates = self.two_hop_cache.get(head)
                if candidates:
                    candidate = int(np.random.choice(list(candidates)))
                else:
                    candidate = int(np.random.choice(self.entities))

            if candidate == tail or candidate in true_tails:
                continue
            negatives.append(candidate)

        if not negatives:
            negatives.append(int(np.random.choice(self.entities)))

        return torch.LongTensor(negatives)

    def __getitem__(self, idx):
        head, rels, tail = self.paths[idx]
        rels_tensor = torch.LongTensor(rels)
        negative_tails = self.sample_negative_tail(head, rels, tail)
        positive = (head, rels_tensor, tail)
        return positive, negative_tails

    @staticmethod
    def collate_fn(data):
        batch_size = len(data)
        rel_lengths = [item[0][1].size(0) for item in data]
        max_len = max(rel_lengths) if rel_lengths else 1
        path_tensor = torch.full((batch_size, max_len), -1, dtype=torch.long)
        path_lengths = torch.LongTensor(rel_lengths)
        heads = torch.LongTensor([item[0][0] for item in data])
        tails = torch.LongTensor([item[0][2] for item in data])
        negatives = torch.stack([item[1] for item in data], dim=0)

        for i, (pos, _) in enumerate(data):
            rel_ids = pos[1]
            path_tensor[i, :rel_ids.size(0)] = rel_ids

        return heads, path_tensor, path_lengths, tails, negatives

class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_head[(relation, tail)], 
                    assume_unique=True, 
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_tail[(head, relation)], 
                    assume_unique=True, 
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        negative_sample = torch.LongTensor(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)
            
        return positive_sample, negative_sample, subsampling_weight, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode
    
    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail

    
class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)
            
        tmp = torch.LongTensor(tmp)            
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))
            
        return positive_sample, negative_sample, filter_bias, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode
    
class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
