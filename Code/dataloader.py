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
        neg_list = [item[1].view(-1) for item in data]
        max_neg_len = max((neg.numel() for neg in neg_list), default=1)
        padded_negs = []
        for neg in neg_list:
            if neg.numel() < max_neg_len:
                repeat = max_neg_len - neg.numel()
                extra = neg[torch.arange(repeat) % neg.numel()]
                neg = torch.cat([neg, extra], dim=0)
            padded_negs.append(neg)
        negatives = torch.stack(padded_negs, dim=0)

        for i, (pos, _) in enumerate(data):
            rel_ids = pos[1]
            path_tensor[i, :rel_ids.size(0)] = rel_ids

        return heads, path_tensor, path_lengths, tails, negatives

class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode,
                 hard_relation_ids=None, hard_negative_fraction=0.5,
                 structural_cache=None, structural_negative_fraction=0.0,
                 candidate_cache=None, candidate_negative_fraction=0.0,
                 emu_tail_cache=None, emu_head_cache=None, emu_negative_fraction=0.0):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
        self.hard_relation_ids = set(hard_relation_ids or [])
        if self.hard_relation_ids:
            self.hard_negative_fraction = max(0.0, min(1.0, hard_negative_fraction))
            self.relation_head_pool, self.relation_tail_pool = self._build_relation_pools(self.triples)
        else:
            self.hard_negative_fraction = 0.0
            self.relation_head_pool = {}
            self.relation_tail_pool = {}
        self.structural_cache = structural_cache or {}
        self.structural_negative_fraction = max(0.0, min(1.0, structural_negative_fraction))
        self.candidate_cache = candidate_cache or {}
        self.candidate_negative_fraction = max(0.0, min(1.0, candidate_negative_fraction))
        self.emu_tail_cache = emu_tail_cache or {}
        self.emu_head_cache = emu_head_cache or {}
        self.emu_negative_fraction = max(0.0, min(1.0, emu_negative_fraction))
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        negative_sample_list = []
        negative_sample_size = 0
        hard_samples = np.array([], dtype=np.int64)

        if self.hard_negative_fraction > 0 and relation in self.hard_relation_ids:
            hard_quota = int(self.negative_sample_size * self.hard_negative_fraction)
            if hard_quota > 0:
                if self.mode == 'tail-batch':
                    exclude = set(self.true_tail.get((head, relation), []))
                    exclude.add(tail)
                    pool = self.relation_tail_pool.get(relation)
                else:
                    exclude = set(self.true_head.get((relation, tail), []))
                    exclude.add(head)
                    pool = self.relation_head_pool.get(relation)
                hard_samples = self._sample_from_pool(pool, exclude, hard_quota)
                negative_sample_list.append(hard_samples)
                negative_sample_size += hard_samples.size

        if (self.structural_negative_fraction > 0
                and relation in self.hard_relation_ids
                and self.mode == 'tail-batch'
                and self.structural_cache):
            struct_quota = int(self.negative_sample_size * self.structural_negative_fraction)
            if struct_quota > 0:
                candidates = self.structural_cache.get((head, relation))
                if candidates is None:
                    candidates = self.structural_cache.get(head)
                if candidates:
                    exclude = set(self.true_tail.get((head, relation), []))
                    exclude.add(tail)
                    struct_samples = self._sample_from_candidates(candidates, exclude, struct_quota)
                    if struct_samples.size > 0:
                        negative_sample_list.append(struct_samples)
                        negative_sample_size += struct_samples.size

        if (self.candidate_negative_fraction > 0
                and self.mode == 'tail-batch'
                and self.candidate_cache):
            cand_quota = int(self.negative_sample_size * self.candidate_negative_fraction)
            if cand_quota > 0:
                candidates = self.candidate_cache.get((head, relation))
                if candidates is None:
                    candidates = self.candidate_cache.get(head)
                if candidates:
                    exclude = set(self.true_tail.get((head, relation), []))
                    exclude.add(tail)
                    cand_samples = self._sample_from_candidates(candidates, exclude, cand_quota)
                    if cand_samples.size > 0:
                        negative_sample_list.append(cand_samples)
                        negative_sample_size += cand_samples.size

        if self.emu_negative_fraction > 0:
            emu_quota = int(self.negative_sample_size * self.emu_negative_fraction)
            if emu_quota > 0:
                if self.mode == 'tail-batch' and self.emu_tail_cache:
                    candidates = self.emu_tail_cache.get((head, relation))
                    if candidates is None:
                        candidates = self.emu_tail_cache.get(head)
                    if candidates:
                        exclude = set(self.true_tail.get((head, relation), []))
                        exclude.add(tail)
                        emu_samples = self._sample_from_candidates(candidates, exclude, emu_quota)
                        if emu_samples.size > 0:
                            negative_sample_list.append(emu_samples)
                            negative_sample_size += emu_samples.size
                elif self.mode == 'head-batch' and self.emu_head_cache:
                    candidates = self.emu_head_cache.get((relation, tail))
                    if candidates is None:
                        candidates = self.emu_head_cache.get(tail)
                    if candidates:
                        exclude = set(self.true_head.get((relation, tail), []))
                        exclude.add(head)
                        emu_samples = self._sample_from_candidates(candidates, exclude, emu_quota)
                        if emu_samples.size > 0:
                            negative_sample_list.append(emu_samples)
                            negative_sample_size += emu_samples.size

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

    def _sample_from_pool(self, pool, exclude, size):
        if pool is None or size <= 0:
            return np.array([], dtype=np.int64)
        candidates = pool
        if exclude:
            mask = ~np.in1d(candidates, list(exclude), assume_unique=False)
            filtered = candidates[mask]
        else:
            filtered = candidates
        if filtered.size == 0:
            return np.array([], dtype=np.int64)
        replace = filtered.size < size
        sampled = np.random.choice(filtered, size=size, replace=replace)
        return sampled

    def _sample_from_candidates(self, candidates, exclude, size):
        if not candidates or size <= 0:
            return np.array([], dtype=np.int64)
        arr = np.array(candidates, dtype=np.int64)
        if exclude:
            mask = ~np.in1d(arr, list(exclude), assume_unique=False)
            arr = arr[mask]
        if arr.size == 0:
            return np.array([], dtype=np.int64)
        replace = arr.size < size
        sampled = np.random.choice(arr, size=size, replace=replace)
        return sampled

    @staticmethod
    def _build_relation_pools(triples):
        head_pool = {}
        tail_pool = {}
        for head, relation, tail in triples:
            head_pool.setdefault(relation, set()).add(head)
            tail_pool.setdefault(relation, set()).add(tail)
        for rel in head_pool:
            head_pool[rel] = np.array(list(head_pool[rel]), dtype=np.int64)
        for rel in tail_pool:
            tail_pool[rel] = np.array(list(tail_pool[rel]), dtype=np.int64)
        return head_pool, tail_pool
    
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
