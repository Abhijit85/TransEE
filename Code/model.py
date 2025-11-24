#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, 
                 double_entity_embedding=False, double_relation_embedding=False,
             use_eras=False, k_prototypes=4,type_map_path=None, entity2id=None,type_lambda=1.0,init_modulus_weight=3.5,init_rel_width=0.1, phase_harmonics=2,
             modulus_sharpness=1.0, phase_sharpness=1.0):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.type_lambda = type_lambda
        self.modulus_sharpness = modulus_sharpness
        self.phase_sharpness = phase_sharpness


        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim

        #Slope-Weighted L1 Versions 
        # self.rel_width = nn.Parameter(torch.ones(nrelation, self.relation_dim // 2))
        self.rel_width = nn.Parameter(torch.full((nrelation, self.relation_dim // 2), init_rel_width))





        # Debugging

        # print(f"Entity Embedding Dimension: {self.entity_dim}")
        # print(f"Relation Embedding Dimension: {self.relation_dim}")


        # ERAS variant
        self.use_eras = use_eras
        self.k_prototypes = k_prototypes

        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )

        # Learnable weights for RelatE score decomposition with Per-Relation Tensors
        # as vectors per-relation
        if model_name == 'RelatE':
            # self.phase_weight = nn.Parameter(torch.Tensor([1.0]))
            # self.modulus_weight = nn.Parameter(torch.Tensor([3.5]))
            self.phase_weight = nn.Parameter(torch.ones(self.nrelation, 1) * (init_modulus_weight * 0.65))
            self.modulus_weight = nn.Parameter(torch.ones(self.nrelation, 1) * init_modulus_weight)
            self.phase_harmonics = max(1, phase_harmonics)
            self.phase_freq_param = nn.Parameter(torch.ones(self.nrelation, self.phase_harmonics))
        else:
            self.phase_harmonics = 1
            self.phase_freq_param = None

           
        self.use_type_bias = False  # Default
        self.base_nrelation = 0

        #  Store the entity2id mapping for lookups
        self.entity2id = entity2id
        self.tie_inverses = False

        if type_map_path is not None and os.path.exists(type_map_path):
            import json
            with open(type_map_path, 'r') as f:
                type_map = json.load(f)

            all_types = sorted(set(type_map.values()))
            self.type_to_id = {t: i for i, t in enumerate(all_types)}

            # Initialize entity_type_ids tensor
            self.entity_type_ids = torch.zeros(nentity, dtype=torch.long)

            missing = 0
            for ent_name, type_name in type_map.items():
                if ent_name in self.entity2id:
                    eid = self.entity2id[ent_name]
                    self.entity_type_ids[eid] = self.type_to_id.get(type_name, 0)
                else:
                    missing += 1

            logging.info(f" Loaded type bias: {len(type_map)} entities, {missing} unmapped.")

            # self.type_embedding = nn.Embedding(len(self.type_to_id), hidden_dim)
            # self.use_type_bias = True

            self.type_embedding = nn.Embedding(len(self.type_to_id), self.entity_dim)
            self.use_type_bias = True





        '''
        Creating protoptype and attention for ERAS search
        ERAS: Only initialize if requested for RelatE
        '''
        if model_name == 'RelatE' and use_eras:
            self.rel_prototypes = nn.Parameter(torch.randn(
                k_prototypes, self.relation_dim
            ))

            self.eras_attention = nn.Sequential(
                nn.Linear(self.entity_dim * 2, k_prototypes),
                nn.Softmax(dim=-1)
            )


        
        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE','RelatE']:
            raise ValueError('model %s not supported' % model_name)
            
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')
        
    def _lookup_relation_embedding(self, relation_ids):
        rel = torch.index_select(
            self.relation_embedding,
            dim=0,
            index=relation_ids.view(-1)
        ).view(*relation_ids.shape, -1)

        if self.tie_inverses and self.base_nrelation > 0:
            mask = relation_ids >= self.base_nrelation
            if mask.any():
                base_ids = relation_ids.clone()
                base_ids[mask] -= self.base_nrelation
                base_rel = torch.index_select(
                    self.relation_embedding,
                    dim=0,
                    index=base_ids.view(-1)
                ).view(*relation_ids.shape, -1)
                mask_expand = mask.unsqueeze(-1)
                rel = torch.where(mask_expand, base_rel, rel)
                phase_dim = rel.size(-1) // 2
                phase_slice = rel[..., phase_dim:]
                phase_slice = torch.where(mask_expand, -phase_slice, phase_slice)
                rel = torch.cat([rel[..., :phase_dim], phase_slice], dim=-1)
        return rel

    def aggregate_relation_weights(self, relation_ids, mask=None):
        if relation_ids.dim() == 1:
            relation_ids = relation_ids.unsqueeze(1)
            if mask is None:
                mask = torch.ones_like(relation_ids, dtype=torch.bool)
        if mask is None:
            mask = relation_ids >= 0
        safe_ids = relation_ids.clone()
        safe_ids[~mask] = 0
        mask_float = mask.float()
        denom = mask_float.sum(dim=1, keepdim=True).clamp(min=1.0)

        phase_vals = F.softplus(self.phase_weight[safe_ids]).squeeze(-1)
        modulus_vals = F.softplus(self.modulus_weight[safe_ids]).squeeze(-1)
        if self.phase_freq_param is not None:
            freq_vals = F.softplus(self.phase_freq_param[safe_ids])
        else:
            freq_vals = torch.ones(*safe_ids.shape, 1, device=relation_ids.device)

        phase_avg = (phase_vals * mask_float).sum(dim=1, keepdim=True) / denom
        modulus_avg = (modulus_vals * mask_float).sum(dim=1, keepdim=True) / denom
        freq_mask = mask.unsqueeze(-1).float()
        freq_avg = (freq_vals * freq_mask).sum(dim=1) / denom
        return phase_avg, modulus_avg, freq_avg

    def compute_phase_component(self, phase_argument, freq_weights):
        """
        phase_argument: [B, N, D] raw phase difference (no /2 applied yet)
        freq_weights: [B, K]
        """
        if freq_weights.dim() == 1:
            freq_weights = freq_weights.unsqueeze(1)
        batch = phase_argument.size(0)
        scores = 0
        for idx in range(freq_weights.size(1)):
            harmonic = idx + 1
            weight = freq_weights[:, idx].view(batch, 1, 1)
            scores = scores + weight * torch.abs(torch.sin(harmonic * phase_argument / 2))
        return scores
        
    # def forward(self, sample, mode='single', step=None):
    #     '''
    #     Forward function that calculate the score of a batch of triples.
    #     In the 'single' mode, sample is a batch of triple.
    #     In the 'head-batch' or 'tail-batch' mode, sample consists two part.
    #     The first part is usually the positive sample.
    #     And the second part is the entities in the negative samples.
    #     Because negative samples and positive samples usually share two elements 
    #     in their triple ((head, relation) or (relation, tail)).

    #     '''
    #     # Type constraint original head saving steps
    #     if mode == 'single':
    #         self.current_positive_sample = sample  # [B, 3]

    #     elif mode == 'head-batch':
    #         tail_part, head_part = sample
    #         self.current_positive_sample = tail_part  # [B, 3] → head is corrupted, so we save tail-part

    #     elif mode == 'tail-batch':
    #         head_part, tail_part = sample
    #         self.current_positive_sample = head_part  # [B, 3] → tail is corrupted, so we save head-part


    #     self.current_step = step  # Save the step
    #     if mode == 'single':
    #         batch_size, negative_sample_size = sample.size(0), 1
            
    #         head = torch.index_select(
    #             self.entity_embedding, 
    #             dim=0, 
    #             index=sample[:,0]
    #         ).unsqueeze(1)
            
    #         relation = torch.index_select(
    #             self.relation_embedding, 
    #             dim=0, 
    #             index=sample[:,1]
    #         ).unsqueeze(1)
            
    #         tail = torch.index_select(
    #             self.entity_embedding, 
    #             dim=0, 
    #             index=sample[:,2]
    #         ).unsqueeze(1)
            
    #     elif mode == 'head-batch':
    #         tail_part, head_part = sample
    #         batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
    #         head = torch.index_select(
    #             self.entity_embedding, 
    #             dim=0, 
    #             index=head_part.view(-1)
    #         ).view(batch_size, negative_sample_size, -1)
            
    #         relation = torch.index_select(
    #             self.relation_embedding, 
    #             dim=0, 
    #             index=tail_part[:, 1]
    #         ).unsqueeze(1)
            
    #         tail = torch.index_select(
    #             self.entity_embedding, 
    #             dim=0, 
    #             index=tail_part[:, 2]
    #         ).unsqueeze(1)
            
    #     elif mode == 'tail-batch':
    #         head_part, tail_part = sample
    #         batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
    #         head = torch.index_select(
    #             self.entity_embedding, 
    #             dim=0, 
    #             index=head_part[:, 0]
    #         ).unsqueeze(1)
            
    #         relation = torch.index_select(
    #             self.relation_embedding,
    #             dim=0,
    #             index=head_part[:, 1]
    #         ).unsqueeze(1)
            
    #         tail = torch.index_select(
    #             self.entity_embedding, 
    #             dim=0, 
    #             index=tail_part.view(-1)
    #         ).view(batch_size, negative_sample_size, -1)
            
    #     else:
    #         raise ValueError('mode %s not supported' % mode)
            
    #     model_func = {
    #         'TransE': self.TransE,
    #         'DistMult': self.DistMult,
    #         'ComplEx': self.ComplEx,
    #         'RotatE': self.RotatE,
    #         'pRotatE': self.pRotatE,
    #         # 'RelatE': self.RelatE,
    #         'RelatE': self.RelatE_ERAS if getattr(self, 'use_eras', False) else self.RelatE
    #     }
        
    #     if self.model_name in model_func:
    #         score = model_func[self.model_name](head, relation, tail, mode)
    #     else:
    #         raise ValueError('model %s not supported' % self.model_name)
        
    #     return score

    # def forward(self, sample, mode='single', step=None):
    #     self.current_step = step

    #     if mode == 'single':
    #         batch_size, negative_sample_size = sample.size(0), 1

    #         head_ids = sample[:, 0]
    #         relation_ids = sample[:, 1]
    #         tail_ids = sample[:, 2]

    #         head = torch.index_select(self.entity_embedding, dim=0, index=head_ids).unsqueeze(1)
    #         relation = torch.index_select(self.relation_embedding, dim=0, index=relation_ids).unsqueeze(1)
    #         tail = torch.index_select(self.entity_embedding, dim=0, index=tail_ids).unsqueeze(1)

    #     elif mode == 'head-batch':
    #         tail_part, head_part = sample
    #         batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

    #         head_ids = head_part[:, 0]
    #         relation_ids = tail_part[:, 1]
    #         tail_ids = tail_part[:, 2]

    #         head = torch.index_select(self.entity_embedding, dim=0, index=head_part.view(-1)).view(batch_size, negative_sample_size, -1)
    #         relation = torch.index_select(self.relation_embedding, dim=0, index=relation_ids).unsqueeze(1)
    #         tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part[:, 2]).unsqueeze(1)

    #     elif mode == 'tail-batch':
    #         head_part, tail_part = sample
    #         batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

    #         head_ids = head_part[:, 0]
    #         relation_ids = head_part[:, 1]
    #         tail_ids = tail_part[:, 0]

    #         head = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
    #         relation = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1)
    #         tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)

    #     else:
    #         raise ValueError('mode %s not supported' % mode)

    #     model_func = {
    #         'TransE': self.TransE,
    #         'DistMult': self.DistMult,
    #         'ComplEx': self.ComplEx,
    #         'RotatE': self.RotatE,
    #         'pRotatE': self.pRotatE,
    #         'RelatE': self.RelatE_ERAS if getattr(self, 'use_eras', False) else self.RelatE
    #     }

    #     if self.model_name in model_func:
    #         # score = model_func[self.model_name](head, relation, tail, mode, head_ids=head_ids, tail_ids=tail_ids)
    #         score = model_func[self.model_name](head, relation, tail, mode, head_ids=head_ids, tail_ids=tail_ids, relation_ids=relation_ids)

    #     else:
    #         raise ValueError('model %s not supported' % self.model_name)

    #     return score

    def forward(self, sample, mode='single', step=None):
        self.current_step = step

        # Safely extract positive_sample always
        if mode == 'single':
            positive_sample = sample
        elif mode == 'head-batch':
            positive_sample, _ = sample
        elif mode == 'tail-batch':
            positive_sample, _ = sample
        else:
            raise ValueError('Unsupported mode: %s' % mode)

        head_ids = positive_sample[:, 0]
        relation_ids = positive_sample[:, 1]
        tail_ids = positive_sample[:, 2]

        # Now, prepare embeddings based on mode
        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(self.entity_embedding, dim=0, index=head_ids).unsqueeze(1)
            relation = self._lookup_relation_embedding(relation_ids).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_ids).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(self.entity_embedding, dim=0, index=head_part.view(-1)).view(batch_size, negative_sample_size, -1)
            relation = self._lookup_relation_embedding(tail_part[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part[:, 2]).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
            relation = self._lookup_relation_embedding(head_part[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE,
            'RelatE': self.RelatE_ERAS if getattr(self, 'use_eras', False) else self.RelatE
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode,
                                                head_ids=head_ids,
                                                tail_ids=tail_ids,
                                                relation_ids=relation_ids)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def path_forward(self, head_ids, relation_paths, tail_ids):
        """
        Score multi-hop paths specified by a tensor of relation ids (padding with -1).
        head_ids: [B]
        relation_paths: [B, L] with -1 padding
        tail_ids: [B] or [B, N]
        """
        mask = relation_paths >= 0
        safe_ids = relation_paths.clone()
        safe_ids[~mask] = 0
        rel_embed = self._lookup_relation_embedding(safe_ids)
        rel_embed = rel_embed * mask.unsqueeze(-1)

        rel_modulus, rel_phase = torch.chunk(rel_embed, 2, dim=2)
        bias_relation = torch.clamp(rel_modulus, max=1)
        rel_modulus = torch.abs(rel_modulus)
        indicator = (bias_relation < -rel_modulus)
        bias_relation[indicator] = -rel_modulus[indicator]

        phase_lambda, modulus_lambda, freq_weights = self.aggregate_relation_weights(safe_ids, mask=mask)

        phase_path = rel_phase.sum(dim=1, keepdim=True)

        ones_mod = torch.ones_like(rel_modulus)
        a_components = torch.where(mask.unsqueeze(-1), rel_modulus + bias_relation, ones_mod)
        b_components = torch.where(mask.unsqueeze(-1), 1 - bias_relation, ones_mod)
        A_path = torch.prod(a_components, dim=1, keepdim=True)
        B_path = torch.prod(b_components, dim=1, keepdim=True)

        rel_width_vals = torch.index_select(
            self.rel_width,
            dim=0,
            index=safe_ids.view(-1)
        ).view(*safe_ids.shape, -1)
        rel_width_vals = F.softplus(rel_width_vals) * mask.unsqueeze(-1).float()
        width_denom = mask.unsqueeze(-1).float().sum(dim=1, keepdim=True).clamp(min=1.0)
        width_path = rel_width_vals.sum(dim=1, keepdim=True) / width_denom

        head = torch.index_select(self.entity_embedding, dim=0, index=head_ids).unsqueeze(1)
        if tail_ids.dim() == 1:
            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_ids).unsqueeze(1)
        else:
            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_ids.view(-1)
            ).view(tail_ids.size(0), tail_ids.size(1), -1)

        tail_count = tail.size(1)
        head_expand = head.expand(head.size(0), tail_count, -1)

        phase_path_exp = phase_path.expand(head.size(0), tail_count, -1)
        A_path_exp = A_path.expand(head.size(0), tail_count, -1)
        B_path_exp = B_path.expand(head.size(0), tail_count, -1)
        width_path_exp = width_path.expand(head.size(0), tail_count, -1)

        head_modulus, head_phase = torch.chunk(head_expand, 2, dim=2)
        tail_modulus, tail_phase = torch.chunk(tail, 2, dim=2)

        phase_argument = head_phase + phase_path_exp - tail_phase
        phase_component = self.compute_phase_component(phase_argument, freq_weights)
        phase_score = phase_component.sum(dim=2, keepdim=True)

        mod_dist = torch.abs(head_modulus * A_path_exp - tail_modulus * B_path_exp)
        modulus_score = torch.sum(width_path_exp * mod_dist, dim=2, keepdim=True)

        phase_scale = phase_lambda.view(-1, 1, 1)
        mod_scale = modulus_lambda.view(-1, 1, 1)
        phase_score = phase_score * phase_scale
        modulus_score = modulus_score * mod_scale

        return self.gamma.item() - (phase_score + modulus_score)
    
    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        
        #Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)            
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score
    
    # def RelatE(self, head, relation, tail, mode):
    def RelatE(self, head, relation, tail, mode, head_ids=None, tail_ids=None, relation_ids=None):
        assert head_ids is not None, "Head entity IDs must be passed to RelatE."
        assert tail_ids is not None, "Tail entity IDs must be passed to RelatE."
        assert relation_ids is not None, "Relation IDs must be passed to RelatE."

        phase_lambda, modulus_lambda, freq_weights = self.aggregate_relation_weights(relation_ids)
        # Split embeddings
        head_modulus, head_phase = torch.chunk(head, 2, dim=2)
        rel_modulus, rel_phase = torch.chunk(relation, 2, dim=2)
        tail_modulus, tail_phase = torch.chunk(tail, 2, dim=2)

        head_phase = torch.remainder(head_phase, 2 * np.pi)
        rel_phase = torch.remainder(rel_phase, 2 * np.pi)
        tail_phase = torch.remainder(tail_phase, 2 * np.pi)

        # Adjust relation modulus bias
        bias_relation = torch.clamp(rel_modulus, max=1)
        rel_modulus = torch.abs(rel_modulus)
        indicator = (bias_relation < -rel_modulus)
        bias_relation[indicator] = -rel_modulus[indicator]

        # Fetch phase and modulus weights
        phase_w = F.softplus(self.phase_weight[relation_ids]).view(-1, 1)   # [B, 1]
        modulus_w = F.softplus(self.modulus_weight[relation_ids]).view(-1, 1)  # [B, 1]

        sharp_mod = max(self.modulus_sharpness, 1e-6)
        sharp_phase = max(self.phase_sharpness, 1e-6)

        # Compute scores
        if mode == 'head-batch':
            phase_argument = tail_phase - rel_phase - head_phase
            phase_component = self.compute_phase_component(phase_argument, freq_weights)
            if sharp_phase != 1.0:
                phase_component = phase_component.pow(sharp_phase)
            phase_score = phase_component.sum(dim=2, keepdim=True)
            # modulus_score = torch.norm(tail_modulus * (1 - bias_relation) - head_modulus * (rel_modulus + bias_relation), p=2, dim=2, keepdim=True)
            # Assuming self.rel_width is a learnable parameter of shape [nrelation, dim] initialized in __init__:
            # self.rel_width = nn.Parameter(torch.ones(nrelation, embedding_dim // 2))

            # Head-batch
            mod_dist = torch.abs(tail_modulus * (1 - bias_relation) - head_modulus * (rel_modulus + bias_relation))
            if sharp_mod != 1.0:
                mod_dist = mod_dist.pow(sharp_mod)
            # modulus_score = torch.sum(self.rel_width[relation_ids].unsqueeze(1) * mod_dist, dim=2, keepdim=True)
            # rel_width_exp = self.rel_width[relation_ids] 
            rel_width_exp = F.softplus(self.rel_width[relation_ids]) # shape: [B, d]
            if mod_dist.size(1) != 1:  # For head-batch or tail-batch (with negatives)
                rel_width_exp = rel_width_exp.unsqueeze(1).expand(-1, mod_dist.size(1), -1)  # [B, N, d]
                
            else:
                rel_width_exp = rel_width_exp.unsqueeze(1)  # [B, 1, d]
                

            modulus_score = torch.sum(rel_width_exp * mod_dist, dim=2, keepdim=True)



        elif mode == 'tail-batch':
            phase_argument = head_phase + rel_phase - tail_phase
            phase_component = self.compute_phase_component(phase_argument, freq_weights)
            if sharp_phase != 1.0:
                phase_component = phase_component.pow(sharp_phase)
            phase_score = phase_component.sum(dim=2, keepdim=True)
            # modulus_score = torch.norm(head_modulus * (rel_modulus + bias_relation) - tail_modulus * (1 - bias_relation), p=2, dim=2, keepdim=True)

            # Tail-batch
            mod_dist = torch.abs(head_modulus * (rel_modulus + bias_relation) - tail_modulus * (1 - bias_relation))
            if sharp_mod != 1.0:
                mod_dist = mod_dist.pow(sharp_mod)
            # modulus_score = torch.sum(self.rel_width[relation_ids].unsqueeze(1) * mod_dist, dim=2, keepdim=True)
            # rel_width_exp = self.rel_width[relation_ids]  # shape: [B, d]
            rel_width_exp = F.softplus(self.rel_width[relation_ids]) # shape: [B, d]
            if mod_dist.size(1) != 1:  # For head-batch or tail-batch (with negatives)
                rel_width_exp = rel_width_exp.unsqueeze(1).expand(-1, mod_dist.size(1), -1)  # [B, N, d]
            else:
                rel_width_exp = rel_width_exp.unsqueeze(1)  # [B, 1, d]

            modulus_score = torch.sum(rel_width_exp * mod_dist, dim=2, keepdim=True)



        else:  # default
            phase_argument = head_phase + rel_phase - tail_phase
            phase_component = self.compute_phase_component(phase_argument, freq_weights)
            if sharp_phase != 1.0:
                phase_component = phase_component.pow(sharp_phase)
            phase_score = phase_component.sum(dim=2, keepdim=True)
            # modulus_score = torch.norm(head_modulus * (rel_modulus + bias_relation) - tail_modulus * (1 - bias_relation), p=2, dim=2, keepdim=True)

            # Rest
            mod_dist = torch.abs(head_modulus * (rel_modulus + bias_relation) - tail_modulus * (1 - bias_relation))
            if sharp_mod != 1.0:
                mod_dist = mod_dist.pow(sharp_mod)
            # modulus_score = torch.sum(self.rel_width[relation_ids].unsqueeze(1) * mod_dist, dim=2, keepdim=True)
            # rel_width_exp = self.rel_width[relation_ids]  # shape: [B, d]
            rel_width_exp = F.softplus(self.rel_width[relation_ids]) # shape: [B, d]
            if mod_dist.size(1) != 1:  # For head-batch or tail-batch (with negatives)
                rel_width_exp = rel_width_exp.unsqueeze(1).expand(-1, mod_dist.size(1), -1)  # [B, N, d]
            else:
                rel_width_exp = rel_width_exp.unsqueeze(1)  # [B, 1, d]

            modulus_score = torch.sum(rel_width_exp * mod_dist, dim=2, keepdim=True)


        # Expand phase/modulus weights if needed
        if phase_score.size(1) != phase_w.size(1):
            phase_w = phase_w.unsqueeze(1).expand_as(phase_score)   # [B, N, 1]
            modulus_w = modulus_w.unsqueeze(1).expand_as(modulus_score)  # [B, N, 1]


        # Apply weighting
        phase_score = phase_score * phase_lambda.view(-1, 1, 1)
        modulus_score = modulus_score * modulus_lambda.view(-1, 1, 1)

        # Base score calculation
        base_score = self.gamma.item() - (modulus_score + phase_score)

        # Inject type bias if enabled
        # if self.use_type_bias:
        #     entity_type_ids = self.entity_type_ids.to(head.device)
        #     head_type_ids = entity_type_ids[head_ids]
        #     tail_type_ids = entity_type_ids[tail_ids]

        #     head_type_vec = self.type_embedding(head_type_ids)
        #     tail_type_vec = self.type_embedding(tail_type_ids)

        #     if head.size(1) == 1:
        #         head_vec = head.squeeze(1)
        #     else:
        #         head_vec = head[:, 0, :]

        #     if tail.size(1) == 1:
        #         tail_vec = tail.squeeze(1)
        #     else:
        #         tail_vec = tail[:, 0, :]

        #     head_type_bias = torch.sum(tail_type_vec * head_vec, dim=1, keepdim=True)
        #     tail_type_bias = torch.sum(head_type_vec * tail_vec, dim=1, keepdim=True)
        #     type_bias = head_type_bias + tail_type_bias

        #     # Gradual warmup scaling of type lambda
        #     if self.training and self.current_step is not None:
        #         max_warmup_steps = 20000
        #         warmup_scale = min(1.0, self.current_step / max_warmup_steps)
        #         scaled_type_lambda = self.type_lambda * warmup_scale
        #     else:
        #         scaled_type_lambda = self.type_lambda

        #     if type_bias.shape == base_score.shape:
        #         final_score = base_score + scaled_type_lambda * type_bias
        #     else:
        #         final_score = base_score
        # else:
        #     final_score = base_score

        # Inject type bias if enabled
        if self.use_type_bias:
            entity_type_ids = self.entity_type_ids.to(head.device)
            head_type_ids = entity_type_ids[head_ids]
            tail_type_ids = entity_type_ids[tail_ids]

            head_type_vec = self.type_embedding(head_type_ids)
            tail_type_vec = self.type_embedding(tail_type_ids)

            # Handle shapes for training (B, 1, D) vs validation (B, N, D)
            if head.size(1) != 1:
                head_vec = head[:, 0, :]
            else:
                head_vec = head.squeeze(1)

            if tail.size(1) != 1:
                tail_vec = tail[:, 0, :]
            else:
                tail_vec = tail.squeeze(1)

            head_type_bias = torch.sum(tail_type_vec * head_vec, dim=1, keepdim=True)
            tail_type_bias = torch.sum(head_type_vec * tail_vec, dim=1, keepdim=True)
            type_bias = head_type_bias + tail_type_bias

            # Type lambda scaling
            if self.training and self.current_step is not None:
                max_warmup_steps = 20000
                warmup_scale = min(1.0, self.current_step / max_warmup_steps)
                scaled_type_lambda = self.type_lambda * warmup_scale
            else:
                scaled_type_lambda = self.type_lambda

            if type_bias.shape == base_score.shape:
                final_score = base_score + scaled_type_lambda * type_bias
            else:
                final_score = base_score
        else:
            final_score = base_score



        # Return
        return final_score.squeeze(-1)




    def RelatE_ERAS(self, head, relation, tail, mode):
        # Compute attention weights over K prototypes from head and tail
        if mode == 'head-batch':
            # head shape: [B, N, d], tail shape: [B, 1, d]
            tail_expand = tail.expand(-1, head.shape[1], -1)
            att_input = torch.cat([head, tail_expand], dim=2)
        elif mode == 'tail-batch':
            head_expand = head.expand(-1, tail.shape[1], -1)
            att_input = torch.cat([head_expand, tail], dim=2)
        else:
            att_input = torch.cat([head, tail], dim=2)

        att_weights = self.eras_attention(att_input)  # shape: [B, N, k] or [B, 1, k]
        
        # Get weighted relation embedding
        rel_proto = self.rel_prototypes.unsqueeze(0)  # shape: [1, k, d]
        rel_e = torch.matmul(att_weights, rel_proto)  # shape: [B, N, d]

        # Use the existing RelatE implementation with soft-relational embedding
        head_modulus, head_phase = torch.chunk(head, 2, dim=2)
        rel_modulus, rel_phase = torch.chunk(rel_e, 2, dim=2)
        tail_modulus, tail_phase = torch.chunk(tail, 2, dim=2)

        head_phase = torch.remainder(head_phase, 2 * np.pi)
        rel_phase = torch.remainder(rel_phase, 2 * np.pi)
        tail_phase = torch.remainder(tail_phase, 2 * np.pi)

        bias_relation = torch.clamp(rel_modulus, max=1)
        rel_modulus = torch.abs(rel_modulus)
        indicator = (bias_relation < -rel_modulus)
        bias_relation[indicator] = -rel_modulus[indicator]

        if mode == 'head-batch':
            phase_score = torch.abs(torch.sin((tail_phase - rel_phase - head_phase) / 2)).sum(dim=2) * 1
            modulus_score = torch.norm(
                tail_modulus * (1 - bias_relation) - head_modulus * (rel_modulus + bias_relation),
                p=2,
                dim=2
            ) * 3.5
        elif mode == 'tail-batch':
            phase_score = torch.abs(torch.sin((head_phase + rel_phase - tail_phase) / 2)).sum(dim=2) * 1
            modulus_score = torch.norm(
                head_modulus * (rel_modulus + bias_relation) - tail_modulus * (1 - bias_relation),
                p=2,
                dim=2
            ) * 3.5
        else:
            phase_score = torch.abs(torch.sin((head_phase + rel_phase - tail_phase) / 2)).sum(dim=2) * 1
            modulus_score = torch.norm(
                head_modulus * (rel_modulus + bias_relation) - tail_modulus * (1 - bias_relation),
                p=2,
                dim=2
            ) * 3.5

        return self.gamma.item() - (modulus_score + phase_score)




    
    @staticmethod
    def train_step(model, optimizer, train_iterator, args, step, path_batch=None,
                   path_weight=0.0, consistency_weight=0.0, zero_grad=True,
                   optimizer_step=True, accumulation_steps=1):
        '''
        A single train step. Apply back-propation and return the loss
        '''
        base_model = model.module if hasattr(model, 'module') else model
        model.current_step = step
        model.train()

        if zero_grad:
            optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)

        # if args.negative_adversarial_sampling:
        #     #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
        #     negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
        #                       * F.logsigmoid(-negative_score)).sum(dim = 1)
        # else:
        #     negative_score = F.logsigmoid(-negative_score).mean(dim = 1)


        if args.negative_adversarial_sampling:
            if negative_score.dim() == 1:
                # If squeezed accidentally, reshape
                negative_score = negative_score.unsqueeze(1)

            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                            * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = model(positive_sample)

        positive_score = F.logsigmoid(positive_score)
        subsampling_weight = subsampling_weight.view(-1)

        flat_positive = positive_score.reshape(-1)
        target_len = subsampling_weight.shape[0]
        if target_len > 0 and flat_positive.numel() % target_len == 0:
            positive_score = flat_positive.view(target_len, -1).mean(dim=1)
        else:
            if flat_positive.shape[0] != target_len:
                logging.warning(
                    'Positive score and subsampling weight have different lengths (%d vs %d). Adjusting to match.',
                    flat_positive.shape[0],
                    target_len
                )
            min_len = min(flat_positive.shape[0], target_len)
            positive_score = flat_positive[:min_len]
            subsampling_weight = subsampling_weight[:min_len]



        


        # if args.uni_weight:
        #     positive_sample_loss = - positive_score.mean()
        #     negative_sample_loss = - negative_score.mean()
        # else:
        #     positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
        #     negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            if negative_score.dim() == 1:
                # Negative score already collapsed (single sample), skip subsampling weighting
                positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
                negative_sample_loss = - negative_score.mean()
            else:
                # Expand subsampling weight safely
                subsampling_weight = subsampling_weight.unsqueeze(1).expand_as(negative_score)
                positive_sample_loss = - (subsampling_weight[:, 0] * positive_score).sum() / subsampling_weight[:, 0].sum()
                negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()


        loss = (positive_sample_loss + negative_sample_loss)/2

        path_logs = {}
        if path_batch is not None and path_weight > 0:
            path_heads, path_relations, path_lengths, path_tails, negative_tails = path_batch
            if args.cuda:
                path_heads = path_heads.cuda()
                path_relations = path_relations.cuda()
                path_lengths = path_lengths.cuda()
                path_tails = path_tails.cuda()
                negative_tails = negative_tails.cuda()
            pos_scores = base_model.path_forward(path_heads, path_relations, path_tails).view(path_heads.size(0), 1)
            neg_scores = base_model.path_forward(path_heads, path_relations, negative_tails).view(path_heads.size(0), -1)
            margin = getattr(args, 'path_margin', args.gamma)
            path_loss_tensor = F.relu(neg_scores - pos_scores + margin)
            path_loss_val = path_loss_tensor.mean()
            loss = loss + path_weight * path_loss_val

            path_logs['path_loss'] = path_loss_val.item()

            if consistency_weight > 0:
                last_indices = (path_lengths - 1).clamp(min=0)
                last_rel = path_relations.gather(1, last_indices.unsqueeze(1)).squeeze(1)
                composed_sample = torch.stack([path_heads, last_rel, path_tails], dim=1)
                single_score = model(composed_sample, mode='single').view(path_heads.size(0), 1)
                consistency_margin = getattr(args, 'path_consistency_margin', 1.0)
                consistency_term = F.relu(single_score - pos_scores + consistency_margin)
                consistency_loss = consistency_term.mean()
                loss = loss + consistency_weight * consistency_loss
                path_logs['consistency_loss'] = consistency_loss.item()

        # Apply L3 regularization
        
        # if args.regularization != 0.0:
        #     #Use L3 regularization for ComplEx and DistMult
        #     regularization = args.regularization * (
        #         model.entity_embedding.norm(p = 3)**3 + 
        #         model.relation_embedding.norm(p = 3)**3
        #     )
        #     loss = loss + regularization
        #     regularization_log = {'regularization': regularization.item()}
        # else:
        #     regularization_log = {}

        #Apply L2 regularization

        # if args.regularization != 0.0:
        #     #Use L3 regularization for ComplEx and DistMult
        #     regularization = args.regularization * (
        #         model.entity_embedding.norm(p = 2)**2 + 
        #         model.relation_embedding.norm(p = 2)**2
        #     )
        #     loss = loss + regularization
        #     regularization_log = {'regularization': regularization.item()}
        # else:
        #     regularization_log = {}

        # N3 regularization (sum of |x|^3)
        
        if args.regularization != 0.0:
            # Directly use the already-doubled embeddings
            reg_entity = torch.sum(torch.abs(base_model.entity_embedding) ** 3)
            reg_relation = torch.sum(torch.abs(base_model.relation_embedding) ** 3)

            # Normalize by total number of entities and relations
            reg = (reg_entity / base_model.nentity) + (reg_relation / base_model.nrelation)

            # Add to loss
            loss = loss + args.regularization * reg

            # Logging
            regularization_log = {'n3_regularization': reg.item()}
        else:
            regularization_log = {}





        loss = loss / max(1, accumulation_steps)
        loss.backward()

        if optimizer_step:
            optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }
        if path_logs:
            log.update(path_logs)

        # print modulus and phase weights every 1000 steps
        if base_model.model_name == 'RelatE':
            if step % 1000 == 0:
                logging.info(
                    "Phase Weight (avg): %.4f, Modulus Weight (avg): %.4f",
                    F.softplus(base_model.phase_weight).mean().item(),
                    F.softplus(base_model.modulus_weight).mean().item()
                )




        return log
    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        
        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
            
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            
            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score = model((positive_sample, negative_sample), mode)
                        score += filter_bias

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim = 1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            #Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                                'HITS@20': 1.0 if ranking <= 20 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics
