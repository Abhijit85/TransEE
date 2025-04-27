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
             use_eras=False, k_prototypes=4,type_map_path=None, entity2id=None,type_lambda=1.0):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.type_lambda = type_lambda


        
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
            self.phase_weight = nn.Parameter(torch.ones(self.nrelation, 1))
            self.modulus_weight = nn.Parameter(torch.ones(self.nrelation, 1) * 3.5)

           
        self.use_type_bias = False  # Default

        #  Store the entity2id mapping for lookups
        self.entity2id = entity2id

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
            relation = torch.index_select(self.relation_embedding, dim=0, index=relation_ids).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_ids).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(self.entity_embedding, dim=0, index=head_part.view(-1)).view(batch_size, negative_sample_size, -1)
            relation = torch.index_select(self.relation_embedding, dim=0, index=tail_part[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part[:, 2]).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
            relation = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1)
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

        # Compute scores
        if mode == 'head-batch':
            phase_score = torch.abs(torch.sin((tail_phase - rel_phase - head_phase) / 2)).sum(dim=2, keepdim=True)
            modulus_score = torch.norm(tail_modulus * (1 - bias_relation) - head_modulus * (rel_modulus + bias_relation), p=2, dim=2, keepdim=True)
        elif mode == 'tail-batch':
            phase_score = torch.abs(torch.sin((head_phase + rel_phase - tail_phase) / 2)).sum(dim=2, keepdim=True)
            modulus_score = torch.norm(head_modulus * (rel_modulus + bias_relation) - tail_modulus * (1 - bias_relation), p=2, dim=2, keepdim=True)
        else:  # default
            phase_score = torch.abs(torch.sin((head_phase + rel_phase - tail_phase) / 2)).sum(dim=2, keepdim=True)
            modulus_score = torch.norm(head_modulus * (rel_modulus + bias_relation) - tail_modulus * (1 - bias_relation), p=2, dim=2, keepdim=True)


        # Expand phase/modulus weights if needed
        if phase_score.size(1) != phase_w.size(1):
            phase_w = phase_w.unsqueeze(1).expand_as(phase_score)   # [B, N, 1]
            modulus_w = modulus_w.unsqueeze(1).expand_as(modulus_score)  # [B, N, 1]


        # Apply weighting
        phase_score = phase_score * phase_w
        modulus_score = modulus_score * modulus_w

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
    def train_step(model, optimizer, train_iterator, args,step):
        '''
        A single train step. Apply back-propation and return the loss
        '''
        model.current_step = step
        model.train()

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

        # positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)
        positive_score = F.logsigmoid(positive_score).squeeze()



        


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
            reg_entity = torch.sum(torch.abs(model.entity_embedding) ** 3)
            reg_relation = torch.sum(torch.abs(model.relation_embedding) ** 3)

            # Normalize by total number of entities and relations
            reg = (reg_entity / model.nentity) + (reg_relation / model.nrelation)

            # Add to loss
            loss = loss + args.regularization * reg

            # Logging
            regularization_log = {'n3_regularization': reg.item()}
        else:
            regularization_log = {}





        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        # print modulus and phase weights every 1000 steps
        if model.model_name == 'RelatE':
            if step % 1000 == 0:
                # logging.info(f"Phase Weight: {F.softplus(model.phase_weight).item():.4f}, Modulus Weight: {F.softplus(model.modulus_weight).item():.4f}")
                logging.info(f"Phase Weight (avg): {F.softplus(model.phase_weight).mean().item():.4f}, Modulus Weight (avg): {F.softplus(model.modulus_weight).mean().item():.4f}")




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
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics
