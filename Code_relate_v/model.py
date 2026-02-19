#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import math
import os
from collections import defaultdict
import multiprocessing

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
             modulus_sharpness=1.0, phase_sharpness=1.0, entity_depths=None, relation_behaviors=None,
             teacher_entity_vectors=None, teacher_alignment_mask=None, teacher_entity_dim=None,
             teacher_query_dim=None,
             relation_loss_weights=None, kd_relation_weights=None,
             relation_prompt_embeddings=None, relation_prompt_weight=0.0,
             relation_prompt_warmup_steps=0,
             use_entity_prompt=False, entity_prompt_weight=0.0,
             entity_prompt_warmup_steps=0,
             use_mos_head=False, mos_components=4,
             mos_hidden_dim=256, mos_entropy_weight=0.0, expose_query_embedding=False,
             phase_weight_scale=0.65, use_region_head=False, region_dim=0,
             region_blend_weight=0.5, use_hyper_subspace=False, hyper_dim=0,
             hyper_blend_warmup_steps=0,
             use_hyperbolic=False, hyperbolic_c=1.0,
             use_relation_gate=False, use_type_mod_norm=False,
             use_hierarchy_mod_head=False,
             plm_entity_vectors=None, plm_relation_vectors=None,
             entity_concepts=None, concept_depths=None,
             use_query_adaptive=False, qa_hidden_dim=256, qa_num_experts=4,
             qa_temperature_floor=0.5, qa_temperature_ceiling=2.5,
             qa_prototype_weight=0.0,
             cc_concept_weight=0.0, cc_depth_weight=0.0, cc_relation_weight=0.0):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.type_lambda = type_lambda
        self.modulus_sharpness = modulus_sharpness
        self.phase_sharpness = phase_sharpness
        self.plm_entity_proj = None
        self.plm_relation_proj = None


        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.is_relate_family = model_name in {'RelatE', 'RelateV', 'CCRelatE', 'BKRelatE', 'ARelatE'}
        self.is_ccmurp = bool(model_name == 'CCMuRP')
        self.is_murp = bool(model_name == 'MuRP')
        self.use_hyperbolic = use_hyperbolic and self.is_relate_family
        if self.use_hyperbolic and (double_entity_embedding or double_relation_embedding):
            raise ValueError('Hyperbolic RelatE does not support double embeddings.')
        self.use_relation_gate = bool(use_relation_gate and self.is_relate_family and not self.use_hyperbolic)
        self.use_type_mod_norm = bool(use_type_mod_norm)
        self.use_hierarchy_mod_head = bool(use_hierarchy_mod_head and self.is_relate_family and not self.use_hyperbolic)
        self.hyperbolic_c = float(max(hyperbolic_c, 1e-4))
        self.hyper_eps = 1e-5
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        self.use_hyper_subspace = bool(use_hyper_subspace and self.is_relate_family and hyper_dim > 0)
        self.hyper_dim = int(hyper_dim if self.use_hyper_subspace else 0)

        #Slope-Weighted L1 Versions 
        # self.rel_width = nn.Parameter(torch.ones(nrelation, self.relation_dim // 2))
        self.rel_width = nn.Parameter(torch.full((nrelation, self.relation_dim // 2), init_rel_width))





        # Debugging

        # print(f"Entity Embedding Dimension: {self.entity_dim}")
        # print(f"Relation Embedding Dimension: {self.relation_dim}")


        # ERAS variant
        self.use_eras = use_eras and not self.use_hyperbolic
        self.k_prototypes = k_prototypes

        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        if self.use_hyper_subspace:
            self.entity_embedding_hyper = nn.Parameter(
                torch.zeros(nentity, self.hyper_dim)
            )
            nn.init.uniform_(
                tensor=self.entity_embedding_hyper,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
        else:
            self.entity_embedding_hyper = None
        
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        if self.use_hyper_subspace:
            self.relation_hyper_shift = nn.Parameter(torch.zeros(nrelation, self.hyper_dim))
            self.relation_hyper_scale = nn.Parameter(torch.ones(nrelation, self.hyper_dim))
        else:
            self.relation_hyper_shift = None
            self.relation_hyper_scale = None
        if self.use_hyperbolic:
            self._project_in_place(self.entity_embedding)
            self._project_in_place(self.relation_embedding)

        if plm_entity_vectors is not None:
            plm_ent = plm_entity_vectors if isinstance(plm_entity_vectors, torch.Tensor) else torch.tensor(plm_entity_vectors, dtype=torch.float32)
            if hasattr(self, 'plm_entity_vectors'):
                delattr(self, 'plm_entity_vectors')
            self.register_buffer('plm_entity_vectors', plm_ent)
            self.plm_entity_proj = nn.Linear(plm_ent.size(1), self.entity_dim, bias=False)
        else:
            self.plm_entity_vectors = None
        if plm_relation_vectors is not None:
            plm_rel = plm_relation_vectors if isinstance(plm_relation_vectors, torch.Tensor) else torch.tensor(plm_relation_vectors, dtype=torch.float32)
            if hasattr(self, 'plm_relation_vectors'):
                delattr(self, 'plm_relation_vectors')
            self.register_buffer('plm_relation_vectors', plm_rel)
            self.plm_relation_proj = nn.Linear(plm_rel.size(1), self.relation_dim, bias=False)
        else:
            self.plm_relation_vectors = None

        # Learnable weights for RelatE score decomposition with Per-Relation Tensors
        # as vectors per-relation
        if self.is_relate_family and not self.use_hyperbolic:
            # self.phase_weight = nn.Parameter(torch.Tensor([1.0]))
            # self.modulus_weight = nn.Parameter(torch.Tensor([3.5]))
            phase_init = init_modulus_weight * phase_weight_scale
            self.phase_weight = nn.Parameter(torch.ones(self.nrelation, 1) * phase_init)
            self.modulus_weight = nn.Parameter(torch.ones(self.nrelation, 1) * init_modulus_weight)
            self.phase_harmonics = max(1, phase_harmonics)
            self.phase_freq_param = nn.Parameter(torch.ones(self.nrelation, self.phase_harmonics))
            if self.use_relation_gate:
                self.relation_gate = nn.Parameter(torch.zeros(self.nrelation, 1))
            else:
                self.relation_gate = None
            if self.use_hierarchy_mod_head:
                self.hierarchy_mod_scale = nn.Parameter(torch.zeros(self.nrelation, 1))
            else:
                self.hierarchy_mod_scale = None
            # Dataset-adaptive scorer: learned per training run (hence per dataset run).
            # Initialize to scale=1.0 so old behavior is preserved at step 0.
            softplus_inv_one = math.log(math.exp(1.0) - 1.0)
            self.dataset_phase_scale_logit = nn.Parameter(torch.tensor([softplus_inv_one], dtype=torch.float32))
            self.dataset_modulus_scale_logit = nn.Parameter(torch.tensor([softplus_inv_one], dtype=torch.float32))
        else:
            self.phase_harmonics = 1
            self.phase_freq_param = None
            self.phase_weight = nn.Parameter(torch.ones(self.nrelation, 1), requires_grad=False)
            self.modulus_weight = nn.Parameter(torch.ones(self.nrelation, 1), requires_grad=False)
            self.relation_gate = None
            self.hierarchy_mod_scale = None
            self.dataset_phase_scale_logit = None
            self.dataset_modulus_scale_logit = None

           
        self.use_type_bias = False  # Default
        self.base_nrelation = 0
        self.type_mod_scale = None

        #  Store the entity2id mapping for lookups
        self.entity2id = entity2id
        self.tie_inverses = False
        if teacher_entity_vectors is not None and teacher_entity_dim:
            vec_tensor = teacher_entity_vectors if isinstance(teacher_entity_vectors, torch.Tensor) else torch.tensor(teacher_entity_vectors, dtype=torch.float32)
            self.register_buffer('teacher_entity_vectors', vec_tensor)
            if teacher_alignment_mask is not None:
                mask_tensor = teacher_alignment_mask if isinstance(teacher_alignment_mask, torch.Tensor) else torch.tensor(teacher_alignment_mask, dtype=torch.bool)
            else:
                mask_tensor = torch.ones(vec_tensor.size(0), dtype=torch.bool)
            self.register_buffer('teacher_alignment_mask', mask_tensor)
            proj_input_dim = self.entity_dim + (self.hyper_dim if self.use_hyper_subspace else 0)
            self.teacher_projector = nn.Linear(proj_input_dim, teacher_entity_dim, bias=False)
        else:
            self.teacher_entity_vectors = None
            self.teacher_alignment_mask = None
            self.teacher_projector = None
        if teacher_query_dim:
            self.teacher_query_projector = nn.Linear(self.entity_dim, teacher_query_dim, bias=False)
        else:
            self.teacher_query_projector = None
        self.relation_loss_weights = None
        if kd_relation_weights is not None:
            if isinstance(kd_relation_weights, torch.Tensor):
                kd_tensor = kd_relation_weights.float()
            else:
                kd_tensor = torch.tensor(kd_relation_weights, dtype=torch.float32)
            self.register_buffer('kd_relation_weights', kd_tensor.view(-1))
        else:
            self.kd_relation_weights = None
        self.relation_prompt_proj = None
        self.relation_prompt_gates = None
        self.relation_prompt_warmup_steps = int(max(0, relation_prompt_warmup_steps))
        if relation_prompt_embeddings is not None:
            prompt_tensor = relation_prompt_embeddings
            if not isinstance(prompt_tensor, torch.Tensor):
                prompt_tensor = torch.tensor(prompt_tensor, dtype=torch.float32)
            if prompt_tensor.size(0) != nrelation:
                raise ValueError('Relation prompt tensor must match nrelation (got %d vs %d)' %
                                 (prompt_tensor.size(0), nrelation))
            self.register_buffer('relation_prompt_embeddings', prompt_tensor)
            prompt_dim = prompt_tensor.size(1)
            self.relation_prompt_proj = nn.Linear(prompt_dim, self.relation_dim, bias=False)
            init_weight = float(relation_prompt_weight)
            if init_weight <= 0.0:
                init_logit = -10.0
            elif init_weight >= 1.0:
                init_logit = 10.0
            else:
                init_logit = math.log(init_weight / (1.0 - init_weight))
            self.relation_prompt_gates = nn.Parameter(torch.full((nrelation, 1), init_logit))
        else:
            self.relation_prompt_embeddings = None
        self.use_entity_prompt = bool(use_entity_prompt and teacher_entity_vectors is not None and teacher_entity_dim)
        self.entity_prompt_proj = None
        self.entity_prompt_gates = None
        self.entity_prompt_warmup_steps = int(max(0, entity_prompt_warmup_steps))
        if self.use_entity_prompt:
            self.entity_prompt_proj = nn.Linear(teacher_entity_dim, self.entity_dim, bias=False)
            init_weight = float(entity_prompt_weight)
            if init_weight <= 0.0:
                init_logit = -10.0
            elif init_weight >= 1.0:
                init_logit = 10.0
            else:
                init_logit = math.log(init_weight / (1.0 - init_weight))
            self.entity_prompt_gates = nn.Parameter(torch.full((nentity, 1), init_logit))
        else:
            self.use_entity_prompt = False
        self.current_step = None
        # Disable region head when running hyperbolic RelatE until support is added.
        if self.use_hyperbolic:
            use_region_head = False
        self.use_region_head = use_region_head
        self.region_blend_weight = region_blend_weight
        if self.use_region_head:
            self.region_dim = max(1, region_dim)
            self.entity_region_center = nn.Parameter(torch.empty(nentity, self.region_dim))
            self.entity_region_extent = nn.Parameter(torch.empty(nentity, self.region_dim))
            nn.init.xavier_uniform_(self.entity_region_center)
            nn.init.zeros_(self.entity_region_extent)
            self.relation_region_shift = nn.Parameter(torch.zeros(nrelation, self.region_dim))
            self.relation_region_scale = nn.Parameter(torch.zeros(nrelation, self.region_dim))
        else:
            self.region_dim = 0
            self.entity_region_center = None
            self.entity_region_extent = None
            self.relation_region_shift = None
            self.relation_region_scale = None
        if entity_depths:
            depth_tensor = torch.full((nentity,), -1.0, dtype=torch.float)
            for idx, depth in entity_depths.items():
                if 0 <= idx < nentity:
                    depth_tensor[idx] = float(depth)
            self.register_buffer('entity_depths', depth_tensor)
        else:
            self.register_buffer('entity_depths', None)
        if entity_concepts:
            concept_tensor = torch.full((nentity,), -1, dtype=torch.long)
            for idx, concept_id in entity_concepts.items():
                if 0 <= idx < nentity:
                    concept_tensor[idx] = int(concept_id)
            self.register_buffer('entity_concepts', concept_tensor)
        else:
            self.register_buffer('entity_concepts', None)
        if concept_depths:
            max_idx = max(int(k) for k in concept_depths.keys())
            concept_depth_tensor = torch.full((max_idx + 1,), -1.0, dtype=torch.float)
            for concept_id, depth in concept_depths.items():
                cid = int(concept_id)
                if cid >= 0:
                    concept_depth_tensor[cid] = float(depth)
            self.register_buffer('concept_depths', concept_depth_tensor)
        else:
            self.register_buffer('concept_depths', None)
        rel_behaviors = relation_behaviors or {}
        sym_mask = torch.zeros(nrelation, dtype=torch.bool)
        for idx in rel_behaviors.get('symmetric', []):
            if 0 <= idx < nrelation:
                sym_mask[idx] = True
        anti_mask = torch.zeros(nrelation, dtype=torch.bool)
        for idx in rel_behaviors.get('antisymmetric', []):
            if 0 <= idx < nrelation:
                anti_mask[idx] = True
        hier_mask = torch.zeros(nrelation, dtype=torch.bool)
        for idx in rel_behaviors.get('hierarchical', []):
            if 0 <= idx < nrelation:
                hier_mask[idx] = True
        mero_mask = torch.zeros(nrelation, dtype=torch.bool)
        for idx in rel_behaviors.get('meronymic', []):
            if 0 <= idx < nrelation:
                mero_mask[idx] = True
        inverse_pairs = torch.zeros((0, 2), dtype=torch.long)
        pair_list = rel_behaviors.get('inverse_pairs', [])
        if pair_list:
            tensor_list = []
            for lhs, rhs in pair_list:
                if 0 <= lhs < nrelation and 0 <= rhs < nrelation:
                    tensor_list.append([lhs, rhs])
            if tensor_list:
                inverse_pairs = torch.tensor(tensor_list, dtype=torch.long)
        self.register_buffer('symmetric_rel_mask', sym_mask)
        self.register_buffer('antisymmetric_rel_mask', anti_mask)
        self.register_buffer('hierarchical_rel_mask', hier_mask)
        self.register_buffer('meronymic_rel_mask', mero_mask)
        self.register_buffer('inverse_rel_pairs', inverse_pairs)
        comp_pairs = rel_behaviors.get('compositions', [])
        comp_tensor = torch.zeros((0, 3), dtype=torch.long)
        if comp_pairs:
            temp = []
            for r1, r2, r3 in comp_pairs:
                if 0 <= r1 < nrelation and 0 <= r2 < nrelation and 0 <= r3 < nrelation:
                    temp.append([r1, r2, r3])
            if temp:
                comp_tensor = torch.tensor(temp, dtype=torch.long)
        self.register_buffer('composition_rel_triples', comp_tensor, persistent=False)
        if self.use_region_head:
            region_mask = hier_mask | mero_mask
        else:
            region_mask = torch.zeros_like(hier_mask)
        self.register_buffer('region_rel_mask', region_mask)
        if self.use_hyper_subspace:
            blend_init = torch.full((nrelation, 1), 0.2)
            blend_init[self.hierarchical_rel_mask] = 0.5
            logits = torch.log(blend_init / (1 - blend_init))
            self.hyper_blend_logits = nn.Parameter(logits)
            self.hyper_blend_warmup_steps = hyper_blend_warmup_steps
        else:
            self.hyper_blend_logits = None
            self.hyper_blend_warmup_steps = 0
        self.use_bk_relate = bool(model_name == 'BKRelatE' and self.use_hyper_subspace)
        self.bk_eps = 1e-5
        if self.use_bk_relate:
            bk_init = torch.full((nrelation, 1), 0.1)
            bk_focus = self.hierarchical_rel_mask | self.meronymic_rel_mask
            bk_init[bk_focus] = 0.35
            bk_logits = torch.log(bk_init / (1 - bk_init))
            self.bk_relation_blend_logit = nn.Parameter(bk_logits)
        else:
            self.bk_relation_blend_logit = None
        self.use_arel_relate = bool(model_name == 'ARelatE' and self.is_relate_family)
        if self.use_arel_relate:
            gate_init = torch.full((nrelation, 1), 0.15)
            gate_focus = self.hierarchical_rel_mask | self.meronymic_rel_mask
            gate_init[gate_focus] = 0.3
            gate_logits = torch.log(gate_init / (1 - gate_init))
            self.arel_gate_logit = nn.Parameter(gate_logits)
            softplus_inv_one = math.log(math.exp(1.0) - 1.0)
            self.arel_temp_logit = nn.Parameter(torch.full((nrelation, 1), softplus_inv_one))
            self.arel_bias = nn.Parameter(torch.zeros(nrelation, 1))
        else:
            self.arel_gate_logit = None
            self.arel_temp_logit = None
            self.arel_bias = None

        # Query-adaptive scorer (optional)
        self.use_query_adaptive = bool(use_query_adaptive and self.is_relate_family and not self.use_hyperbolic)
        self.qa_prototype_weight = float(max(0.0, qa_prototype_weight))
        self.qa_temperature_floor = float(qa_temperature_floor)
        self.qa_temperature_ceiling = float(qa_temperature_ceiling)
        if self.use_query_adaptive:
            qa_input_dim = self.entity_dim + self.relation_dim
            qa_hidden = int(max(32, qa_hidden_dim))
            self.qa_num_experts = int(max(1, qa_num_experts))
            self.qa_gate_mlp = nn.Sequential(
                nn.Linear(qa_input_dim, qa_hidden),
                nn.ReLU(),
                nn.Linear(qa_hidden, self.qa_num_experts + 2)  # experts + alpha + temperature
            )
            # Expert deltas over [phase_scale, modulus_scale]
            self.qa_expert_delta = nn.Parameter(torch.zeros(self.qa_num_experts, 2))

            # Relation-aware prior over experts (encourages specialization by relation family)
            expert_bias = torch.zeros(nrelation, self.qa_num_experts)
            if self.qa_num_experts >= 2:
                expert_bias[self.hierarchical_rel_mask, 1] = 1.0
            if self.qa_num_experts >= 3:
                expert_bias[self.meronymic_rel_mask, 2] = 1.0
            if self.qa_num_experts >= 4:
                expert_bias[self.symmetric_rel_mask, 3] = 0.5
            self.register_buffer('qa_relation_expert_bias', expert_bias)

            # Optional concept prototype memory
            if self.entity_concepts is not None and self.entity_concepts.numel() > 0:
                valid_ids = self.entity_concepts[self.entity_concepts >= 0]
                if valid_ids.numel() > 0:
                    nconcept = int(valid_ids.max().item()) + 1
                    self.concept_prototypes = nn.Parameter(torch.zeros(nconcept, self.entity_dim))
                    nn.init.uniform_(
                        tensor=self.concept_prototypes,
                        a=-self.embedding_range.item(),
                        b=self.embedding_range.item()
                    )
                else:
                    self.concept_prototypes = None
            else:
                self.concept_prototypes = None

            # Contrastive projection heads
            self.qa_query_proj = nn.Linear(self.entity_dim, self.entity_dim, bias=False)
            self.qa_entity_proj = nn.Linear(self.entity_dim, self.entity_dim, bias=False)
        else:
            self.qa_num_experts = 0
            self.qa_gate_mlp = None
            self.qa_expert_delta = None
            self.qa_relation_expert_bias = None
            self.concept_prototypes = None
            self.qa_query_proj = None
            self.qa_entity_proj = None

        # Concept-Calibrated RelatE (standalone variant)
        self.use_cc_relate = bool(model_name in {'CCRelatE', 'CCMuRP'})
        self.cc_concept_weight = float(max(0.0, cc_concept_weight))
        self.cc_depth_weight = float(max(0.0, cc_depth_weight))
        self.cc_relation_weight = float(max(0.0, cc_relation_weight))
        if self.use_cc_relate and self.entity_concepts is not None and self.entity_concepts.numel() > 0:
            valid_ids = self.entity_concepts[self.entity_concepts >= 0]
            if valid_ids.numel() > 0:
                nconcept = int(valid_ids.max().item()) + 1
                self.cc_concept_prototypes = nn.Parameter(torch.zeros(nconcept, self.entity_dim))
                nn.init.uniform_(
                    tensor=self.cc_concept_prototypes,
                    a=-self.embedding_range.item(),
                    b=self.embedding_range.item()
                )
                self.cc_relation_concept_shift = nn.Parameter(torch.zeros(nrelation, self.entity_dim))
                self.cc_relation_gate = nn.Parameter(torch.zeros(nrelation, 1))
                self.cc_depth_scale = nn.Parameter(torch.zeros(nrelation, 1))
            else:
                self.cc_concept_prototypes = None
                self.cc_relation_concept_shift = None
                self.cc_relation_gate = None
                self.cc_depth_scale = None
        else:
            self.cc_concept_prototypes = None
            self.cc_relation_concept_shift = None
            self.cc_relation_gate = None
            self.cc_depth_scale = None
        if self.is_ccmurp and self.relation_dim != self.entity_dim:
            self.ccmurp_relation_proj = nn.Linear(self.relation_dim, self.entity_dim, bias=False)
        else:
            self.ccmurp_relation_proj = None

        # Query embeddings are required when the MoS head is enabled
        self.expose_query_embedding = expose_query_embedding or use_mos_head or self.use_query_adaptive
        self.last_query_embedding = None
        self.latest_gate_entropy = None

        # MoS head parameters
        self.use_mos_head = use_mos_head
        self.mos_entropy_weight = mos_entropy_weight
        if self.use_mos_head:
            mos_input_dim = self.entity_dim
            self.mos_components = mos_components
            self.mos_context = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(mos_input_dim, mos_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(mos_hidden_dim, mos_input_dim)
                )
                for _ in range(mos_components)
            ])
            self.mos_gate = nn.Linear(mos_input_dim, mos_components)
        else:
            self.mos_gate = None
            self.mos_components = 0

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
            if self.use_type_mod_norm:
                self.type_mod_scale = nn.Embedding(len(self.type_to_id), 1)
                nn.init.zeros_(self.type_mod_scale.weight)
            else:
                self.type_mod_scale = None





        '''
        Creating protoptype and attention for ERAS search
        ERAS: Only initialize if requested for RelatE
        '''
        if self.is_relate_family and use_eras:
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
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE', 'RelatE', 'RelateV', 'CCRelatE', 'BKRelatE', 'ARelatE', 'CCMuRP', 'MuRP']:
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
        if (self.relation_prompt_embeddings is not None and
                self.relation_prompt_proj is not None and
                self.relation_prompt_gates is not None):
            prompt = torch.index_select(
                self.relation_prompt_embeddings,
                dim=0,
                index=relation_ids.view(-1)
            ).view(*relation_ids.shape, -1)
            proj = self.relation_prompt_proj(prompt)
            gate_logits = torch.index_select(
                self.relation_prompt_gates,
                dim=0,
                index=relation_ids.view(-1)
            ).view(*relation_ids.shape, 1)
            gate = torch.sigmoid(gate_logits)
            if self.relation_prompt_warmup_steps > 0 and getattr(self, 'current_step', None) is not None:
                scale = min(1.0, max(0.0, self.current_step) / self.relation_prompt_warmup_steps)
                gate = gate * scale
            rel = rel + gate * proj
        return rel

    def _lookup_entity_prompt(self, entity_ids, device):
        if not self.use_entity_prompt or self.entity_prompt_proj is None or self.entity_prompt_gates is None:
            return None
        teacher_vecs = torch.index_select(
            self.teacher_entity_vectors,
            dim=0,
            index=entity_ids
        ).to(device)
        projected = self.entity_prompt_proj(teacher_vecs)
        gate = torch.sigmoid(
            torch.index_select(
                self.entity_prompt_gates,
                dim=0,
                index=entity_ids
            ).to(device)
        )
        if self.entity_prompt_warmup_steps > 0 and getattr(self, 'current_step', None) is not None:
            scale = min(1.0, max(0.0, self.current_step) / self.entity_prompt_warmup_steps)
            gate = gate * scale
        if self.teacher_alignment_mask is not None:
            mask_vals = self.teacher_alignment_mask.index_select(0, entity_ids).unsqueeze(-1).to(device).float()
            gate = gate * mask_vals
        return projected * gate

    def _apply_entity_prompt(self, tensor, entity_ids, shape):
        addition = self._lookup_entity_prompt(entity_ids, tensor.device)
        if addition is None:
            return tensor
        addition = addition.view(*shape, -1)
        return tensor + addition

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

    def _compose_query_embedding(self, phase_arg, mod_core):
        phase_flat = phase_arg.reshape(phase_arg.size(0), phase_arg.size(1), -1)
        mod_flat = mod_core.reshape(mod_core.size(0), mod_core.size(1), -1)
        return torch.cat([phase_flat, mod_flat], dim=-1)

    def _apply_query_adaptive_scoring(self, phase_score, modulus_score, query_embedding, relation, relation_ids):
        if (not self.use_query_adaptive or query_embedding is None or
                self.qa_gate_mlp is None or self.qa_expert_delta is None):
            return phase_score, modulus_score, None

        # Use one query per sample for gating, expand to all candidates later.
        query_base = query_embedding[:, 0, :]
        rel_base = relation[:, 0, :]
        qa_input = torch.cat([query_base, rel_base], dim=-1)
        gate_out = self.qa_gate_mlp(qa_input)
        expert_logits = gate_out[:, :self.qa_num_experts]
        alpha_logit = gate_out[:, self.qa_num_experts:self.qa_num_experts + 1]
        temp_logit = gate_out[:, self.qa_num_experts + 1:self.qa_num_experts + 2]

        if self.qa_relation_expert_bias is not None:
            expert_logits = expert_logits + self.qa_relation_expert_bias.index_select(0, relation_ids)

        expert_w = F.softmax(expert_logits, dim=-1)
        delta = torch.matmul(expert_w, self.qa_expert_delta)
        phase_scale = F.softplus(1.0 + delta[:, 0:1]).view(-1, 1, 1)
        mod_scale = F.softplus(1.0 + delta[:, 1:2]).view(-1, 1, 1)

        phase_adj = phase_score * phase_scale
        mod_adj = modulus_score * mod_scale

        alpha = torch.sigmoid(alpha_logit).view(-1, 1, 1)
        t_floor = min(self.qa_temperature_floor, self.qa_temperature_ceiling)
        t_ceil = max(self.qa_temperature_floor, self.qa_temperature_ceiling)
        temp = t_floor + (t_ceil - t_floor) * torch.sigmoid(temp_logit).view(-1, 1, 1)

        return alpha * phase_adj / temp, (1.0 - alpha) * mod_adj / temp, query_base

    def _compute_mos_log_probs(self, query_embedding, candidate_embeddings):
        if query_embedding is None or not self.use_mos_head:
            return None, None

        if query_embedding.size(1) == 0:
            return None, None

        base_query = query_embedding[:, 0, :]
        comp_logits = []
        for proj in self.mos_context:
            context = proj(base_query)  # [B, D]
            # candidate_embeddings: [B, N, D]
            logits = torch.einsum('bd,bnd->bn', context, candidate_embeddings)
            comp_logits.append(logits)

        comp_logits = torch.stack(comp_logits, dim=1)  # [B, K, N]
        comp_log_probs = F.log_softmax(comp_logits, dim=2)

        gate_logits = self.mos_gate(base_query)
        gate_log_probs = F.log_softmax(gate_logits, dim=1).unsqueeze(-1)  # [B, K, 1]
        gate_probs = gate_log_probs.exp()

        mixed_log_probs = torch.logsumexp(gate_log_probs + comp_log_probs, dim=1)
        entropy = -(gate_probs * gate_log_probs).sum(dim=1).mean()

        return mixed_log_probs, entropy

    # --- Hyperbolic helpers (stubs for future hybrid scorer) ---
    def _hyper_log_map(self, tensor):
        norm = torch.norm(tensor, dim=-1, keepdim=True).clamp(min=1e-10, max=1 - 1e-5)
        return 0.5 * torch.log((1 + norm) / (1 - norm)) * tensor / norm

    def _hyper_exp_map(self, tensor):
        norm = torch.norm(tensor, dim=-1, keepdim=True).clamp(min=1e-10)
        return torch.tanh(norm) * tensor / norm

    def _hyper_mobius_add(self, x, y):
        sqx = torch.sum(x * x, dim=-1, keepdim=True).clamp(max=1 - 1e-5)
        sqy = torch.sum(y * y, dim=-1, keepdim=True).clamp(max=1 - 1e-5)
        dot = torch.sum(x * y, dim=-1, keepdim=True)
        numerator = (1 + 2 * dot + sqy) * x + (1 - sqx) * y
        denominator = 1 + 2 * dot + sqx * sqy
        return numerator / denominator.clamp(min=1e-5)

    def _hyper_neg(self, x):
        return -x

    def _compute_hyper_score(self, head_hyper, relation_ids, tail_hyper):
        if head_hyper is None or tail_hyper is None or self.relation_hyper_shift is None:
            return None
        rel_shift = self.relation_hyper_shift.index_select(0, relation_ids)
        rel_scale = self.relation_hyper_scale.index_select(0, relation_ids)
        if head_hyper.size(1) != tail_hyper.size(1):
            if head_hyper.size(1) == 1:
                head_hyper = head_hyper.expand_as(tail_hyper)
            else:
                tail_hyper = tail_hyper.expand_as(head_hyper)
        if head_hyper.size(1) != 1:
            rel_shift = rel_shift.unsqueeze(1).expand(-1, head_hyper.size(1), -1)
            rel_scale = rel_scale.unsqueeze(1).expand(-1, head_hyper.size(1), -1)
        else:
            rel_shift = rel_shift.unsqueeze(1)
            rel_scale = rel_scale.unsqueeze(1)
        head_log = self._hyper_log_map(head_hyper)
        head_scaled = head_log * rel_scale
        head_trans = self._hyper_exp_map(head_scaled)
        tail_shifted = self._hyper_mobius_add(tail_hyper, rel_shift)
        diff = self._hyper_mobius_add(self._hyper_neg(head_trans), tail_shifted)
        norm = torch.norm(diff, dim=-1).clamp(min=1e-10, max=1 - 1e-5)
        dist = (2 * torch.atanh(norm)) ** 2
        return -dist

    def _project_klein(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.bk_eps)
        max_norm = 1.0 - self.bk_eps
        scale = (max_norm / norm).clamp(max=1.0)
        return x * scale

    def _klein_distance(self, x, y):
        x = self._project_klein(x)
        y = self._project_klein(y)
        x2 = torch.sum(x * x, dim=-1).clamp(max=1.0 - self.bk_eps)
        y2 = torch.sum(y * y, dim=-1).clamp(max=1.0 - self.bk_eps)
        xy = torch.sum(x * y, dim=-1)
        denom = torch.sqrt((1.0 - x2) * (1.0 - y2)).clamp(min=self.bk_eps)
        arg = ((1.0 - xy) / denom).clamp(min=1.0 + self.bk_eps)
        return torch.acosh(arg)

    def _compute_klein_score(self, head_hyper, relation_ids, tail_hyper, mode):
        if (head_hyper is None or tail_hyper is None or relation_ids is None or
                self.relation_hyper_shift is None or self.relation_hyper_scale is None):
            return None

        if head_hyper.size(1) != tail_hyper.size(1):
            if head_hyper.size(1) == 1:
                head_hyper = head_hyper.expand_as(tail_hyper)
            else:
                tail_hyper = tail_hyper.expand_as(head_hyper)

        rel_shift = self.relation_hyper_shift.index_select(0, relation_ids).unsqueeze(1)
        rel_scale = F.softplus(self.relation_hyper_scale.index_select(0, relation_ids)).unsqueeze(1) + self.bk_eps
        if head_hyper.size(1) != 1:
            rel_shift = rel_shift.expand(-1, head_hyper.size(1), -1)
            rel_scale = rel_scale.expand(-1, head_hyper.size(1), -1)

        head_k = self._project_klein(head_hyper)
        tail_k = self._project_klein(tail_hyper)
        if mode == 'head-batch':
            query_k = (tail_k - rel_shift) / rel_scale
            cand_k = head_k
        else:
            query_k = head_k * rel_scale + rel_shift
            cand_k = tail_k
        query_k = self._project_klein(query_k)
        cand_k = self._project_klein(cand_k)
        dist = self._klein_distance(query_k, cand_k)
        return self.gamma.item() - dist

    def _project_to_ball(self, x):
        c = self.hyperbolic_c
        norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.hyper_eps)
        max_norm = (1.0 - self.hyper_eps) / math.sqrt(c)
        cond = norm > max_norm
        if cond.any():
            scale = max_norm / norm
            x = torch.where(cond, x * scale, x)
        return x

    def _project_in_place(self, param):
        with torch.no_grad():
            param.data = self._project_to_ball(param.data)

    def _mobius_add(self, x, y):
        c = self.hyperbolic_c
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denominator = 1 + 2 * c * xy + c * c * x2 * y2
        return numerator / denominator.clamp(min=self.hyper_eps)

    def _mobius_neg(self, x):
        return -x

    def _poincare_distance(self, x, y):
        c = self.hyperbolic_c
        x_norm = torch.clamp(c * torch.sum(x * x, dim=-1, keepdim=True), max=1 - self.hyper_eps)
        y_norm = torch.clamp(c * torch.sum(y * y, dim=-1, keepdim=True), max=1 - self.hyper_eps)
        diff = self._mobius_add(self._mobius_neg(x), y)
        diff_norm = torch.clamp(torch.norm(diff, dim=-1), max=(1 - self.hyper_eps) / math.sqrt(c))
        # Using arcosh formulation via atanh
        sqrt_c = math.sqrt(c)
        z = torch.clamp(sqrt_c * diff_norm, max=1 - self.hyper_eps)
        atanh_z = 0.5 * torch.log((1 + z) / (1 - z))
        dist = 2.0 / sqrt_c * atanh_z
        return dist

    def _compute_region_score(self, head_center, head_width, relation_ids,
                              tail_center, tail_width, mode):
        if (head_center is None or head_width is None
                or tail_center is None or tail_width is None
                or self.relation_region_shift is None):
            return None

        head_extent = F.softplus(head_width)
        tail_points = tail_center

        rel_shift = torch.index_select(
            self.relation_region_shift,
            dim=0,
            index=relation_ids
        )
        rel_scale = F.softplus(torch.index_select(
            self.relation_region_scale,
            dim=0,
            index=relation_ids
        ))

        rel_shift = rel_shift.unsqueeze(1)
        rel_scale = rel_scale.unsqueeze(1)

        query_center = head_center + rel_shift
        query_width = head_extent + rel_scale

        query_count = query_center.size(1)
        tail_count = tail_points.size(1)
        if query_count > tail_count:
            tail_points = tail_points.expand(-1, query_count, -1)
        elif tail_count > query_count:
            query_center = query_center.expand(-1, tail_count, -1)
            query_width = query_width.expand(-1, tail_count, -1)

        lower = query_center - query_width
        upper = query_center + query_width
        violation = F.relu(lower - tail_points) + F.relu(tail_points - upper)
        penalty = violation.sum(dim=2)
        return -penalty

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
        head_region_center = None
        head_region_width = None
        tail_region_center = None
        tail_region_width = None
        head_hyper_embed = None
        tail_hyper_embed = None
        candidate_ids = None

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
            head = self._apply_entity_prompt(head, head_ids, (batch_size, 1))
            tail = self._apply_entity_prompt(tail, tail_ids, (batch_size, 1))
            if self.use_region_head:
                head_region_center = torch.index_select(self.entity_region_center, 0, head_ids).unsqueeze(1)
                head_region_width = torch.index_select(self.entity_region_extent, 0, head_ids).unsqueeze(1)
                tail_region_center = torch.index_select(self.entity_region_center, 0, tail_ids).unsqueeze(1)
                tail_region_width = torch.index_select(self.entity_region_extent, 0, tail_ids).unsqueeze(1)
            if self.use_hyper_subspace:
                head_hyper_embed = torch.index_select(self.entity_embedding_hyper, 0, head_ids).unsqueeze(1)
                tail_hyper_embed = torch.index_select(self.entity_embedding_hyper, 0, tail_ids).unsqueeze(1)
            candidate_ids = tail_ids.unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(self.entity_embedding, dim=0, index=head_part.reshape(-1)).view(batch_size, negative_sample_size, -1)
            relation = self._lookup_relation_embedding(tail_part[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part[:, 2]).unsqueeze(1)
            head = self._apply_entity_prompt(head, head_part.reshape(-1), (batch_size, negative_sample_size))
            tail = self._apply_entity_prompt(tail, tail_part[:, 2], (batch_size, 1))
            if self.use_region_head:
                head_region_center = torch.index_select(self.entity_region_center, 0, head_part.reshape(-1)).view(batch_size, negative_sample_size, -1)
                head_region_width = torch.index_select(self.entity_region_extent, 0, head_part.reshape(-1)).view(batch_size, negative_sample_size, -1)
                tail_region_center = torch.index_select(self.entity_region_center, 0, tail_part[:, 2]).unsqueeze(1)
                tail_region_width = torch.index_select(self.entity_region_extent, 0, tail_part[:, 2]).unsqueeze(1)
            if self.use_hyper_subspace:
                head_hyper_embed = torch.index_select(self.entity_embedding_hyper, 0, head_part.reshape(-1)).view(batch_size, negative_sample_size, -1)
                tail_hyper_embed = torch.index_select(self.entity_embedding_hyper, 0, tail_part[:, 2]).unsqueeze(1)
            candidate_ids = head_part

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
            relation = self._lookup_relation_embedding(head_part[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.reshape(-1)).view(batch_size, negative_sample_size, -1)
            head = self._apply_entity_prompt(head, head_part[:, 0], (batch_size, 1))
            tail = self._apply_entity_prompt(tail, tail_part.reshape(-1), (batch_size, negative_sample_size))
            if self.use_region_head:
                head_region_center = torch.index_select(self.entity_region_center, 0, head_part[:, 0]).unsqueeze(1)
                head_region_width = torch.index_select(self.entity_region_extent, 0, head_part[:, 0]).unsqueeze(1)
                tail_region_center = torch.index_select(self.entity_region_center, 0, tail_part.reshape(-1)).view(batch_size, negative_sample_size, -1)
                tail_region_width = torch.index_select(self.entity_region_extent, 0, tail_part.reshape(-1)).view(batch_size, negative_sample_size, -1)
            if self.use_hyper_subspace:
                head_hyper_embed = torch.index_select(self.entity_embedding_hyper, 0, head_part[:, 0]).unsqueeze(1)
                tail_hyper_embed = torch.index_select(self.entity_embedding_hyper, 0, tail_part.reshape(-1)).view(batch_size, negative_sample_size, -1)
            candidate_ids = tail_part

        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE,
            'RelatE': self.RelatE_ERAS if getattr(self, 'use_eras', False) else self.RelatE,
            'RelateV': self.RelatE_ERAS if getattr(self, 'use_eras', False) else self.RelatE,
            'CCRelatE': self.CCRelatE,
            'BKRelatE': self.BKRelatE,
            'ARelatE': self.ARelatE,
            'CCMuRP': self.CCMuRP,
            'MuRP': self.MuRP
        }

        if self.model_name in {'RelatE', 'RelateV', 'CCRelatE', 'BKRelatE', 'ARelatE', 'CCMuRP', 'MuRP'}:
            score = model_func[self.model_name](head, relation, tail, mode,
                                                head_ids=head_ids,
                                                tail_ids=tail_ids,
                                                relation_ids=relation_ids,
                                                candidate_ids=candidate_ids,
                                                head_region_center=head_region_center,
                                                head_region_width=head_region_width,
                                                tail_region_center=tail_region_center,
                                                tail_region_width=tail_region_width,
                                                head_hyper_embed=head_hyper_embed,
                                                tail_hyper_embed=tail_hyper_embed)
        elif self.model_name in model_func:
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
    def RelatE(self, head, relation, tail, mode, head_ids=None, tail_ids=None, relation_ids=None,
               candidate_ids=None,
               head_region_center=None, head_region_width=None,
               tail_region_center=None, tail_region_width=None,
               head_hyper_embed=None, tail_hyper_embed=None):
        if self.use_hyperbolic:
            return self.HyperbolicRelatE(
                head, relation, tail, mode,
                head_ids=head_ids,
                tail_ids=tail_ids,
                relation_ids=relation_ids
            )
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

        query_embedding = None
        need_query = self.expose_query_embedding

        # Fetch phase and modulus weights
        phase_w = F.softplus(self.phase_weight[relation_ids]).view(-1, 1)   # [B, 1]
        modulus_w = F.softplus(self.modulus_weight[relation_ids]).view(-1, 1)  # [B, 1]

        sharp_mod = max(self.modulus_sharpness, 1e-6)
        sharp_phase = max(self.phase_sharpness, 1e-6)

        # Compute scores
        if mode == 'head-batch':
            if need_query:
                phase_core = tail_phase - rel_phase
                mod_core = tail_modulus * (1 - bias_relation)
                query_embedding = self._compose_query_embedding(phase_core, mod_core)
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
            if self.type_mod_scale is not None:
                entity_type_ids = self.entity_type_ids.to(head.device)
                head_scale = F.softplus(self.type_mod_scale(entity_type_ids[head_ids])).view(-1, 1)
                tail_scale = F.softplus(self.type_mod_scale(entity_type_ids[tail_ids])).view(-1, 1)
                if mod_dist.size(1) != 1:
                    head_scale = head_scale.unsqueeze(1).expand(-1, mod_dist.size(1), -1)
                    tail_scale = tail_scale.unsqueeze(1).expand(-1, mod_dist.size(1), -1)
                else:
                    head_scale = head_scale.unsqueeze(1)
                    tail_scale = tail_scale.unsqueeze(1)
                mod_dist = mod_dist / (0.5 * (head_scale + tail_scale) + 1e-6)
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
            if need_query:
                phase_core = head_phase + rel_phase
                mod_core = head_modulus * (rel_modulus + bias_relation)
                query_embedding = self._compose_query_embedding(phase_core, mod_core)
            phase_argument = head_phase + rel_phase - tail_phase
            phase_component = self.compute_phase_component(phase_argument, freq_weights)
            if sharp_phase != 1.0:
                phase_component = phase_component.pow(sharp_phase)
            phase_score = phase_component.sum(dim=2, keepdim=True)
            # modulus_score = torch.norm(head_modulus * (rel_modulus + bias_relation) - tail_modulus * (1 - bias_relation), p=2, dim=2, keepdim=True)

            # Tail-batch
            mod_dist = torch.abs(head_modulus * (rel_modulus + bias_relation) - tail_modulus * (1 - bias_relation))
            if self.type_mod_scale is not None:
                entity_type_ids = self.entity_type_ids.to(head.device)
                head_scale = F.softplus(self.type_mod_scale(entity_type_ids[head_ids])).view(-1, 1)
                tail_scale = F.softplus(self.type_mod_scale(entity_type_ids[tail_ids])).view(-1, 1)
                if mod_dist.size(1) != 1:
                    head_scale = head_scale.unsqueeze(1).expand(-1, mod_dist.size(1), -1)
                    tail_scale = tail_scale.unsqueeze(1).expand(-1, mod_dist.size(1), -1)
                else:
                    head_scale = head_scale.unsqueeze(1)
                    tail_scale = tail_scale.unsqueeze(1)
                mod_dist = mod_dist / (0.5 * (head_scale + tail_scale) + 1e-6)
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
            if need_query:
                phase_core = head_phase + rel_phase
                mod_core = head_modulus * (rel_modulus + bias_relation)
                query_embedding = self._compose_query_embedding(phase_core, mod_core)
            phase_argument = head_phase + rel_phase - tail_phase
            phase_component = self.compute_phase_component(phase_argument, freq_weights)
            if sharp_phase != 1.0:
                phase_component = phase_component.pow(sharp_phase)
            phase_score = phase_component.sum(dim=2, keepdim=True)
            # modulus_score = torch.norm(head_modulus * (rel_modulus + bias_relation) - tail_modulus * (1 - bias_relation), p=2, dim=2, keepdim=True)

            # Rest
            mod_dist = torch.abs(head_modulus * (rel_modulus + bias_relation) - tail_modulus * (1 - bias_relation))
            if self.type_mod_scale is not None:
                entity_type_ids = self.entity_type_ids.to(head.device)
                head_scale = F.softplus(self.type_mod_scale(entity_type_ids[head_ids])).view(-1, 1)
                tail_scale = F.softplus(self.type_mod_scale(entity_type_ids[tail_ids])).view(-1, 1)
                if mod_dist.size(1) != 1:
                    head_scale = head_scale.unsqueeze(1).expand(-1, mod_dist.size(1), -1)
                    tail_scale = tail_scale.unsqueeze(1).expand(-1, mod_dist.size(1), -1)
                else:
                    head_scale = head_scale.unsqueeze(1)
                    tail_scale = tail_scale.unsqueeze(1)
                mod_dist = mod_dist / (0.5 * (head_scale + tail_scale) + 1e-6)
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
        if self.dataset_phase_scale_logit is not None and self.dataset_modulus_scale_logit is not None:
            dataset_phase_scale = F.softplus(self.dataset_phase_scale_logit).view(1, 1, 1)
            dataset_modulus_scale = F.softplus(self.dataset_modulus_scale_logit).view(1, 1, 1)
            phase_score = phase_score * dataset_phase_scale
            modulus_score = modulus_score * dataset_modulus_scale

        if self.hierarchy_mod_scale is not None:
            hier_mask = self.hierarchical_rel_mask.index_select(0, relation_ids).float().view(-1, 1, 1)
            hier_scale = 1.0 + hier_mask * F.softplus(self.hierarchy_mod_scale[relation_ids]).view(-1, 1, 1)
            modulus_score = modulus_score * hier_scale

        qa_query_base = None
        phase_score, modulus_score, qa_query_base = self._apply_query_adaptive_scoring(
            phase_score,
            modulus_score,
            query_embedding,
            relation,
            relation_ids
        )

        if self.relation_gate is not None:
            gate = torch.sigmoid(self.relation_gate[relation_ids]).view(-1, 1, 1)
            if gate.size(1) != phase_score.size(1):
                gate = gate.expand(-1, phase_score.size(1), -1)
            combined = gate * phase_score + (1.0 - gate) * modulus_score
            base_score = self.gamma.item() - combined
        else:
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

        if (self.use_query_adaptive and self.qa_prototype_weight > 0 and
                qa_query_base is not None and self.concept_prototypes is not None and
                self.entity_concepts is not None):
            head_con = self.entity_concepts.index_select(0, head_ids).clamp(min=0)
            tail_con = self.entity_concepts.index_select(0, tail_ids).clamp(min=0)
            max_cid = self.concept_prototypes.size(0) - 1
            head_con = head_con.clamp(max=max_cid)
            tail_con = tail_con.clamp(max=max_cid)
            head_proto = self.concept_prototypes.index_select(0, head_con)
            tail_proto = self.concept_prototypes.index_select(0, tail_con)
            if mode == 'head-batch':
                proto_vec = head_proto
            else:
                proto_vec = tail_proto
            qn = F.normalize(qa_query_base, dim=-1)
            pn = F.normalize(proto_vec, dim=-1)
            proto_score = (qn * pn).sum(dim=-1, keepdim=True).unsqueeze(-1)
            if proto_score.size(1) != final_score.size(1):
                proto_score = proto_score.expand(-1, final_score.size(1), -1)
            final_score = final_score + self.qa_prototype_weight * proto_score



        self.latest_gate_entropy = None
        if self.use_mos_head and query_embedding is not None:
            candidate_embeddings = head if mode == 'head-batch' else tail
            mos_scores, gate_entropy = self._compute_mos_log_probs(
                query_embedding,
                candidate_embeddings.contiguous()
            )
            if mos_scores is not None:
                final_score = mos_scores.unsqueeze(-1)
                self.latest_gate_entropy = gate_entropy

        final_score = final_score.squeeze(-1)
        raw_hyper_score = None
        if self.use_hyper_subspace:
            hyper_score = self._compute_hyper_score(head_hyper_embed, relation_ids, tail_hyper_embed)
            if hyper_score is not None:
                if hyper_score.dim() > 1 and hyper_score.size(-1) == 1:
                    hyper_score = hyper_score.squeeze(-1)
                if hyper_score.dim() > 1 and hyper_score.size(1) == 1:
                    hyper_score = hyper_score.squeeze(1)
                raw_hyper_score = hyper_score
                blend_logits = self.hyper_blend_logits.index_select(0, relation_ids).squeeze(-1)
                blend = torch.sigmoid(blend_logits)
                if getattr(self, 'hyper_blend_warmup_steps', 0) > 0 and getattr(self, 'current_step', None) is not None:
                    scale = min(1.0, max(0.0, self.current_step) / self.hyper_blend_warmup_steps)
                    blend = blend * scale
                if hyper_score.dim() > 1:
                    blend = blend.unsqueeze(1).expand_as(hyper_score)
                final_score = (1.0 - blend) * final_score + blend * hyper_score
        if self.use_region_head:
            region_score = self._compute_region_score(
                head_region_center,
                head_region_width,
                relation_ids,
                tail_region_center,
                tail_region_width,
                mode
            )
            if region_score is not None:
                mask = self.region_rel_mask.index_select(0, relation_ids)
                if mask.any():
                    if region_score.dim() == 1:
                        region_mask = mask
                    else:
                        region_mask = mask.unsqueeze(1).expand_as(region_score)
                    blend = self.region_blend_weight
                    mixed = (1.0 - blend) * final_score + blend * region_score
                    final_score = torch.where(region_mask, mixed, final_score)

        if query_embedding is not None:
            return final_score, query_embedding, raw_hyper_score
        return final_score, None, raw_hyper_score

    def CCRelatE(self, head, relation, tail, mode, head_ids=None, tail_ids=None, relation_ids=None,
                 candidate_ids=None, head_region_center=None, head_region_width=None,
                 tail_region_center=None, tail_region_width=None,
                 head_hyper_embed=None, tail_hyper_embed=None):
        base_score, query_embedding, raw_hyper_score = self.RelatE(
            head, relation, tail, mode,
            head_ids=head_ids,
            tail_ids=tail_ids,
            relation_ids=relation_ids,
            candidate_ids=candidate_ids,
            head_region_center=head_region_center,
            head_region_width=head_region_width,
            tail_region_center=tail_region_center,
            tail_region_width=tail_region_width,
            head_hyper_embed=head_hyper_embed,
            tail_hyper_embed=tail_hyper_embed
        )

        if (self.cc_concept_prototypes is None or self.entity_concepts is None or
                relation_ids is None or head_ids is None or tail_ids is None):
            return base_score, query_embedding, raw_hyper_score

        if mode == 'head-batch':
            candidate_embed = head
            target_ids = head_ids
            other_ids = tail_ids
            if candidate_ids is None:
                candidate_ids = head_ids.unsqueeze(1)
        else:
            candidate_embed = tail
            target_ids = tail_ids
            other_ids = head_ids
            if candidate_ids is None:
                candidate_ids = tail_ids.unsqueeze(1)

        nconcept = self.cc_concept_prototypes.size(0)
        target_concepts = self.entity_concepts.index_select(0, target_ids)
        target_valid = (target_concepts >= 0) & (target_concepts < nconcept)
        if not target_valid.any():
            return base_score, query_embedding, raw_hyper_score

        safe_target = target_concepts.clamp(min=0, max=nconcept - 1)
        target_proto = self.cc_concept_prototypes.index_select(0, safe_target)
        rel_gate = torch.sigmoid(self.cc_relation_gate.index_select(0, relation_ids)).view(-1, 1)

        cand_norm = F.normalize(candidate_embed, dim=-1)
        target_proto_norm = F.normalize(target_proto, dim=-1).unsqueeze(1)
        concept_sim = (cand_norm * target_proto_norm).sum(dim=-1)
        valid_mask = target_valid.float().unsqueeze(1)

        bonus = 0.0
        if self.cc_concept_weight > 0:
            bonus = bonus + self.cc_concept_weight * rel_gate * concept_sim * valid_mask

        if self.cc_relation_weight > 0 and self.cc_relation_concept_shift is not None:
            other_concepts = self.entity_concepts.index_select(0, other_ids)
            other_valid = (other_concepts >= 0) & (other_concepts < nconcept)
            safe_other = other_concepts.clamp(min=0, max=nconcept - 1)
            other_proto = self.cc_concept_prototypes.index_select(0, safe_other)
            rel_shift = self.cc_relation_concept_shift.index_select(0, relation_ids)
            if mode == 'head-batch':
                expected_proto = other_proto - rel_shift
            else:
                expected_proto = other_proto + rel_shift
            expected_norm = F.normalize(expected_proto, dim=-1).unsqueeze(1)
            relation_sim = (cand_norm * expected_norm).sum(dim=-1)
            rel_valid = (target_valid & other_valid).float().unsqueeze(1)
            bonus = bonus + self.cc_relation_weight * rel_gate * relation_sim * rel_valid

        if (self.cc_depth_weight > 0 and self.concept_depths is not None and
                self.cc_depth_scale is not None and candidate_ids is not None):
            candidate_flat = candidate_ids.reshape(-1)
            cand_concepts = self.entity_concepts.index_select(0, candidate_flat).reshape(candidate_ids.shape)
            safe_cand_concepts = cand_concepts.clamp(min=0, max=nconcept - 1)
            safe_target_concepts = safe_target

            max_depth_id = self.concept_depths.size(0) - 1
            safe_cand_depth_idx = safe_cand_concepts.clamp(min=0, max=max_depth_id)
            safe_target_depth_idx = safe_target_concepts.clamp(min=0, max=max_depth_id)
            cand_depth = self.concept_depths.index_select(0, safe_cand_depth_idx.reshape(-1)).reshape(safe_cand_concepts.shape)
            target_depth = self.concept_depths.index_select(0, safe_target_depth_idx).unsqueeze(1)

            cand_valid = (cand_concepts >= 0) & (safe_cand_depth_idx >= 0) & (cand_depth >= 0)
            target_depth_valid = (target_depth >= 0) & target_valid.unsqueeze(1)
            depth_valid = cand_valid & target_depth_valid

            depth_diff = torch.abs(cand_depth - target_depth)
            depth_scale = F.softplus(self.cc_depth_scale.index_select(0, relation_ids)).view(-1, 1)
            hierarchy_mask = (
                self.hierarchical_rel_mask.index_select(0, relation_ids) |
                self.meronymic_rel_mask.index_select(0, relation_ids)
            ).float().view(-1, 1)
            depth_bonus = -depth_diff * depth_scale * hierarchy_mask
            depth_bonus = torch.where(depth_valid, depth_bonus, torch.zeros_like(depth_bonus))
            bonus = bonus + self.cc_depth_weight * depth_bonus

        if isinstance(bonus, float):
            return base_score, query_embedding, raw_hyper_score
        if base_score.dim() == 1 and bonus.dim() == 2 and bonus.size(1) == 1:
            bonus = bonus.squeeze(1)
        elif base_score.dim() == 2 and bonus.dim() == 1:
            bonus = bonus.unsqueeze(1).expand_as(base_score)

        return base_score + bonus, query_embedding, raw_hyper_score

    def BKRelatE(self, head, relation, tail, mode, head_ids=None, tail_ids=None, relation_ids=None,
                 candidate_ids=None, head_region_center=None, head_region_width=None,
                 tail_region_center=None, tail_region_width=None,
                 head_hyper_embed=None, tail_hyper_embed=None):
        base_score, query_embedding, raw_hyper_score = self.RelatE(
            head, relation, tail, mode,
            head_ids=head_ids,
            tail_ids=tail_ids,
            relation_ids=relation_ids,
            candidate_ids=candidate_ids,
            head_region_center=head_region_center,
            head_region_width=head_region_width,
            tail_region_center=tail_region_center,
            tail_region_width=tail_region_width,
            head_hyper_embed=head_hyper_embed,
            tail_hyper_embed=tail_hyper_embed
        )

        if not self.use_bk_relate or relation_ids is None:
            return base_score, query_embedding, raw_hyper_score

        klein_score = self._compute_klein_score(head_hyper_embed, relation_ids, tail_hyper_embed, mode)
        if klein_score is None:
            return base_score, query_embedding, raw_hyper_score

        rel_mask = (self.hierarchical_rel_mask.index_select(0, relation_ids) |
                    self.meronymic_rel_mask.index_select(0, relation_ids)).float()
        blend = torch.sigmoid(self.bk_relation_blend_logit.index_select(0, relation_ids)).squeeze(-1) * rel_mask
        if klein_score.dim() > 1:
            blend = blend.unsqueeze(1).expand_as(klein_score)
        mixed = (1.0 - blend) * base_score + blend * klein_score
        return mixed, query_embedding, raw_hyper_score

    def ARelatE(self, head, relation, tail, mode, head_ids=None, tail_ids=None, relation_ids=None,
                candidate_ids=None, head_region_center=None, head_region_width=None,
                tail_region_center=None, tail_region_width=None,
                head_hyper_embed=None, tail_hyper_embed=None):
        """
        Adaptive RelatE:
        - Reuses RelatE geometry/scoring
        - Adds relation-gated query-candidate cosine calibration
        - Applies relation-wise temperature and bias for top-rank calibration
        """
        base_score, query_embedding, raw_hyper_score = self.RelatE(
            head, relation, tail, mode,
            head_ids=head_ids,
            tail_ids=tail_ids,
            relation_ids=relation_ids,
            candidate_ids=candidate_ids,
            head_region_center=head_region_center,
            head_region_width=head_region_width,
            tail_region_center=tail_region_center,
            tail_region_width=tail_region_width,
            head_hyper_embed=head_hyper_embed,
            tail_hyper_embed=tail_hyper_embed
        )

        if (not self.use_arel_relate) or relation_ids is None:
            return base_score, query_embedding, raw_hyper_score

        if mode == 'head-batch':
            candidate_embed = head
            anchor_embed = tail[:, 0, :]
        else:
            candidate_embed = tail
            anchor_embed = head[:, 0, :]

        if query_embedding is not None:
            query_vec = query_embedding[:, 0, :]
        else:
            query_vec = anchor_embed

        q = F.normalize(query_vec, dim=-1).unsqueeze(1)
        c = F.normalize(candidate_embed, dim=-1)
        calib_score = (q * c).sum(dim=-1) * self.gamma.item()

        gate = torch.sigmoid(self.arel_gate_logit.index_select(0, relation_ids)).squeeze(-1)
        temp = F.softplus(self.arel_temp_logit.index_select(0, relation_ids)).squeeze(-1) + 1e-6
        bias = self.arel_bias.index_select(0, relation_ids).squeeze(-1)

        if base_score.dim() > 1:
            gate = gate.unsqueeze(1).expand_as(base_score)
            temp = temp.unsqueeze(1).expand_as(base_score)
            bias = bias.unsqueeze(1).expand_as(base_score)
        mixed = (1.0 - gate) * base_score + gate * calib_score
        mixed = mixed / temp + bias
        return mixed, query_embedding, raw_hyper_score

    def CCMuRP(self, head, relation, tail, mode, head_ids=None, tail_ids=None, relation_ids=None,
               candidate_ids=None, **kwargs):
        """
        Concept-Coached MuRP:
        - Hyperbolic MuRP distance is the primary score.
        - Concept and depth terms are integrated directly into final distance.
        """
        del kwargs
        if relation_ids is None or head_ids is None or tail_ids is None:
            raise ValueError('CCMuRP requires head_ids, tail_ids, and relation_ids.')

        if candidate_ids is None:
            if mode == 'head-batch':
                candidate_ids = head_ids.unsqueeze(1)
            else:
                candidate_ids = tail_ids.unsqueeze(1)

        head_h = self._project_to_ball(head)
        tail_h = self._project_to_ball(tail)
        relation_h = relation
        if self.ccmurp_relation_proj is not None:
            relation_h = self.ccmurp_relation_proj(relation_h)
        relation_h = self._project_to_ball(relation_h)

        if mode == 'head-batch':
            query = self._mobius_add(tail_h, self._mobius_neg(relation_h))
            cand = head_h
        else:
            query = self._mobius_add(head_h, relation_h)
            cand = tail_h

        base_dist = self._poincare_distance(query, cand)
        base_dist = torch.nan_to_num(base_dist, nan=1e3, posinf=1e3, neginf=0.0).clamp(min=0.0, max=1e3)
        final_dist = base_dist

        if (self.cc_concept_prototypes is None or self.entity_concepts is None or
                self.cc_relation_gate is None):
            return self.gamma.item() - final_dist, None, None

        nconcept = self.cc_concept_prototypes.size(0)
        rel_gate = torch.sigmoid(self.cc_relation_gate.index_select(0, relation_ids)).view(-1, 1)
        if final_dist.dim() > 1:
            rel_gate = rel_gate.expand(-1, final_dist.size(1))

        if mode == 'head-batch':
            anchor_concepts = self.entity_concepts.index_select(0, tail_ids)
        else:
            anchor_concepts = self.entity_concepts.index_select(0, head_ids)
        anchor_valid = (anchor_concepts >= 0) & (anchor_concepts < nconcept)
        safe_anchor = anchor_concepts.clamp(min=0, max=nconcept - 1)
        anchor_proto = self.cc_concept_prototypes.index_select(0, safe_anchor)
        rel_shift = self.cc_relation_concept_shift.index_select(0, relation_ids)
        if mode == 'head-batch':
            query_proto = anchor_proto - rel_shift
        else:
            query_proto = anchor_proto + rel_shift

        cand_flat = candidate_ids.reshape(-1)
        cand_concepts = self.entity_concepts.index_select(0, cand_flat).reshape(candidate_ids.shape)
        cand_valid = (cand_concepts >= 0) & (cand_concepts < nconcept)
        safe_cand = cand_concepts.clamp(min=0, max=nconcept - 1)
        cand_proto = self.cc_concept_prototypes.index_select(0, safe_cand.reshape(-1)).reshape(*safe_cand.shape, -1)

        qn = F.normalize(query_proto, dim=-1).unsqueeze(1)
        cn = F.normalize(cand_proto, dim=-1)
        concept_penalty = 1.0 - (qn * cn).sum(dim=-1)
        concept_penalty = torch.nan_to_num(concept_penalty, nan=0.0, posinf=10.0, neginf=-10.0).clamp(min=0.0, max=10.0)
        concept_mask = anchor_valid.unsqueeze(1) & cand_valid
        concept_penalty = torch.where(concept_mask, concept_penalty, torch.zeros_like(concept_penalty))

        final_dist = final_dist + self.cc_concept_weight * rel_gate * concept_penalty

        if self.cc_relation_weight > 0:
            relation_penalty = (query_proto.unsqueeze(1) - cand_proto).pow(2).sum(dim=-1).sqrt()
            relation_penalty = torch.nan_to_num(relation_penalty, nan=0.0, posinf=1e3, neginf=0.0).clamp(min=0.0, max=1e3)
            relation_penalty = torch.where(concept_mask, relation_penalty, torch.zeros_like(relation_penalty))
            final_dist = final_dist + self.cc_relation_weight * rel_gate * relation_penalty

        if self.cc_depth_weight > 0 and self.concept_depths is not None and self.cc_depth_scale is not None:
            max_depth_id = self.concept_depths.size(0) - 1
            safe_anchor_depth = safe_anchor.clamp(min=0, max=max_depth_id)
            safe_cand_depth = safe_cand.clamp(min=0, max=max_depth_id)
            anchor_depth = self.concept_depths.index_select(0, safe_anchor_depth).unsqueeze(1)
            cand_depth = self.concept_depths.index_select(0, safe_cand_depth.reshape(-1)).reshape(safe_cand.shape)
            depth_valid = concept_mask & (anchor_depth >= 0) & (cand_depth >= 0)
            depth_diff = torch.abs(anchor_depth - cand_depth)
            depth_scale = F.softplus(self.cc_depth_scale.index_select(0, relation_ids)).view(-1, 1)
            hier_mask = (
                self.hierarchical_rel_mask.index_select(0, relation_ids) |
                self.meronymic_rel_mask.index_select(0, relation_ids)
            ).float().view(-1, 1)
            depth_penalty = depth_diff * depth_scale * hier_mask
            depth_penalty = torch.nan_to_num(depth_penalty, nan=0.0, posinf=1e3, neginf=0.0).clamp(min=0.0, max=1e3)
            depth_penalty = torch.where(depth_valid, depth_penalty, torch.zeros_like(depth_penalty))
            final_dist = final_dist + self.cc_depth_weight * depth_penalty

        final_dist = torch.nan_to_num(final_dist, nan=1e3, posinf=1e3, neginf=0.0).clamp(min=0.0, max=1e3)
        score = self.gamma.item() - final_dist
        score = torch.nan_to_num(score, nan=-1e3, posinf=1e3, neginf=-1e3)
        return score, None, None

    def MuRP(self, head, relation, tail, mode, head_ids=None, tail_ids=None, relation_ids=None,
             candidate_ids=None, **kwargs):
        """
        Vanilla MuRP path (no concept coach terms).
        Reuses the same hyperbolic core as CCMuRP with coach weights forced off.
        """
        del kwargs
        return self.CCMuRP(
            head, relation, tail, mode,
            head_ids=head_ids,
            tail_ids=tail_ids,
            relation_ids=relation_ids,
            candidate_ids=candidate_ids
        )



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

    def HyperbolicRelatE(self, head, relation, tail, mode, **kwargs):
        head = self._project_to_ball(head)
        tail = self._project_to_ball(tail)
        relation = self._project_to_ball(relation)

        if mode == 'tail-batch':
            query = self._mobius_add(head, relation)
            query = self._project_to_ball(query)
            target = tail
        elif mode == 'head-batch':
            query = self._mobius_add(tail, self._mobius_neg(relation))
            query = self._project_to_ball(query)
            target = head
        else:
            query = self._mobius_add(head, relation)
            query = self._project_to_ball(query)
            target = tail

        dist = self._poincare_distance(query, target)
        return self.gamma.item() - dist




    
    @staticmethod
    def train_step(model, optimizer, train_iterator, args, step, path_batch=None,
                   path_weight=0.0, consistency_weight=0.0, zero_grad=True,
                   optimizer_step=True, accumulation_steps=1, batch=None,
                   teacher_scores=None):
        '''
        A single train step. Apply back-propation and return the loss
        '''
        base_model = model.module if hasattr(model, 'module') else model
        model.current_step = step
        model.train()

        if zero_grad:
            optimizer.zero_grad()

        def _unpack(output):
            if isinstance(output, tuple):
                if len(output) == 3:
                    return output
                if len(output) == 2:
                    return output[0], output[1], None
            return output, None, None

        if batch is None:
            positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)
        else:
            positive_sample, negative_sample, subsampling_weight, mode = batch

        def _mixkg_logits():
            if not getattr(args, 'mixkg_enable', False):
                return None
            if negative_sample is None or negative_sample.numel() == 0:
                return None
            if mode not in {'tail-batch', 'head-batch'}:
                return None
            if negative_logits is None or negative_logits.dim() <= 1:
                return None
            base = base_model
            model_name = getattr(base, 'model_name', None)
            if model_name == 'CCRelatE':
                relate_scorer = base.CCRelatE
            elif model_name == 'BKRelatE':
                relate_scorer = base.BKRelatE
            elif model_name == 'ARelatE':
                relate_scorer = base.ARelatE
            elif model_name == 'CCMuRP':
                relate_scorer = base.CCMuRP
            elif model_name == 'MuRP':
                relate_scorer = base.MuRP
            else:
                relate_scorer = base.RelatE
            neg_count = negative_sample.size(1)
            topk = min(getattr(args, 'mixkg_topk', 64), neg_count)
            mix_count = int(getattr(args, 'mixkg_mix_count', 32))
            if topk < 2 or mix_count <= 0:
                return None

            # Select hard negatives by score
            topk_idx = negative_logits.topk(topk, dim=1).indices
            cand_ids = negative_sample.gather(1, topk_idx)
            cand_logits = negative_logits.gather(1, topk_idx)

            # Optionally refine using similarity to positive entity
            if getattr(args, 'mixkg_use_similarity', False):
                pos_ids = positive_sample[:, 2] if mode == 'tail-batch' else positive_sample[:, 0]
                pos_emb = base.entity_embedding.index_select(0, pos_ids)
                cand_emb = base.entity_embedding.index_select(0, cand_ids.view(-1)).view(cand_ids.size(0), topk, -1)
                pos_norm = F.normalize(pos_emb, dim=-1).unsqueeze(1)
                cand_norm = F.normalize(cand_emb, dim=-1)
                sim = (cand_norm * pos_norm).sum(dim=-1)

                def _z(x):
                    mean = x.mean(dim=1, keepdim=True)
                    std = x.std(dim=1, keepdim=True) + 1e-6
                    return (x - mean) / std

                w = float(getattr(args, 'mixkg_score_weight', 0.5))
                score = w * _z(cand_logits) + (1.0 - w) * _z(sim)
                mix_topk = min(topk, max(2, topk // 2))
                sel_idx = score.topk(mix_topk, dim=1).indices
                cand_ids = cand_ids.gather(1, sel_idx)
                cand_emb = cand_emb.gather(1, sel_idx.unsqueeze(-1).expand(-1, -1, cand_emb.size(-1)))
            else:
                cand_emb = base.entity_embedding.index_select(0, cand_ids.view(-1)).view(cand_ids.size(0), topk, -1)
                mix_topk = topk

            if mix_topk < 2:
                return None

            # Sample pairs and mix
            device = cand_ids.device
            idx1 = torch.randint(0, mix_topk, (cand_ids.size(0), mix_count), device=device)
            idx2 = torch.randint(0, mix_topk, (cand_ids.size(0), mix_count), device=device)
            beta = torch.distributions.Beta(getattr(args, 'mixkg_alpha', 0.5), getattr(args, 'mixkg_alpha', 0.5))
            lam = beta.sample((cand_ids.size(0), mix_count)).to(device).unsqueeze(-1)

            emb1 = cand_emb.gather(1, idx1.unsqueeze(-1).expand(-1, -1, cand_emb.size(-1)))
            emb2 = cand_emb.gather(1, idx2.unsqueeze(-1).expand(-1, -1, cand_emb.size(-1)))
            mix_emb = lam * emb1 + (1.0 - lam) * emb2

            # Use first component id for id-based features
            mix_ids = cand_ids.gather(1, idx1)

            head_ids = positive_sample[:, 0]
            rel_ids = positive_sample[:, 1]
            tail_ids = positive_sample[:, 2]
            relation = base._lookup_relation_embedding(rel_ids).unsqueeze(1)

            head_region_center = head_region_width = tail_region_center = tail_region_width = None
            head_hyper_embed = tail_hyper_embed = None

            if mode == 'tail-batch':
                head = base.entity_embedding.index_select(0, head_ids).unsqueeze(1)
                if base.use_region_head:
                    head_region_center = base.entity_region_center.index_select(0, head_ids).unsqueeze(1)
                    head_region_width = base.entity_region_extent.index_select(0, head_ids).unsqueeze(1)
                    tail_region_center = base.entity_region_center.index_select(0, mix_ids.view(-1)).view(mix_ids.size(0), mix_ids.size(1), -1)
                    tail_region_width = base.entity_region_extent.index_select(0, mix_ids.view(-1)).view(mix_ids.size(0), mix_ids.size(1), -1)
                if base.use_hyper_subspace:
                    head_hyper_embed = base.entity_embedding_hyper.index_select(0, head_ids).unsqueeze(1)
                    tail_hyper_embed = base.entity_embedding_hyper.index_select(0, mix_ids.view(-1)).view(mix_ids.size(0), mix_ids.size(1), -1)
                mix_score = relate_scorer(
                    head, relation, mix_emb, 'tail-batch',
                    head_ids=head_ids, tail_ids=tail_ids, relation_ids=rel_ids,
                    candidate_ids=mix_ids,
                    head_region_center=head_region_center, head_region_width=head_region_width,
                    tail_region_center=tail_region_center, tail_region_width=tail_region_width,
                    head_hyper_embed=head_hyper_embed, tail_hyper_embed=tail_hyper_embed
                )
            else:
                tail = base.entity_embedding.index_select(0, tail_ids).unsqueeze(1)
                if base.use_region_head:
                    tail_region_center = base.entity_region_center.index_select(0, tail_ids).unsqueeze(1)
                    tail_region_width = base.entity_region_extent.index_select(0, tail_ids).unsqueeze(1)
                    head_region_center = base.entity_region_center.index_select(0, mix_ids.view(-1)).view(mix_ids.size(0), mix_ids.size(1), -1)
                    head_region_width = base.entity_region_extent.index_select(0, mix_ids.view(-1)).view(mix_ids.size(0), mix_ids.size(1), -1)
                if base.use_hyper_subspace:
                    tail_hyper_embed = base.entity_embedding_hyper.index_select(0, tail_ids).unsqueeze(1)
                    head_hyper_embed = base.entity_embedding_hyper.index_select(0, mix_ids.view(-1)).view(mix_ids.size(0), mix_ids.size(1), -1)
                mix_score = relate_scorer(
                    mix_emb, relation, tail, 'head-batch',
                    head_ids=head_ids, tail_ids=tail_ids, relation_ids=rel_ids,
                    candidate_ids=mix_ids,
                    head_region_center=head_region_center, head_region_width=head_region_width,
                    tail_region_center=tail_region_center, tail_region_width=tail_region_width,
                    head_hyper_embed=head_hyper_embed, tail_hyper_embed=tail_hyper_embed
                )

            if mix_score is None:
                return None
            if mix_score.dim() == 2:
                return mix_score
            return mix_score.squeeze(-1)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        mos_enabled = getattr(base_model, 'use_mos_head', False)
        full_ranking_ce = bool(getattr(args, 'full_ranking_ce', False))
        base_model.latest_gate_entropy = None

        hyper_positive_logits = None
        hyper_negative_logits = None
        negative_logits = None
        query_embedding = None
        positive_score = None
        negative_score = None

        if full_ranking_ce:
            batch_size = positive_sample.size(0)
            if mode == 'tail-batch':
                target_ids = positive_sample[:, 2]
            elif mode == 'head-batch':
                target_ids = positive_sample[:, 0]
            else:
                raise ValueError('Full-ranking CE is only compatible with head-batch/tail-batch modes.')

            entity_ids = torch.arange(args.nentity, device=positive_sample.device, dtype=torch.long)
            chunk_size = int(max(64, getattr(args, 'full_ranking_chunk_size', 2048)))
            lse = None
            logits_sum = None
            positive_logits = torch.empty(batch_size, device=positive_sample.device, dtype=torch.float32)
            for start in range(0, args.nentity, chunk_size):
                end = min(args.nentity, start + chunk_size)
                cand = entity_ids[start:end].unsqueeze(0).expand(batch_size, -1)
                chunk_logits, _, _ = _unpack(model((positive_sample, cand), mode=mode))
                chunk_lse = torch.logsumexp(chunk_logits, dim=1)
                chunk_sum = chunk_logits.sum(dim=1)
                rel_pos = target_ids - start
                in_chunk = (rel_pos >= 0) & (rel_pos < (end - start))
                if in_chunk.any():
                    rows = in_chunk.nonzero(as_tuple=False).squeeze(-1)
                    cols = rel_pos[in_chunk].long()
                    positive_logits[rows] = chunk_logits[rows, cols]
                if lse is None:
                    lse = chunk_lse
                    logits_sum = chunk_sum
                else:
                    lse = torch.logaddexp(lse, chunk_lse)
                    logits_sum = logits_sum + chunk_sum

            positive_logits = positive_logits.reshape(-1)
            lse = lse.reshape(-1)
            nll = lse - positive_logits

            smoothing = float(max(0.0, min(1.0, getattr(args, 'full_ranking_label_smoothing', 0.0))))
            if smoothing > 0.0:
                mean_logits = logits_sum / float(args.nentity)
                uniform_nll = lse - mean_logits
                ce_per_sample = (1.0 - smoothing) * nll + smoothing * uniform_nll
            else:
                ce_per_sample = nll

            rel_weight_tensor = getattr(base_model, 'relation_loss_weights', None)
            rel_weights = None
            if rel_weight_tensor is not None:
                rel_ids = positive_sample[:, 1]
                rel_weights = rel_weight_tensor.index_select(0, rel_ids).to(ce_per_sample.device)

            if args.uni_weight:
                if rel_weights is None:
                    positive_sample_loss = ce_per_sample.mean()
                else:
                    denom = rel_weights.sum().clamp(min=1e-9)
                    positive_sample_loss = (rel_weights * ce_per_sample).sum() / denom
            else:
                base_weights = subsampling_weight
                if rel_weights is not None:
                    base_weights = base_weights * rel_weights
                denom = base_weights.sum().clamp(min=1e-9)
                positive_sample_loss = (base_weights * ce_per_sample).sum() / denom
            negative_sample_loss = torch.zeros_like(positive_sample_loss)
            loss = positive_sample_loss

        elif mos_enabled:
            if mode == 'tail-batch':
                pos_ids = positive_sample[:, 2].unsqueeze(1)
            elif mode == 'head-batch':
                pos_ids = positive_sample[:, 0].unsqueeze(1)
            else:
                raise ValueError('MoS head is only compatible with head-batch/tail-batch modes.')
            all_candidates = torch.cat([pos_ids, negative_sample], dim=1)
            logits, query_embedding, hyper_logits_full = _unpack(model((positive_sample, all_candidates), mode=mode))
            positive_logits = logits[:, 0]
            negative_logits = logits[:, 1:]
            positive_score = positive_logits
            negative_score = negative_logits
            if hyper_logits_full is not None:
                hyper_positive_logits = hyper_logits_full[:, 0]
                hyper_negative_logits = hyper_logits_full[:, 1:]
        else:
            negative_logits, _, hyper_neg_logits = _unpack(model((positive_sample, negative_sample), mode=mode))
            if hyper_neg_logits is not None:
                hyper_negative_logits = hyper_neg_logits

            mix_logits = _mixkg_logits()
            if mix_logits is not None and mix_logits.numel() > 0:
                if mix_logits.dim() == 1:
                    mix_logits = mix_logits.unsqueeze(1)
                negative_logits = torch.cat([negative_logits, mix_logits], dim=1)

            if args.negative_adversarial_sampling:
                if negative_logits.dim() == 1:
                    negative_logits = negative_logits.unsqueeze(1)
                adv_weight = F.softmax(negative_logits * args.adversarial_temperature, dim=1).detach()
                negative_score = (adv_weight * F.logsigmoid(-negative_logits)).sum(dim=1)
            else:
                negative_score = F.logsigmoid(-negative_logits).mean(dim=1)

            positive_logits, query_embedding, hyper_pos_logits = _unpack(model(positive_sample))
            if hyper_pos_logits is not None:
                hyper_positive_logits = hyper_pos_logits.squeeze(-1)
            positive_score = F.logsigmoid(positive_logits)

        if hyper_positive_logits is not None:
            if hyper_positive_logits.dim() > 1 and hyper_positive_logits.size(-1) == 1:
                hyper_positive_logits = hyper_positive_logits.squeeze(-1)
            if hyper_positive_logits.dim() > 1 and hyper_positive_logits.size(1) == 1:
                hyper_positive_logits = hyper_positive_logits.squeeze(1)
        if hyper_negative_logits is not None and hyper_negative_logits.dim() > 2 and hyper_negative_logits.size(-1) == 1:
            hyper_negative_logits = hyper_negative_logits.squeeze(-1)

        subsampling_weight = subsampling_weight.view(-1)

        rel_weight_tensor = getattr(base_model, 'relation_loss_weights', None)
        rel_weights = None
        if not full_ranking_ce:
            loss = None
            positive_sample_loss = None
            negative_sample_loss = None
        hyper_student_logits = None
        if hyper_positive_logits is not None and hyper_negative_logits is not None:
            hyper_student_logits = (hyper_positive_logits, hyper_negative_logits)

        if full_ranking_ce:
            # Loss already computed in the full-ranking CE branch above.
            pass
        elif mos_enabled:
            base_weights = subsampling_weight
            if rel_weight_tensor is not None:
                rel_ids = positive_sample[:, 1]
                rel_weights = rel_weight_tensor.index_select(0, rel_ids).to(base_weights.device)
                base_weights = base_weights * rel_weights
            weight_sum = base_weights.sum().clamp(min=1e-9)
            positive_sample_loss = - (base_weights * positive_score).sum() / weight_sum
            negative_sample_loss = torch.zeros_like(positive_sample_loss)
            loss = positive_sample_loss
        else:
            if rel_weight_tensor is not None:
                rel_ids = positive_sample[:, 1]
                rel_weights = rel_weight_tensor.index_select(0, rel_ids).to(positive_score.device)

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

            def _weighted_mean(values, weights):
                if weights is None:
                    return values.mean()
                denom = weights.sum().clamp(min=1e-9)
                return (weights * values).sum() / denom

            if args.uni_weight:
                positive_sample_loss = -_weighted_mean(positive_score, rel_weights)
                negative_sample_loss = -_weighted_mean(negative_score, rel_weights)
            else:
                base_weights = subsampling_weight
                if rel_weights is not None:
                    base_weights = base_weights * rel_weights
                weight_sum = base_weights.sum().clamp(min=1e-9)
                positive_sample_loss = - (base_weights * positive_score).sum() / weight_sum
                if negative_score.dim() == 1:
                    negative_sample_loss = - (base_weights * negative_score).sum() / weight_sum
                else:
                    expanded = base_weights.unsqueeze(1).expand_as(negative_score)
                    negative_sample_loss = - (expanded * negative_score).sum() / expanded.sum().clamp(min=1e-9)

            loss = (positive_sample_loss + negative_sample_loss) / 2

        kd_logs = {}
        kd_lambda = getattr(args, 'kd_lambda', 0.0)
        kd_weight = kd_lambda
        kd_warmup = getattr(args, 'kd_warmup_steps', 0)
        kd_decay_start = getattr(args, 'kd_decay_start', None)
        kd_decay_duration = getattr(args, 'kd_decay_duration', 0)
        current_step = getattr(base_model, 'current_step', None)
        if kd_weight > 0.0 and kd_warmup > 0 and current_step is not None:
            warm_scale = min(1.0, max(0.0, current_step) / kd_warmup)
            kd_weight = kd_weight * warm_scale
        if (kd_weight > 0.0 and kd_decay_start is not None and kd_decay_duration > 0
                and current_step is not None and current_step >= kd_decay_start):
            decay_progress = min(1.0, max(0.0, (current_step - kd_decay_start) / kd_decay_duration))
            kd_weight = kd_weight * max(0.0, 1.0 - decay_progress)
        kd_relation_weights = getattr(base_model, 'kd_relation_weights', None)
        if teacher_scores and kd_weight > 0.0 and not mos_enabled and negative_logits is not None:
            kd_loss = KGEModel._compute_kd_loss(
                positive_logits,
                negative_logits,
                teacher_scores,
                args,
                relation_ids=positive_sample[:, 1],
                kd_weights=kd_relation_weights
            )
            if kd_loss is not None:
                loss = loss + kd_weight * kd_loss
                kd_logs['kd_loss'] = kd_loss.item()
        hyper_kd_weight = getattr(args, 'kd_hyper_weight', 1.0)
        if hyper_kd_weight > 0 and getattr(base_model, 'current_step', None) is not None:
            hyper_warmup = getattr(args, 'hyper_kd_warmup_steps', 0)
            if hyper_warmup > 0:
                scale = min(1.0, max(0.0, base_model.current_step) / hyper_warmup)
                hyper_kd_weight = hyper_kd_weight * scale

        if (teacher_scores and kd_weight > 0.0 and hyper_student_logits is not None
                and negative_logits is not None
                and hyper_kd_weight > 0.0):
            hyper_kd_loss = KGEModel._compute_kd_loss(
                hyper_student_logits[0],
                hyper_student_logits[1],
                teacher_scores,
                args,
                relation_ids=positive_sample[:, 1],
                kd_weights=kd_relation_weights
            )
            if hyper_kd_loss is not None:
                loss = loss + kd_weight * hyper_kd_weight * hyper_kd_loss
                kd_logs['kd_hyper_loss'] = hyper_kd_loss.item()

        alignment_logs = {}
        align_weight = getattr(args, 'teacher_align_weight', 0.0)
        if align_weight > 0 and getattr(base_model, 'teacher_projector', None) is not None:
            align_loss = KGEModel._compute_alignment_loss(
                base_model,
                positive_sample,
                negative_sample,
                args
            )
            if align_loss is not None:
                loss = loss + align_weight * align_loss
                alignment_logs['alignment_loss'] = align_loss.item()

        query_align_weight = getattr(args, 'teacher_query_align_weight', 0.0)
        if query_align_weight > 0 and teacher_scores and query_embedding is not None:
            query_align_loss = KGEModel._compute_query_alignment_loss(
                base_model,
                query_embedding,
                teacher_scores
            )
            if query_align_loss is not None:
                loss = loss + query_align_weight * query_align_loss
                alignment_logs['query_alignment_loss'] = query_align_loss.item()

        qa_logs = {}
        qa_contrastive_weight = getattr(args, 'qa_contrastive_weight', 0.0)
        if (qa_contrastive_weight > 0 and query_embedding is not None and
                getattr(base_model, 'qa_query_proj', None) is not None and
                getattr(base_model, 'qa_entity_proj', None) is not None and
                mode in {'head-batch', 'tail-batch'} and negative_sample is not None):
            batch_size = positive_sample.size(0)
            if mode == 'tail-batch':
                pos_ids = positive_sample[:, 2].unsqueeze(1)
            else:
                pos_ids = positive_sample[:, 0].unsqueeze(1)
            cand_ids = torch.cat([pos_ids, negative_sample], dim=1)
            cand_emb = base_model.entity_embedding.index_select(
                0, cand_ids.view(-1)
            ).view(batch_size, cand_ids.size(1), -1)
            q_vec = query_embedding[:, 0, :]
            q_vec = base_model.qa_query_proj(q_vec)
            cand_emb = base_model.qa_entity_proj(cand_emb)
            q_vec = F.normalize(q_vec, dim=-1)
            cand_emb = F.normalize(cand_emb, dim=-1)
            temp = max(1e-6, float(getattr(args, 'qa_contrastive_temp', 0.07)))
            logits = torch.einsum('bd,bnd->bn', q_vec, cand_emb) / temp
            targets = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
            qa_contrastive_loss = F.cross_entropy(logits, targets)
            loss = loss + qa_contrastive_weight * qa_contrastive_loss
            qa_logs['qa_contrastive_loss'] = qa_contrastive_loss.item()

        region_logs = {}
        if getattr(base_model, 'use_region_head', False):
            volume_weight = getattr(args, 'region_volume_penalty', 0.0)
            if volume_weight > 0:
                batch_entities = torch.unique(
                    torch.cat([positive_sample[:, 0], positive_sample[:, 2]], dim=0)
                )
                if batch_entities.numel() > 0:
                    widths = F.softplus(
                        base_model.entity_region_extent.index_select(0, batch_entities)
                    )
                    volume_penalty = widths.mean()
                    loss = loss + volume_weight * volume_penalty
                    region_logs['region_volume_penalty'] = volume_penalty.item()

            region_depth_weight = getattr(args, 'region_depth_weight', 0.0)
            depth_tensor = getattr(base_model, 'entity_depths', None)
            if region_depth_weight > 0 and depth_tensor is not None:
                relation_ids = positive_sample[:, 1]
                hier_mask = base_model.hierarchical_rel_mask.index_select(0, relation_ids)
                if hier_mask.any():
                    child_idx = positive_sample[hier_mask, 0]
                    parent_idx = positive_sample[hier_mask, 2]
                    child_width = F.softplus(
                        base_model.entity_region_extent.index_select(0, child_idx)
                    ).mean(dim=1)
                    parent_width = F.softplus(
                        base_model.entity_region_extent.index_select(0, parent_idx)
                    ).mean(dim=1)
                    margin = getattr(args, 'region_depth_margin', 0.0)
                    region_depth_penalty = F.relu(child_width - parent_width + margin).mean()
                    loss = loss + region_depth_weight * region_depth_penalty
                    region_logs['region_depth_penalty'] = region_depth_penalty.item()

        hyper_logs = {}
        hyper_radius_weight = getattr(args, 'hyper_radius_weight', 0.0)
        if hyper_radius_weight > 0 and getattr(base_model, 'current_step', None) is not None:
            radius_warmup = getattr(args, 'hyper_radius_warmup_steps', 0)
            if radius_warmup > 0:
                scale = min(1.0, max(0.0, base_model.current_step) / radius_warmup)
                hyper_radius_weight = hyper_radius_weight * scale

        if (hyper_radius_weight > 0 and getattr(base_model, 'use_hyper_subspace', False)
                and getattr(base_model, 'entity_embedding_hyper', None) is not None):
            rel_ids = positive_sample[:, 1]
            hier_mask = base_model.hierarchical_rel_mask.index_select(0, rel_ids)
            if hier_mask.any():
                child_idx = positive_sample[hier_mask, 0]
                parent_idx = positive_sample[hier_mask, 2]
                child_vec = base_model.entity_embedding_hyper.index_select(0, child_idx)
                parent_vec = base_model.entity_embedding_hyper.index_select(0, parent_idx)
                child_radius = torch.norm(child_vec, dim=1)
                parent_radius = torch.norm(parent_vec, dim=1)
                radius_margin = getattr(args, 'hyper_radius_margin', 0.0)
                radius_penalty = F.relu(child_radius - parent_radius + radius_margin).mean()
                loss = loss + hyper_radius_weight * radius_penalty
                hyper_logs['hyper_radius_penalty'] = radius_penalty.item()

        mos_logs = {}
        if mos_enabled and base_model.mos_entropy_weight > 0:
            gate_entropy = getattr(base_model, 'latest_gate_entropy', None)
            if gate_entropy is not None:
                loss = loss - base_model.mos_entropy_weight * gate_entropy
                mos_logs['mos_entropy'] = gate_entropy.item()

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
                single_score, _, _ = _unpack(model(composed_sample, mode='single'))
                single_score = single_score.view(path_heads.size(0), 1)
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

        hierarchy_log = {}
        depth_weight = getattr(args, 'depth_penalty_weight', 0.0)
        depth_tensor = getattr(base_model, 'entity_depths', None)
        if depth_weight > 0 and depth_tensor is not None:
            entity_dim_half = base_model.entity_dim // 2
            head_idx = positive_sample[:, 0]
            tail_idx = positive_sample[:, 2]
            head_depth = depth_tensor[head_idx]
            tail_depth = depth_tensor[tail_idx]
            head_valid = head_depth >= 0
            tail_valid = tail_depth >= 0
            valid_mask = head_valid & tail_valid
            if valid_mask.any():
                head_emb = base_model.entity_embedding.index_select(0, head_idx)
                tail_emb = base_model.entity_embedding.index_select(0, tail_idx)
                head_modulus = torch.norm(head_emb[:, :entity_dim_half], p=2, dim=1)
                tail_modulus = torch.norm(tail_emb[:, :entity_dim_half], p=2, dim=1)
                margin_base = args.depth_penalty_margin
                penalties = []
                parent_head = valid_mask & (head_depth < tail_depth)
                if parent_head.any():
                    child_norm = tail_modulus[parent_head]
                    parent_norm = head_modulus[parent_head]
                    depth_gap = (tail_depth[parent_head] - head_depth[parent_head]).abs()
                    margin = margin_base * depth_gap if args.depth_penalty_scale_gap else margin_base
                    penalties.append(F.relu(child_norm - parent_norm + margin))
                parent_tail = valid_mask & (tail_depth < head_depth)
                if parent_tail.any():
                    child_norm = head_modulus[parent_tail]
                    parent_norm = tail_modulus[parent_tail]
                    depth_gap = (head_depth[parent_tail] - tail_depth[parent_tail]).abs()
                    margin = margin_base * depth_gap if args.depth_penalty_scale_gap else margin_base
                    penalties.append(F.relu(child_norm - parent_norm + margin))
                if penalties:
                    hierarchy_penalty = torch.cat(penalties).mean()
                    loss = loss + depth_weight * hierarchy_penalty
                    hierarchy_log = {'hierarchy_penalty': hierarchy_penalty.item()}

        hier_contrastive_weight = getattr(args, 'hier_contrastive_weight', 0.0)
        if hier_contrastive_weight > 0 and depth_tensor is not None:
            hier_mask = getattr(base_model, 'hierarchical_rel_mask', None)
            if hier_mask is not None:
                relation_ids = positive_sample[:, 1]
                hier_batch_mask = hier_mask.index_select(0, relation_ids)
            else:
                hier_batch_mask = None
            if hier_batch_mask is not None and hier_batch_mask.any():
                temp = max(1e-6, float(getattr(args, 'hier_contrastive_temp', 0.07)))
                phase_only = getattr(args, 'hier_contrastive_phase_only', False)
                depth_neg_k = int(getattr(args, 'hier_depth_negatives', 16))
                entity_emb = base_model.entity_embedding
                relation_emb = base_model.relation_embedding
                if phase_only:
                    half = base_model.entity_dim // 2
                    entity_emb = entity_emb[:, half:]
                    relation_emb = relation_emb[:, half:]
                tails = positive_sample[hier_batch_mask, 2]
                heads = positive_sample[hier_batch_mask, 0]
                rels = positive_sample[hier_batch_mask, 1]
                if tails.numel() > 0:
                    query = entity_emb.index_select(0, heads) + relation_emb.index_select(0, rels)
                    pos_vec = entity_emb.index_select(0, tails)
                    query = F.normalize(query, dim=-1)
                    pos_vec = F.normalize(pos_vec, dim=-1)
                    inbatch_tails = torch.unique(tails)
                    inbatch_vec = entity_emb.index_select(0, inbatch_tails)
                    inbatch_vec = F.normalize(inbatch_vec, dim=-1)
                    losses = []
                    for idx in range(tails.size(0)):
                        pos_id = tails[idx].item()
                        pos = pos_vec[idx:idx + 1]
                        q = query[idx:idx + 1]
                        candidates = []
                        # Always include positive at position 0
                        candidates.append(pos)
                        # In-batch negatives
                        if inbatch_tails.numel() > 1:
                            mask = inbatch_tails != pos_id
                            if mask.any():
                                candidates.append(inbatch_vec[mask])
                        # Depth-matched negatives
                        if depth_neg_k > 0:
                            pos_depth = depth_tensor[pos_id]
                            if pos_depth >= 0:
                                depth_ids = (depth_tensor == pos_depth).nonzero(as_tuple=False).view(-1)
                                depth_ids = depth_ids[depth_ids != pos_id]
                                if depth_ids.numel() > 0:
                                    if depth_ids.numel() > depth_neg_k:
                                        rand_idx = torch.randint(0, depth_ids.numel(), (depth_neg_k,), device=depth_ids.device)
                                        depth_ids = depth_ids.index_select(0, rand_idx)
                                    depth_vec = entity_emb.index_select(0, depth_ids)
                                    depth_vec = F.normalize(depth_vec, dim=-1)
                                    candidates.append(depth_vec)
                        if len(candidates) <= 1:
                            continue
                        cand = torch.cat(candidates, dim=0)
                        logits = (q @ cand.t()).squeeze(0) / temp
                        log_probs = F.log_softmax(logits, dim=0)
                        losses.append(-log_probs[0])
                    if losses:
                        hier_contrast_loss = torch.stack(losses).mean()
                        loss = loss + hier_contrastive_weight * hier_contrast_loss
                        hierarchy_log['hier_contrastive_loss'] = hier_contrast_loss.item()

        concept_logs = {}
        concept_ids = getattr(base_model, 'entity_concepts', None)
        if concept_ids is not None:
            concept_phase_weight = getattr(args, 'concept_phase_weight', 0.0)
            concept_mod_weight = getattr(args, 'concept_modulus_weight', 0.0)
            concept_rel_weight = getattr(args, 'concept_relation_weight', 0.0)
            concept_depth_margin = getattr(args, 'concept_depth_margin', 0.0)

            head_idx = positive_sample[:, 0]
            rel_idx = positive_sample[:, 1]
            tail_idx = positive_sample[:, 2]
            head_concepts = concept_ids.index_select(0, head_idx)
            tail_concepts = concept_ids.index_select(0, tail_idx)
            valid_concepts = (head_concepts >= 0) & (tail_concepts >= 0)

            half = base_model.entity_dim // 2
            entity_phase = base_model.entity_embedding[:, half:]
            entity_mod = base_model.entity_embedding[:, :half]

            if concept_phase_weight > 0:
                same_concept = valid_concepts & (head_concepts == tail_concepts)
                if same_concept.any():
                    head_phase = entity_phase.index_select(0, head_idx[same_concept])
                    tail_phase = entity_phase.index_select(0, tail_idx[same_concept])
                    concept_phase_loss = (head_phase - tail_phase).abs().mean()
                    loss = loss + concept_phase_weight * concept_phase_loss
                    concept_logs['concept_phase_loss'] = concept_phase_loss.item()

            concept_depths = getattr(base_model, 'concept_depths', None)
            if concept_mod_weight > 0:
                if concept_depths is not None:
                    head_depth = concept_depths.index_select(0, head_concepts.clamp(min=0))
                    tail_depth = concept_depths.index_select(0, tail_concepts.clamp(min=0))
                    depth_valid = valid_concepts & (head_depth >= 0) & (tail_depth >= 0)
                    if depth_valid.any():
                        hm = torch.norm(entity_mod.index_select(0, head_idx[depth_valid]), p=2, dim=1)
                        tm = torch.norm(entity_mod.index_select(0, tail_idx[depth_valid]), p=2, dim=1)
                        modulus_vals = torch.cat([hm, tm], dim=0)
                        depth_vals = torch.cat([head_depth[depth_valid], tail_depth[depth_valid]], dim=0)
                        m_norm = (modulus_vals - modulus_vals.mean()) / (modulus_vals.std() + 1e-6)
                        d_norm = (depth_vals - depth_vals.mean()) / (depth_vals.std() + 1e-6)
                        concept_mod_loss = F.mse_loss(m_norm, d_norm)
                        loss = loss + concept_mod_weight * concept_mod_loss
                        concept_logs['concept_modulus_loss'] = concept_mod_loss.item()
                else:
                    same_concept = valid_concepts & (head_concepts == tail_concepts)
                    if same_concept.any():
                        hm = torch.norm(entity_mod.index_select(0, head_idx[same_concept]), p=2, dim=1)
                        tm = torch.norm(entity_mod.index_select(0, tail_idx[same_concept]), p=2, dim=1)
                        concept_mod_loss = (hm - tm).abs().mean()
                        loss = loss + concept_mod_weight * concept_mod_loss
                        concept_logs['concept_modulus_loss'] = concept_mod_loss.item()

            if concept_rel_weight > 0:
                rel_losses = []
                hier_mask = getattr(base_model, 'hierarchical_rel_mask', None)
                if hier_mask is not None and concept_depths is not None:
                    hier_batch = hier_mask.index_select(0, rel_idx)
                    if hier_batch.any():
                        h_con = head_concepts[hier_batch]
                        t_con = tail_concepts[hier_batch]
                        h_valid = h_con >= 0
                        t_valid = t_con >= 0
                        pair_valid = h_valid & t_valid
                        if pair_valid.any():
                            h_depth = concept_depths.index_select(0, h_con[pair_valid])
                            t_depth = concept_depths.index_select(0, t_con[pair_valid])
                            d_valid = (h_depth >= 0) & (t_depth >= 0)
                            if d_valid.any():
                                # For hierarchical edges (child -> parent), child depth should be >= parent depth.
                                hier_loss = F.relu(t_depth[d_valid] - h_depth[d_valid] + concept_depth_margin).mean()
                                rel_losses.append(hier_loss)

                mero_mask = getattr(base_model, 'meronymic_rel_mask', None)
                if mero_mask is not None:
                    mero_batch = mero_mask.index_select(0, rel_idx) & valid_concepts
                    if mero_batch.any():
                        hp = entity_phase.index_select(0, head_idx[mero_batch])
                        tp = entity_phase.index_select(0, tail_idx[mero_batch])
                        mero_loss = (hp - tp).abs().mean()
                        rel_losses.append(mero_loss)

                if rel_losses:
                    concept_rel_loss = torch.stack(rel_losses).mean()
                    loss = loss + concept_rel_weight * concept_rel_loss
                    concept_logs['concept_relation_loss'] = concept_rel_loss.item()

        relation_phase_logs = {}
        rel_phase_start = base_model.relation_dim // 2
        relation_phase = base_model.relation_embedding[:, rel_phase_start:]
        sym_mask = getattr(base_model, 'symmetric_rel_mask', None)
        sym_weight = getattr(args, 'symmetric_phase_weight', 0.0)
        if sym_weight > 0 and sym_mask is not None and sym_mask.any():
            sym_phase = relation_phase[sym_mask]
            sym_penalty = sym_phase.abs().mean()
            loss = loss + sym_weight * sym_penalty
            relation_phase_logs['symmetric_phase_penalty'] = sym_penalty.item()
        anti_mask = getattr(base_model, 'antisymmetric_rel_mask', None)
        anti_weight = getattr(args, 'antisymmetric_phase_weight', 0.0)
        if anti_weight > 0 and anti_mask is not None and anti_mask.any():
            anti_phase = relation_phase[anti_mask]
            anti_penalty = ((anti_phase - math.pi) ** 2).mean()
            loss = loss + anti_weight * anti_penalty
            relation_phase_logs['antisymmetric_phase_penalty'] = anti_penalty.item()
        inverse_pairs = getattr(base_model, 'inverse_rel_pairs', None)
        inv_weight = getattr(args, 'inverse_phase_weight', 0.0)
        if inv_weight > 0 and inverse_pairs is not None and inverse_pairs.numel() > 0:
            lhs_phase = relation_phase[inverse_pairs[:, 0]]
            rhs_phase = relation_phase[inverse_pairs[:, 1]]
            inv_penalty = (lhs_phase + rhs_phase).abs().mean()
            loss = loss + inv_weight * inv_penalty
            relation_phase_logs['inverse_phase_penalty'] = inv_penalty.item()

        plm_logs = {}
        ent_reg_w = getattr(args, 'plm_entity_reg_weight', 0.0)
        if ent_reg_w > 0 and getattr(base_model, 'plm_entity_vectors', None) is not None:
            ent_ids = torch.cat([positive_sample[:, 0], positive_sample[:, 2]], dim=0)
            ent_unique = torch.unique(ent_ids)
            emb = base_model.entity_embedding.index_select(0, ent_unique)
            plm = base_model.plm_entity_vectors.index_select(0, ent_unique)
            plm_proj = base_model.plm_entity_proj(plm)
            ent_reg = F.mse_loss(emb, plm_proj)
            loss = loss + ent_reg_w * ent_reg
            plm_logs['plm_entity_reg'] = ent_reg.item()

        rel_reg_w = getattr(args, 'plm_relation_reg_weight', 0.0)
        if rel_reg_w > 0 and getattr(base_model, 'plm_relation_vectors', None) is not None:
            rel_ids = torch.unique(positive_sample[:, 1])
            emb = base_model.relation_embedding.index_select(0, rel_ids)
            plm = base_model.plm_relation_vectors.index_select(0, rel_ids)
            plm_proj = base_model.plm_relation_proj(plm)
            rel_reg = F.mse_loss(emb, plm_proj)
            loss = loss + rel_reg_w * rel_reg
            plm_logs['plm_relation_reg'] = rel_reg.item()

        comp_phase_weight = getattr(args, 'composition_phase_weight', 0.0)
        comp_mod_weight = getattr(args, 'composition_modulus_weight', 0.0)
        comp_triples = getattr(base_model, 'composition_rel_triples', None)
        if (comp_phase_weight > 0 or comp_mod_weight > 0) and comp_triples is not None and comp_triples.numel() > 0:
            r1 = comp_triples[:, 0]
            r2 = comp_triples[:, 1]
            r3 = comp_triples[:, 2]
            if comp_phase_weight > 0:
                comp_phase = (relation_phase[r1] + relation_phase[r2] - relation_phase[r3]).abs().mean()
                loss = loss + comp_phase_weight * comp_phase
                relation_phase_logs['composition_phase_penalty'] = comp_phase.item()
            if comp_mod_weight > 0:
                relation_mod = base_model.relation_embedding[:, :rel_phase_start]
                comp_mod = (relation_mod[r1] + relation_mod[r2] - relation_mod[r3]).abs().mean()
                loss = loss + comp_mod_weight * comp_mod
                relation_phase_logs['composition_modulus_penalty'] = comp_mod.item()

        loss = loss / max(1, accumulation_steps)
        loss.backward()

        grad_sq = 0.0
        param_sq = 0.0
        for param in base_model.parameters():
            param_sq += float(param.detach().norm(2).item() ** 2)
            if param.grad is not None:
                grad_sq += float(param.grad.detach().norm(2).item() ** 2)
        grad_norm = math.sqrt(max(grad_sq, 0.0))
        param_norm = math.sqrt(max(param_sq, 0.0))

        if optimizer_step:
            clip_norm = float(getattr(args, 'grad_clip_norm', 0.0) or 0.0)
            if clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), clip_norm)
            optimizer.step()

        log = {
            **regularization_log,
            **hierarchy_log,
            **relation_phase_logs,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
            'grad_norm': grad_norm,
            'param_norm': param_norm
        }
        if kd_logs:
            log.update(kd_logs)
        if alignment_logs:
            log.update(alignment_logs)
        if region_logs:
            log.update(region_logs)
        if hyper_logs:
            log.update(hyper_logs)
        if qa_logs:
            log.update(qa_logs)
        if concept_logs:
            log.update(concept_logs)
        if mos_logs:
            log.update(mos_logs)
        if path_logs:
            log.update(path_logs)

        if getattr(args, 'log_hard_negatives', False) and negative_logits is not None:
            KGEModel._log_hard_negatives(
                positive_sample,
                negative_sample,
                positive_logits,
                negative_logits,
                args,
                mode,
                step
            )
        if (teacher_scores
                and getattr(args, 'teacher_debug_log_path', None)
                and args.teacher_debug_log_path):
            KGEModel._log_teacher_disagreements(
                positive_sample,
                positive_logits,
                teacher_scores,
                args,
                step
            )

        # print modulus and phase weights every 1000 steps
        if base_model.model_name in {'RelatE', 'RelateV', 'CCRelatE', 'BKRelatE', 'ARelatE', 'CCMuRP', 'MuRP'}:
            if step % 1000 == 0:
                logging.info(
                    "Phase Weight (avg): %.4f, Modulus Weight (avg): %.4f",
                    F.softplus(base_model.phase_weight).mean().item(),
                    F.softplus(base_model.modulus_weight).mean().item()
                )




        return log

    @staticmethod
    def _compute_kd_loss(positive_logits, negative_logits, teacher_scores, args,
                        relation_ids=None, kd_weights=None):
        kd_count = teacher_scores.get('neg_count', 0)
        mask = teacher_scores.get('mask')
        if kd_count <= 0 or mask is None or not mask.any():
            return None
        student_logits = torch.cat(
            [positive_logits.view(-1, 1), negative_logits[:, :kd_count]],
            dim=1
        )
        teacher_pos = teacher_scores['positive'].unsqueeze(1).to(student_logits.device)
        teacher_neg = teacher_scores['negative'].to(student_logits.device)
        teacher_logits = torch.cat([teacher_pos, teacher_neg], dim=1)
        mask = mask.to(student_logits.device)
        rel_weights = None
        if relation_ids is not None:
            rel_ids = relation_ids.to(student_logits.device)
            rel_ids = rel_ids[mask]
            if kd_weights is not None and isinstance(kd_weights, torch.Tensor):
                rel_weights = kd_weights.to(student_logits.device).index_select(0, rel_ids)
        student_logits = student_logits[mask]
        teacher_logits = teacher_logits[mask]
        if student_logits.numel() == 0:
            return None

        def _z_norm(tensor):
            mean = tensor.mean(dim=1, keepdim=True)
            std = tensor.std(dim=1, keepdim=True)
            return (tensor - mean) / (std + 1e-6)

        student_norm = _z_norm(student_logits)
        teacher_norm = _z_norm(teacher_logits)
        mse = F.mse_loss(student_norm, teacher_norm, reduction='none').mean(dim=1)
        if rel_weights is not None:
            mse = mse * rel_weights
        return mse.mean()

    @staticmethod
    def _compute_alignment_loss(model, positive_sample, negative_sample, args):
        teacher_vectors = getattr(model, 'teacher_entity_vectors', None)
        teacher_mask = getattr(model, 'teacher_alignment_mask', None)
        projector = getattr(model, 'teacher_projector', None)
        if teacher_vectors is None or projector is None:
            return None
        entity_ids = [positive_sample[:, 0], positive_sample[:, 2]]
        if negative_sample is not None and negative_sample.numel() > 0:
            entity_ids.append(negative_sample.view(-1))
        entity_ids = torch.unique(torch.cat(entity_ids, dim=0))
        if entity_ids.numel() == 0:
            return None
        entity_ids = entity_ids.to(teacher_vectors.device)
        if teacher_mask is not None:
            mask_vals = teacher_mask.index_select(0, entity_ids)
            entity_ids = entity_ids[mask_vals]
            if entity_ids.numel() == 0:
                return None
        student_vecs = model.entity_embedding.index_select(0, entity_ids)
        if getattr(model, 'entity_embedding_hyper', None) is not None:
            hyper_vecs = model.entity_embedding_hyper.index_select(0, entity_ids)
            student_vecs = torch.cat([student_vecs, hyper_vecs], dim=1)
        teacher_vecs = teacher_vectors.index_select(0, entity_ids).to(student_vecs.device)
        projected = projector(student_vecs)
        return F.mse_loss(projected, teacher_vecs)

    @staticmethod
    def _compute_query_alignment_loss(model, query_embedding, teacher_scores):
        teacher_queries = teacher_scores.get('query_vectors') if teacher_scores else None
        if teacher_queries is None or query_embedding is None:
            return None
        if query_embedding.dim() == 3:
            student_query = query_embedding[:, 0, :]
        else:
            student_query = query_embedding
        mask = teacher_scores.get('mask')
        if mask is not None:
            mask = mask.to(student_query.device)
            student_query = student_query[mask]
            teacher_queries = teacher_queries.to(student_query.device)[mask]
        else:
            teacher_queries = teacher_queries.to(student_query.device)
        if student_query.numel() == 0:
            return None
        projector = getattr(model, 'teacher_query_projector', None)
        if projector is not None:
            student_query = projector(student_query)
        return F.mse_loss(student_query, teacher_queries)

    @staticmethod
    def _log_hard_negatives(positive_sample, negative_sample, positive_logits,
                            negative_logits, args, mode, step):
        if negative_sample is None or negative_logits.dim() <= 1:
            return
        log_path = getattr(args, 'hard_negative_log_path', None)
        if not log_path:
            return
        pos_cpu = positive_sample.detach().cpu()
        neg_cpu = negative_sample.detach().cpu()
        pos_logit = positive_logits.detach().cpu().view(-1)
        neg_logit = negative_logits.detach().cpu()
        if neg_logit.dim() != 2:
            return

        row_count = min(neg_logit.size(0), pos_logit.numel(), pos_cpu.size(0))
        if row_count <= 0:
            return

        if neg_cpu.dim() == 1:
            neg_cpu = neg_cpu.view(row_count, -1)
        elif neg_cpu.dim() >= 2 and neg_cpu.size(0) != row_count:
            neg_cpu = neg_cpu[:row_count]

        if neg_cpu.dim() < 2:
            return

        col_count = min(neg_logit.size(1), neg_cpu.size(1))
        if col_count <= 0:
            return

        pos_cpu = pos_cpu[:row_count]
        pos_logit = pos_logit[:row_count]
        neg_logit = neg_logit[:row_count, :col_count]
        neg_cpu = neg_cpu[:row_count, :col_count]

        mask = neg_logit > pos_logit.unsqueeze(1)
        if not mask.any():
            return
        indices = mask.nonzero(as_tuple=False)
        if indices.numel() == 0:
            return
        limit = max(1, getattr(args, 'hard_negative_log_limit', 10))
        id2entity = getattr(args, 'id2entity', {})
        id2relation = getattr(args, 'id2relation', {})
        entries = []
        candidate_role = 'head' if mode == 'head-batch' else 'tail'
        for row, col in indices.tolist():
            if len(entries) >= limit:
                break
            head_id = int(pos_cpu[row, 0].item())
            rel_id = int(pos_cpu[row, 1].item())
            tail_id = int(pos_cpu[row, 2].item())
            neg_id = int(neg_cpu[row, col].item())
            entry = {
                'step': int(step),
                'mode': mode,
                'head_id': head_id,
                'head': id2entity.get(head_id, str(head_id)),
                'relation_id': rel_id,
                'relation': id2relation.get(rel_id, str(rel_id)),
                'tail_id': tail_id,
                'tail': id2entity.get(tail_id, str(tail_id)),
                'candidate_role': candidate_role,
                'negative_id': neg_id,
                'negative': id2entity.get(neg_id, str(neg_id)),
                'positive_logit': float(pos_logit[row].item()),
                'negative_logit': float(neg_logit[row, col].item())
            }
            entries.append(entry)
        if entries:
            dir_name = os.path.dirname(log_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(log_path, 'a', encoding='utf-8') as fout:
                for entry in entries:
                    fout.write(json.dumps(entry) + '\n')

    @staticmethod
    def _log_teacher_disagreements(positive_sample, positive_logits, teacher_scores, args, step):
        log_path = getattr(args, 'teacher_debug_log_path', None)
        if not log_path:
            return
        mask = teacher_scores.get('mask')
        if mask is None or not mask.any():
            return
        mask = mask.to(positive_logits.device)
        if not mask.any():
            return
        teacher_pos = teacher_scores['positive'].to(positive_logits.device)
        student_pos = positive_logits.view(-1)
        valid_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if valid_indices.numel() == 0:
            return
        diff = teacher_pos - student_pos
        diff = diff[valid_indices]
        threshold = getattr(args, 'teacher_debug_threshold', 1.0)
        keep_mask = diff > threshold
        if not keep_mask.any():
            return
        diff = diff[keep_mask]
        valid_indices = valid_indices[keep_mask]
        limit = max(1, min(getattr(args, 'teacher_debug_limit', 10), diff.numel()))
        values, order = torch.topk(diff, k=limit)
        selected_indices = valid_indices[order]
        teacher_selected = teacher_pos[selected_indices]
        student_selected = student_pos[selected_indices]
        pos_cpu = positive_sample.detach().cpu()
        id2entity = getattr(args, 'id2entity', {})
        id2relation = getattr(args, 'id2relation', {})
        entries = []
        for idx in range(limit):
            global_idx = int(selected_indices[idx].item())
            head_id = int(pos_cpu[global_idx, 0].item())
            rel_id = int(pos_cpu[global_idx, 1].item())
            tail_id = int(pos_cpu[global_idx, 2].item())
            entry = {
                'step': int(step),
                'mode': 'tail-batch',
                'head_id': head_id,
                'head': id2entity.get(head_id, str(head_id)),
                'relation_id': rel_id,
                'relation': id2relation.get(rel_id, str(rel_id)),
                'tail_id': tail_id,
                'tail': id2entity.get(tail_id, str(tail_id)),
                'teacher_logit': float(teacher_selected[idx].item()),
                'student_logit': float(student_selected[idx].item()),
                'logit_gap': float(values[idx].item())
            }
            entries.append(entry)
        if entries:
            dir_name = os.path.dirname(log_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(log_path, 'a', encoding='utf-8') as fout:
                for entry in entries:
                    fout.write(json.dumps(entry) + '\n')
    
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
                y_score_out = model(sample)
                if isinstance(y_score_out, tuple):
                    y_score_out = y_score_out[0]
                y_score = y_score_out.squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            prediction_logs = [] if getattr(args, 'log_eval_predictions', False) else None
            relation_stats = defaultdict(lambda: {'count': 0, 'rr_sum': 0.0,
                                                  'hit1': 0, 'hit3': 0, 'hit10': 0})
            id2entity = getattr(args, 'id2entity', {})
            id2relation = getattr(args, 'id2relation', {})
            recorded_topk = max(1, getattr(args, 'eval_topk', 5))
            def _worker_init_logging(_):
                if multiprocessing.current_process().name != 'MainProcess':
                    logging.disable(logging.CRITICAL)

            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                # Allow num_workers=0 to avoid multiprocessing when semaphores are disallowed
                num_workers=max(0, args.cpu_num//2),
                worker_init_fn=_worker_init_logging,
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
                num_workers=max(0, args.cpu_num//2),
                worker_init_fn=_worker_init_logging,
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

                        score_out = model((positive_sample, negative_sample), mode)
                        if isinstance(score_out, tuple):
                            score = score_out[0]
                        else:
                            score = score_out
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
                            relation_id = int(positive_sample[i, 1].item())
                            rel_bucket = relation_stats[relation_id]
                            rel_bucket['count'] += 1
                            rel_bucket['rr_sum'] += 1.0 / ranking
                            rel_bucket['hit1'] += 1 if ranking <= 1 else 0
                            rel_bucket['hit3'] += 1 if ranking <= 3 else 0
                            rel_bucket['hit10'] += 1 if ranking <= 10 else 0

                            logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                                'HITS@20': 1.0 if ranking <= 20 else 0.0,
                            })
                            if prediction_logs is not None:
                                top_cut = min(recorded_topk, argsort.size(1))
                                top_ids = argsort[i, :top_cut].cpu().tolist()
                                predictions = []
                                for pred_id in top_ids:
                                    predictions.append({
                                        'entity_id': int(pred_id),
                                        'entity': id2entity.get(pred_id, str(pred_id)),
                                        'score': float(score[i, pred_id].item())
                                    })
                                head_id = int(positive_sample[i, 0].item())
                                tail_id = int(positive_sample[i, 2].item())
                                entry = {
                                    'mode': mode,
                                    'prediction_role': 'head' if mode == 'head-batch' else 'tail',
                                    'head_id': head_id,
                                    'head': id2entity.get(head_id, str(head_id)),
                                    'relation_id': relation_id,
                                    'relation': id2relation.get(relation_id, str(relation_id)),
                                    'tail_id': tail_id,
                                    'tail': id2entity.get(tail_id, str(tail_id)),
                                    'rank': ranking,
                                    'top_predictions': predictions
                                }
                                prediction_logs.append(entry)

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

            if prediction_logs is not None and getattr(args, 'eval_predictions_path', None):
                path = args.eval_predictions_path
                if path:
                    dir_name = os.path.dirname(path)
                    if dir_name:
                        os.makedirs(dir_name, exist_ok=True)
                    with open(path, 'w', encoding='utf-8') as fout:
                        for entry in prediction_logs:
                            fout.write(json.dumps(entry) + '\n')
                    logging.info('Wrote evaluation predictions to %s', path)

            if relation_stats and getattr(args, 'relation_metrics_path', None):
                rel_metrics = {}
                for rel_id, agg in relation_stats.items():
                    if agg['count'] == 0:
                        continue
                    rel_metrics[str(rel_id)] = {
                        'relation': id2relation.get(rel_id, str(rel_id)),
                        'count': agg['count'],
                        'MRR': agg['rr_sum'] / agg['count'],
                        'HITS@1': agg['hit1'] / agg['count'],
                        'HITS@3': agg['hit3'] / agg['count'],
                        'HITS@10': agg['hit10'] / agg['count'],
                    }
                if rel_metrics:
                    path = args.relation_metrics_path
                    if path:
                        dir_name = os.path.dirname(path)
                        if dir_name:
                            os.makedirs(dir_name, exist_ok=True)
                    with open(path, 'w', encoding='utf-8') as fout:
                        json.dump(rel_metrics, fout, indent=2)
                    logging.info('Wrote relation-level metrics to %s', path)

        return metrics
