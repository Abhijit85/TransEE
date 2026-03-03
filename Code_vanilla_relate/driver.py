
#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import math
import os
import random
import multiprocessing
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv

from torch.utils.data import DataLoader, WeightedRandomSampler

from model import KGEModel
from teacher_integration import SimKGCTeacher, MuRPTeacher, CSPromTeacher

from dataloader import TrainDataset, PathDataset
from relate_compile import parse_anyburl_rules, build_relation_maps, generate_candidates_with_rules
from dataloader import BidirectionalOneShotIterator

from torch.optim.lr_scheduler import CosineAnnealingLR

load_dotenv()


class LookaheadOptimizer(torch.optim.Optimizer):
    def __init__(self, base_optimizer, la_steps=5, la_alpha=0.5):
        self.optimizer = base_optimizer
        self.param_groups = self.optimizer.param_groups
        self.defaults = self.optimizer.defaults
        self.la_steps = max(1, int(la_steps))
        self.la_alpha = float(la_alpha)
        self._step = 0
        self._slow_weights = [
            [p.detach().clone() for p in group['params']]
            for group in self.param_groups
        ]

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict(),
            'la_steps': self.la_steps,
            'la_alpha': self.la_alpha,
            'step': self._step,
            'slow_weights': self._slow_weights
        }

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.la_steps = state_dict.get('la_steps', self.la_steps)
        self.la_alpha = state_dict.get('la_alpha', self.la_alpha)
        self._step = state_dict.get('step', self._step)
        slow = state_dict.get('slow_weights')
        if slow is not None:
            self._slow_weights = slow

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self._step += 1
        if self._step % self.la_steps == 0:
            for group_idx, group in enumerate(self.param_groups):
                for p_idx, param in enumerate(group['params']):
                    if param.grad is None:
                        continue
                    slow = self._slow_weights[group_idx][p_idx].to(param.device)
                    slow.add_(self.la_alpha * (param.data - slow))
                    param.data.copy_(slow)
                    self._slow_weights[group_idx][p_idx] = slow.detach().cpu()
        return loss


class EMAHelper:
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()
        self.backup = {}

    def update(self, model):
        one_minus = 1.0 - self.decay
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = param.detach().clone()
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=one_minus)

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self.shadow:
                continue
            self.backup[name] = param.detach().clone()
            param.data.copy_(self.shadow[name].to(param.device))

    def restore(self, model):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name].to(param.device))
        self.backup = {}

def parse_args(args=None):
    def _parse_int_list(value):
        if not value:
            return None
        tokens = value.replace(',', ' ').split()
        parsed = [int(token) for token in tokens if token]
        return parsed if parsed else None

    def _parse_relation_weights(value):
        if not value:
            return {}
        weights = {}
        tokens = value.replace(',', ' ').split()
        for token in tokens:
            if ':' not in token:
                continue
            rel_id, weight = token.split(':', 1)
            try:
                rel_idx = int(rel_id.strip())
                rel_weight = float(weight.strip())
                weights[rel_idx] = rel_weight
            except ValueError:
                continue
        return weights

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
    parser.add_argument('--weight_decay', default=5e-6, type=float,
                        help='Optimizer weight decay (L2) applied to parameters (default: 5e-6)')
    parser.add_argument('--grad_clip_norm', default=0.0, type=float,
                        help='Clip gradient norm before optimizer step (0 disables)')
    parser.add_argument('--optimizer', default='adamw', choices=['adamw'],
                        help='Optimizer type for this vanilla RelatE fork')
    parser.add_argument('--use_lookahead', action='store_true',
                        help='Wrap optimizer with Lookahead')
    parser.add_argument('--lookahead_steps', type=int, default=5,
                        help='Lookahead sync period')
    parser.add_argument('--lookahead_alpha', type=float, default=0.5,
                        help='Lookahead interpolation alpha')
    parser.add_argument('--use_ema', action='store_true',
                        help='Track EMA of model parameters')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='EMA decay for model parameters')
    parser.add_argument('--eval_with_ema', action='store_true',
                        help='Evaluate with EMA weights when EMA is enabled')
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')
    parser.add_argument('--full_ranking_ce', action='store_true',
                        help='Use full-ranking cross-entropy over all entities instead of sampled negatives')
    parser.add_argument('--full_ranking_chunk_size', default=2048, type=int,
                        help='Chunk size for scoring all entities in full-ranking CE mode')
    parser.add_argument('--full_ranking_label_smoothing', default=0.0, type=float,
                        help='Optional label smoothing for full-ranking CE (0 disables)')
    
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('--lr_t_max', default=None, type=int,
                        help='Optional cosine scheduler period; defaults to max_steps')
    parser.add_argument('--lr_eta_min', default=1e-5, type=float,
                        help='Minimum learning rate for cosine scheduler decay')
    parser.add_argument('--lr_drop_steps', type=int, nargs='+', default=None,
                        help='Optional manual LR drop steps (training iterations)')
    parser.add_argument('--lr_drop_gamma', type=float, default=0.5,
                        help='Multiplicative factor applied at each manual drop step')
    parser.add_argument('--hard_negative_steps', type=int, nargs='+', default=None,
                        help='Training steps at which to multiply negative sample size for harder negatives')
    parser.add_argument('--hard_negative_multiplier', type=float, default=1.5,
                        help='Multiplier applied to negative sample size at each hard-negative step')
    parser.add_argument('--max_negative_sample_size', type=int, default=None,
                        help='Optional upper bound on negative sample size after hard-negative boosts')
    parser.add_argument('--plateau_lr_start_step', type=int, default=8000,
                        help='Start applying validation-plateau LR drops once this step is reached')
    parser.add_argument('--plateau_lr_patience', type=int, default=2,
                        help='Number of consecutive non-improving validations before halving LR')
    parser.add_argument('--plateau_lr_factor', type=float, default=0.5,
                        help='Factor to multiply LR by when plateau trigger fires')
    parser.add_argument('--plateau_lr_min', type=float, default=5e-5,
                        help='Lower bound on LR when plateau drops are applied')
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
    parser.add_argument('--skip_optimizer_state', action='store_true',
                        help='When resuming from a checkpoint, skip loading the optimizer state')
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    parser.add_argument('--warm_up_factor', type=float, default=0.1,
                        help='Multiplicative LR drop applied at each warm-up milestone (default 0.1).')
    parser.add_argument('--warm_up_multiplier', type=float, default=3.0,
                        help='Factor to scale warm-up interval after each drop (default 3.0).')
    
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=5000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')


    parser.add_argument('-eras','--use_eras', action='store_true', help='Enable ERAS for RelatE')
    parser.add_argument('--k_prototypes', default=4, type=int, help='Number of ERAS prototypes')

    parser.add_argument('--init_modulus_weight', type=float, default=2.5,help='Initial value for RelatE modulus weight (default: 3.5)')
    parser.add_argument('--phase_weight_scale', type=float, default=0.65,
                        help='Multiplier applied to init_modulus_weight when initializing phase weights')
    parser.add_argument('--use_region_head', action='store_true',
                        help='Enable auxiliary region/box head for hierarchical or meronymic relations')
    parser.add_argument('--region_dim', type=int, default=64,
                        help='Dimensionality of region embeddings')
    parser.add_argument('--region_blend_weight', type=float, default=0.5,
                        help='Blend factor between RelatE base score and region containment score')
    parser.add_argument('--region_blend_final_weight', type=float, default=None,
                        help='Final blend weight after warmup (defaults to region_blend_weight)')
    parser.add_argument('--region_blend_warmup_steps', type=int, default=0,
                        help='Number of steps to ramp the region blend weight to its final value')
    parser.add_argument('--region_volume_penalty', type=float, default=0.0,
                        help='Weight applied to the average region extent to discourage overly large boxes')
    parser.add_argument('--region_depth_weight', type=float, default=0.0,
                        help='Penalty weight enforcing parent regions to be at least as large as child regions on hierarchical relations')
    parser.add_argument('--region_depth_margin', type=float, default=0.0,
                        help='Margin applied when comparing parent/child region extents')
    parser.add_argument('--use_hyper_subspace', action='store_true',
                        help='Enable auxiliary hyperbolic subspace for hybrid RelatE scoring')
    parser.add_argument('--hyper_dim', type=int, default=0,
                        help='Dimensionality of the optional hyperbolic subspace (0 disables it)')
    parser.add_argument('--hyper_blend_warmup_steps', type=int, default=0,
                        help='Warmup steps for the hyper score blend factor (0 disables)')
    parser.add_argument('--hyper_radius_weight', type=float, default=0.0,
                        help='Weight for enforcing parent hyperball radius >= child radius')
    parser.add_argument('--hyper_radius_margin', type=float, default=0.0,
                        help='Margin applied to the hyperball radius constraint')
    parser.add_argument('--hyper_radius_warmup_steps', type=int, default=0,
                        help='Warmup steps for the hyper radius penalty (0 disables)')



    # Type constraints
    parser.add_argument('--type_map_path', type=str, default=None, help='Path to entity-type map JSON file')
    parser.add_argument('--type_lambda', type=float, default=1.0,help='Scaling factor for type bias injection (default 1.0)')
    parser.add_argument('--init_rel_width', type=float, default=0.1,help='Initial value for relation-specific slope (default: 0.1)')
    parser.add_argument('--modulus_sharpness', type=float, default=1.0, help='Exponent on modulus distance to sharpen scores (>1 increases top-rank separation)')
    parser.add_argument('--phase_sharpness', type=float, default=1.0, help='Exponent on phase component to sharpen scores (>1 increases top-rank separation)')

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
    parser.add_argument('--hierarchy_depth_path', type=str, default=None, help='JSON mapping entity -> depth for hierarchy regularizer')
    parser.add_argument('--depth_penalty_weight', type=float, default=0.0,
                        help='Weight for enforcing parent modulus >= child modulus')
    parser.add_argument('--depth_penalty_margin', type=float, default=0.05,
                        help='Margin applied when comparing parent/child modulus norms')
    parser.add_argument('--depth_penalty_scale_gap', action='store_true',
                        help='Scale the margin by the absolute hierarchy depth difference')
    parser.add_argument('--relation_behavior_path', type=str, default=None,
                        help='JSON describing relation symmetry/antisymmetry/inverse pairs')
    parser.add_argument('--entity_concept_map_path', type=str, default=None,
                        help='JSON mapping entity string -> concept string/id')
    parser.add_argument('--concept_depth_map_path', type=str, default=None,
                        help='Optional JSON mapping concept string/id -> depth value')
    parser.add_argument('--concept_phase_weight', type=float, default=0.0,
                        help='Weight for phase compactness among entities sharing the same concept')
    parser.add_argument('--concept_modulus_weight', type=float, default=0.0,
                        help='Weight for aligning entity modulus norms to concept depth')
    parser.add_argument('--concept_relation_weight', type=float, default=0.0,
                        help='Weight for relation-level concept consistency penalties')
    parser.add_argument('--concept_depth_margin', type=float, default=0.0,
                        help='Margin for concept depth ordering on hierarchical relations')
    parser.add_argument('--cc_concept_weight', type=float, default=0.0,
                        help='CCRelatE: weight for concept-prototype calibration residual')
    parser.add_argument('--cc_depth_weight', type=float, default=0.0,
                        help='CCRelatE: weight for concept-depth residual on hierarchical/meronymic relations')
    parser.add_argument('--cc_relation_weight', type=float, default=0.0,
                        help='CCRelatE: weight for relation-conditioned concept transition residual')
    parser.add_argument('--symmetric_phase_weight', type=float, default=0.0,
                        help='Penalty weight to keep symmetric relations near zero phase')
    parser.add_argument('--antisymmetric_phase_weight', type=float, default=0.0,
                        help='Penalty weight pushing antisymmetric relations toward pi phase')
    parser.add_argument('--inverse_phase_weight', type=float, default=0.0,
                        help='Penalty weight tying inverse relation phase offsets')
    parser.add_argument('--add_reciprocals', action='store_true',
                        help='Append reciprocal triples with relation offset like RotatE')
    # Text-aware relation prompts
    parser.add_argument('--use_rel_prompt_emb', action='store_true',
                        help='Augment relation embeddings with frozen prompt vectors')
    parser.add_argument('--rel_prompt_path', type=str, default=None,
                        help='Path to a NumPy .npy file containing relation prompt embeddings')
    parser.add_argument('--rel_prompt_weight', type=float, default=0.0,
                        help='Initial blend weight for prompt projections (0 disables)')
    parser.add_argument('--rel_prompt_warmup_steps', type=int, default=0,
                        help='Warmup steps to ramp the prompt blend from 0 to the configured weight')
    parser.add_argument('--use_entity_prompt_emb', action='store_true',
                        help='Blend teacher entity vectors directly into RelatE embeddings')
    parser.add_argument('--entity_prompt_weight', type=float, default=0.0,
                        help='Initial blend weight for entity prompt projections')
    parser.add_argument('--entity_prompt_warmup_steps', type=int, default=0,
                        help='Warmup steps for entity prompt blend')
    parser.add_argument('--kd_warmup_steps', type=int, default=0,
                        help='Warmup steps to scale KD lambda from 0 to full strength')
    parser.add_argument('--kd_decay_start', type=int, default=None,
                        help='Step at which to start decaying KD weight to zero')
    parser.add_argument('--kd_decay_duration', type=int, default=0,
                        help='Number of steps over which KD decays to zero once decay starts')
    parser.add_argument('--relation_loss_weights', type=str, default=None,
                        help='Optional relation_id or relation_name to weight (format id:weight or name:weight, comma/space separated)')
    parser.add_argument('--teacher_type', choices=['simkgc', 'murp', 'csprom'], default=None,
                        help='Teacher model type used for distillation')
    parser.add_argument('--teacher_checkpoint', type=str, default=None,
                        help='Path to the trained teacher checkpoint')
    parser.add_argument('--teacher_repo', type=str, default='teachers/simkgc_repo',
                        help='Path to the SimKGC repository for teacher loading')
    parser.add_argument('--murp_repo', type=str, default='teachers/murp',
                        help='Path to the MuRP repository for hyperbolic teacher loading')
    parser.add_argument('--murp_data_dir', type=str, default=None,
                        help='MuRP data directory (defaults to <murp_repo>/data/<DATASET>)')
    parser.add_argument('--csprom_repo', type=str, default='teachers/csprom_kg',
                        help='Path to the CSProm-KG repository for teacher loading')
    parser.add_argument('--csprom_dataset', type=str, default=None,
                        help='Dataset name for CSProm-KG (defaults from data_path name)')
    parser.add_argument('--csprom_data_dir', type=str, default=None,
                        help='CSProm-KG processed dataset directory (defaults to <csprom_repo>/data/processed/<DATASET>)')
    parser.add_argument('--csprom_config', type=str, default=None,
                        help='Optional JSON config with CSProm-KG hyperparameters (overrides checkpoint)')
    parser.add_argument('--teacher_negatives', type=int, default=64,
                        help='Number of negatives scored by the teacher per batch (tail-batch only)')
    parser.add_argument('--teacher_device', type=str, default='cuda',
                        help='Device for teacher inference (cuda or cpu)')
    parser.add_argument('--kd_lambda', type=float, default=0.0,
                        help='Weight on the teacher distillation loss component')
    parser.add_argument('--kd_loss', choices=['mse'], default='mse',
                        help='Distillation loss type')
    parser.add_argument('--kd_relation_weights', type=str, default=None,
                        help='Optional per-relation KD multipliers (format rel:weight, comma/space separated)')
    parser.add_argument('--kd_hyper_weight', type=float, default=1.0,
                        help='Scaling factor for the hyperbolic KD component')
    parser.add_argument('--hyper_kd_warmup_steps', type=int, default=0,
                        help='Warmup steps for the hyperbolic KD component (0 disables)')
    parser.add_argument('--teacher_query_align_weight', type=float, default=0.0,
                        help='Weight on aligning RelatE query embeddings with teacher-composed queries')
    # RELATE-Compile initialization (training-light fine-tune)
    parser.add_argument('--compiled_init_dir', type=str, default=None,
                        help='Directory containing compiled entity/relation phase/modulus .npy files')
    parser.add_argument('--use_mos_head', action='store_true',
                        help='Enable Mixture-of-Softmax (KGE-MoS) output head')
    parser.add_argument('--mos_components', type=int, default=4,
                        help='Number of softmax components in MoS head')
    parser.add_argument('--mos_hidden_dim', type=int, default=256,
                        help='Hidden size for MoS projection networks')
    parser.add_argument('--mos_entropy_weight', type=float, default=0.0,
                        help='Weight on MoS mixture entropy regularizer')
    parser.add_argument('--use_hyperbolic_relate', action='store_true',
                        help='Replace RelatE geometry with a hyperbolic (Poincaré) variant')
    parser.add_argument('--hyperbolic_curvature', type=float, default=1.0,
                        help='Curvature (positive) for the Poincaré ball when using hyperbolic RelatE')
    parser.add_argument('--use_relation_gate', action='store_true',
                        help='Enable relation-gated phase/modulus mixing in RelatE')
    parser.add_argument('--use_type_mod_norm', action='store_true',
                        help='Enable type-aware modulus normalization in RelatE')
    parser.add_argument('--use_hierarchy_mod_head', action='store_true',
                        help='Enable hierarchy-aware modulus head in RelatE')
    parser.add_argument('--composition_phase_weight', type=float, default=0.0,
                        help='Weight for composition phase regularizer')
    parser.add_argument('--composition_modulus_weight', type=float, default=0.0,
                        help='Weight for composition modulus regularizer')
    parser.add_argument('--use_query_adaptive', action='store_true',
                        help='Enable query-conditioned adaptive scoring in RelatE')
    parser.add_argument('--qa_hidden_dim', type=int, default=256,
                        help='Hidden size for query-adaptive gating')
    parser.add_argument('--qa_num_experts', type=int, default=4,
                        help='Number of relation-aware experts in query-adaptive scoring')
    parser.add_argument('--qa_temperature_floor', type=float, default=0.5,
                        help='Minimum temperature for query-adaptive scaling')
    parser.add_argument('--qa_temperature_ceiling', type=float, default=2.5,
                        help='Maximum temperature for query-adaptive scaling')
    parser.add_argument('--qa_prototype_weight', type=float, default=0.0,
                        help='Weight of concept-prototype score correction')
    parser.add_argument('--qa_contrastive_weight', type=float, default=0.0,
                        help='Weight of query contrastive objective')
    parser.add_argument('--qa_contrastive_temp', type=float, default=0.07,
                        help='Temperature for query contrastive loss')
    parser.add_argument('--plm_entity_emb_path', type=str, default=None,
                        help='Path to numpy .npy for PLM entity embeddings')
    parser.add_argument('--plm_relation_emb_path', type=str, default=None,
                        help='Path to numpy .npy for PLM relation embeddings')
    parser.add_argument('--plm_entity_reg_weight', type=float, default=0.0,
                        help='PLM entity regularizer weight')
    parser.add_argument('--plm_relation_reg_weight', type=float, default=0.0,
                        help='PLM relation regularizer weight')
    parser.add_argument('--plm_teacher', action='store_true',
                        help='Use PLM vectors as a pseudo-teacher for KD')
    parser.add_argument('--plm_teacher_temperature', type=float, default=1.0,
                        help='Temperature for PLM pseudo-teacher scores')
    parser.add_argument('--plm_teacher_cosine', action='store_true',
                        help='Use cosine similarity for PLM pseudo-teacher scores')
    parser.add_argument('--hier_contrastive_weight', type=float, default=0.0,
                        help='Weight for hierarchy contrastive loss')
    parser.add_argument('--hier_contrastive_temp', type=float, default=0.07,
                        help='Temperature for hierarchy contrastive loss')
    parser.add_argument('--hier_contrastive_phase_only', action='store_true',
                        help='Use phase-only embeddings for hierarchy contrastive loss')
    parser.add_argument('--hier_depth_negatives', type=int, default=16,
                        help='Number of depth-matched negatives per hierarchy example')
    parser.add_argument('--mixkg_enable', action='store_true',
                        help='Enable MixKG-style mixed negatives')
    parser.add_argument('--mixkg_topk', type=int, default=64,
                        help='Top-K hard negatives to consider for MixKG')
    parser.add_argument('--mixkg_mix_count', type=int, default=32,
                        help='Number of mixed negatives per batch item')
    parser.add_argument('--mixkg_alpha', type=float, default=0.5,
                        help='Beta distribution alpha for mixing')
    parser.add_argument('--mixkg_use_similarity', action='store_true',
                        help='Use similarity to positive when selecting MixKG candidates')
    parser.add_argument('--mixkg_score_weight', type=float, default=0.5,
                        help='Weight for score vs similarity when ranking MixKG candidates')
    parser.add_argument('--teacher_align_weight', type=float, default=0.0,
                        help='Weight for aligning student entity embeddings with teacher embeddings')
    parser.add_argument('--hard_negative_fraction', type=float, default=0.5,
                        help='Fraction of negatives drawn from relation pools for hierarchy relations')
    parser.add_argument('--structural_negative_fraction', type=float, default=0.25,
                        help='Extra fraction of negatives sampled from head-specific structural caches')
    parser.add_argument('--structural_negative_size', type=int, default=256,
                        help='Number of structural candidates retained per head')
    parser.add_argument('--train_anyburl_rules', type=str, default=None,
                        help='Path to AnyBURL rules file for candidate negative sampling')
    parser.add_argument('--candidate_negative_fraction', type=float, default=0.0,
                        help='Fraction of negatives drawn from AnyBURL candidate cache (tail-batch)')
    parser.add_argument('--candidate_rule_topk', type=int, default=3000,
                        help='Rule-only top-k cap per (head, relation) when building candidate cache')
    parser.add_argument('--candidate_fallback_topk', type=int, default=200,
                        help='Fallback top-k per relation when building candidate cache')
    parser.add_argument('--candidate_cache_max', type=int, default=9000,
                        help='Max candidate set size per (head, relation) in cache')
    parser.add_argument('--emu_negative_fraction', type=float, default=0.0,
                        help='Fraction of negatives drawn from EMU-style graph-walk caches')
    parser.add_argument('--emu_num_walks', type=int, default=4,
                        help='Number of random walks per query key when building EMU cache')
    parser.add_argument('--emu_walk_length', type=int, default=3,
                        help='Length of each random walk when building EMU cache')
    parser.add_argument('--emu_cache_size', type=int, default=512,
                        help='Maximum EMU candidate set size per query key')
    parser.add_argument('--emu_relation_quota', type=int, default=128,
                        help='Max relation-matched entities injected into each EMU cache key')
    parser.add_argument('--ramp_start_step', type=int, default=0,
                        help='Start step for random-init ramp-up (0 disables)')
    parser.add_argument('--ramp_end_step', type=int, default=0,
                        help='End step for random-init ramp-up (0 disables)')
    parser.add_argument('--phase_weight_scale_target', type=float, default=None,
                        help='Target PHASE_WEIGHT_SCALE at ramp end')
    parser.add_argument('--phase_sharpness_target', type=float, default=None,
                        help='Target PHASE_SHARPNESS at ramp end')
    parser.add_argument('--modulus_sharpness_target', type=float, default=None,
                        help='Target MODULUS_SHARPNESS at ramp end')
    parser.add_argument('--adversarial_temperature_target', type=float, default=None,
                        help='Target ADVERSARIAL_TEMPERATURE at ramp end')
    parser.add_argument('--negative_sample_size_target', type=int, default=None,
                        help='Target NEGATIVE_SAMPLE_SIZE at ramp end')
    parser.add_argument('--path_loss_weight_target', type=float, default=None,
                        help='Target PATH_LOSS_WEIGHT at ramp end')
    parser.add_argument('--relation_sampling_weights', type=str, default=None,
                        help='Optional per-relation sampling weights (format rel:weight or path to JSON map)')
    parser.add_argument('--extra_hard_relations', type=int, nargs='+', default=None,
                        help='Additional relation IDs to include in hard-negative mining')
    parser.add_argument('--log_eval_predictions', action='store_true',
                        help='Dump per-triple evaluation predictions/top-k rankings')
    parser.add_argument('--eval_predictions_path', type=str, default=None,
                        help='Optional path to write evaluation predictions (JSON lines)')
    parser.add_argument('--eval_topk', type=int, default=5,
                        help='Number of top predictions to record per example')
    parser.add_argument('--relation_metrics_path', type=str, default=None,
                        help='Optional path to write relation-level evaluation metrics (JSON)')
    parser.add_argument('--log_hard_negatives', action='store_true',
                        help='Log hard negatives whose scores exceed positives during training')
    parser.add_argument('--hard_negative_log_path', type=str, default='hard_negatives.jsonl',
                        help='File to append hard-negative debug entries')
    parser.add_argument('--hard_negative_log_limit', type=int, default=10,
                        help='Maximum hard-negative records per training step')
    parser.add_argument('--teacher_debug_log_path', type=str, default=None,
                        help='Optional JSONL file to record teacher/student disagreements')
    parser.add_argument('--teacher_debug_limit', type=int, default=10,
                        help='Maximum disagreement entries per training step')
    parser.add_argument('--teacher_debug_threshold', type=float, default=1.0,
                        help='Minimum (teacher - student) logit gap to log disagreements')
    parser.add_argument('--expose_query_embedding', action='store_true',
                        help='Expose composed query embedding h_{s,r} for advanced heads (e.g., MoS)')
    parser.add_argument('--murp_style_reporting', action='store_true',
                        help='Report test metrics during training (MuRP-style) and bypass strict valid-based model selection')
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
    env_cpu = os.getenv('CPU_NUM')
    if env_cpu:
        parsed_args.cpu_num = int(env_cpu)
    if parsed_args.gradient_accumulation_steps < 1:
        parsed_args.gradient_accumulation_steps = 1

    # Optional sharpening and sampling overrides from environment
    env_neg_size = os.getenv('NEGATIVE_SAMPLE_SIZE')
    if env_neg_size and parsed_args.negative_sample_size == 128:
        parsed_args.negative_sample_size = int(env_neg_size)

    env_sampling_weights = os.getenv('RELATION_SAMPLING_WEIGHTS') or os.getenv('RELATION_SAMPLING_WEIGHTS_PATH')
    if env_sampling_weights and parsed_args.relation_sampling_weights is None:
        parsed_args.relation_sampling_weights = env_sampling_weights

    env_extra_hard = os.getenv('EXTRA_HARD_RELATIONS')
    if env_extra_hard and parsed_args.extra_hard_relations is None:
        parsed_args.extra_hard_relations = _parse_int_list(env_extra_hard)

    env_adv_temp = os.getenv('ADVERSARIAL_TEMPERATURE')
    if env_adv_temp and parsed_args.adversarial_temperature == 1.0:
        parsed_args.adversarial_temperature = float(env_adv_temp)
    env_full_rank = os.getenv('FULL_RANKING_CE')
    if env_full_rank:
        parsed_args.full_ranking_ce = env_full_rank.lower() in {'1', 'true', 'yes', 'y'}
    env_full_rank_chunk = os.getenv('FULL_RANKING_CHUNK_SIZE')
    if env_full_rank_chunk:
        parsed_args.full_ranking_chunk_size = int(env_full_rank_chunk)
    env_full_rank_smooth = os.getenv('FULL_RANKING_LABEL_SMOOTHING')
    if env_full_rank_smooth:
        parsed_args.full_ranking_label_smoothing = float(env_full_rank_smooth)

    if os.getenv('USE_ADVERSARIAL_SAMPLING', '').lower() in {'1', 'true', 'yes'}:
        parsed_args.negative_adversarial_sampling = True

    env_mod_sharp = os.getenv('MODULUS_SHARPNESS')
    env_phase_sharp = os.getenv('PHASE_SHARPNESS')
    if env_mod_sharp and parsed_args.modulus_sharpness == 1.0:
        parsed_args.modulus_sharpness = float(env_mod_sharp)
    if env_phase_sharp and parsed_args.phase_sharpness == 1.0:
        parsed_args.phase_sharpness = float(env_phase_sharp)

    env_reg = os.getenv('REGULARIZATION')
    if env_reg:
        parsed_args.regularization = float(env_reg)

    env_warm_steps = os.getenv('WARM_UP_STEPS')
    if env_warm_steps and parsed_args.warm_up_steps is None:
        parsed_args.warm_up_steps = int(env_warm_steps)

    env_warm_factor = os.getenv('WARM_UP_FACTOR')
    if env_warm_factor:
        parsed_args.warm_up_factor = float(env_warm_factor)

    env_warm_mult = os.getenv('WARM_UP_MULTIPLIER')
    if env_warm_mult:
        parsed_args.warm_up_multiplier = float(env_warm_mult)

    env_weight_decay = os.getenv('WEIGHT_DECAY')
    if env_weight_decay and parsed_args.weight_decay == 5e-6:
        parsed_args.weight_decay = float(env_weight_decay)
    env_grad_clip = os.getenv('GRAD_CLIP_NORM')
    if env_grad_clip:
        parsed_args.grad_clip_norm = float(env_grad_clip)
    if os.getenv('USE_LOOKAHEAD', '').lower() in {'1', 'true', 'yes'}:
        parsed_args.use_lookahead = True
    env_look_steps = os.getenv('LOOKAHEAD_STEPS')
    if env_look_steps:
        parsed_args.lookahead_steps = int(env_look_steps)
    env_look_alpha = os.getenv('LOOKAHEAD_ALPHA')
    if env_look_alpha:
        parsed_args.lookahead_alpha = float(env_look_alpha)
    if os.getenv('USE_EMA', '').lower() in {'1', 'true', 'yes'}:
        parsed_args.use_ema = True
    env_ema_decay = os.getenv('EMA_DECAY')
    if env_ema_decay:
        parsed_args.ema_decay = float(env_ema_decay)
    if os.getenv('EVAL_WITH_EMA', '').lower() in {'1', 'true', 'yes'}:
        parsed_args.eval_with_ema = True

    env_plateau_start = os.getenv('PLATEAU_LR_START_STEP')
    if env_plateau_start and parsed_args.plateau_lr_start_step == 8000:
        parsed_args.plateau_lr_start_step = int(env_plateau_start)

    env_plateau_patience = os.getenv('PLATEAU_LR_PATIENCE')
    if env_plateau_patience and parsed_args.plateau_lr_patience == 2:
        parsed_args.plateau_lr_patience = int(env_plateau_patience)

    env_plateau_factor = os.getenv('PLATEAU_LR_FACTOR')
    if env_plateau_factor and parsed_args.plateau_lr_factor == 0.5:
        parsed_args.plateau_lr_factor = float(env_plateau_factor)

    env_plateau_min = os.getenv('PLATEAU_LR_MIN')
    if env_plateau_min and parsed_args.plateau_lr_min == 5e-5:
        parsed_args.plateau_lr_min = float(env_plateau_min)

    env_hard_neg_steps = os.getenv('HARD_NEGATIVE_STEPS')
    if env_hard_neg_steps and not parsed_args.hard_negative_steps:
        parsed_args.hard_negative_steps = _parse_int_list(env_hard_neg_steps) or parsed_args.hard_negative_steps

    env_hard_neg_mult = os.getenv('HARD_NEGATIVE_MULTIPLIER')
    if env_hard_neg_mult and parsed_args.hard_negative_multiplier == 1.5:
        parsed_args.hard_negative_multiplier = float(env_hard_neg_mult)

    env_max_neg_size = os.getenv('MAX_NEGATIVE_SAMPLE_SIZE')
    if env_max_neg_size:
        parsed_args.max_negative_sample_size = int(env_max_neg_size)

    env_lr_drop_steps = os.getenv('LR_DROP_STEPS')
    if env_lr_drop_steps and not parsed_args.lr_drop_steps:
        parsed_args.lr_drop_steps = _parse_int_list(env_lr_drop_steps) or parsed_args.lr_drop_steps

    if os.getenv('STOP_AT_FIRST_PEAK', '').lower() in {'1', 'true', 'yes'}:
        parsed_args.stop_at_first_peak = True

    if os.getenv('MURP_STYLE_REPORTING', '').lower() in {'1', 'true', 'yes'}:
        parsed_args.murp_style_reporting = True

    if os.getenv('ADD_RECIPROCALS', '').lower() in {'1', 'true', 'yes'}:
        parsed_args.add_reciprocals = True

    env_depth_path = os.getenv('HIERARCHY_DEPTH_PATH')
    if env_depth_path and parsed_args.hierarchy_depth_path is None:
        parsed_args.hierarchy_depth_path = env_depth_path

    env_depth_weight = os.getenv('DEPTH_PENALTY_WEIGHT')
    if env_depth_weight:
        parsed_args.depth_penalty_weight = float(env_depth_weight)

    env_depth_margin = os.getenv('DEPTH_PENALTY_MARGIN')
    if env_depth_margin:
        parsed_args.depth_penalty_margin = float(env_depth_margin)

    if os.getenv('DEPTH_PENALTY_SCALE_GAP', '').lower() in {'1', 'true', 'yes'}:
        parsed_args.depth_penalty_scale_gap = True

    env_rel_behavior = os.getenv('RELATION_BEHAVIOR_PATH')
    if env_rel_behavior and parsed_args.relation_behavior_path is None:
        parsed_args.relation_behavior_path = env_rel_behavior
    env_entity_concepts = os.getenv('ENTITY_CONCEPT_MAP_PATH')
    if env_entity_concepts and parsed_args.entity_concept_map_path is None:
        parsed_args.entity_concept_map_path = env_entity_concepts
    env_concept_depths = os.getenv('CONCEPT_DEPTH_MAP_PATH')
    if env_concept_depths and parsed_args.concept_depth_map_path is None:
        parsed_args.concept_depth_map_path = env_concept_depths
    env_concept_phase_w = os.getenv('CONCEPT_PHASE_WEIGHT')
    if env_concept_phase_w:
        parsed_args.concept_phase_weight = float(env_concept_phase_w)
    env_concept_mod_w = os.getenv('CONCEPT_MODULUS_WEIGHT')
    if env_concept_mod_w:
        parsed_args.concept_modulus_weight = float(env_concept_mod_w)
    env_concept_rel_w = os.getenv('CONCEPT_RELATION_WEIGHT')
    if env_concept_rel_w:
        parsed_args.concept_relation_weight = float(env_concept_rel_w)
    env_concept_margin = os.getenv('CONCEPT_DEPTH_MARGIN')
    if env_concept_margin:
        parsed_args.concept_depth_margin = float(env_concept_margin)
    env_cc_concept_w = os.getenv('CC_CONCEPT_WEIGHT')
    if env_cc_concept_w:
        parsed_args.cc_concept_weight = float(env_cc_concept_w)
    env_cc_depth_w = os.getenv('CC_DEPTH_WEIGHT')
    if env_cc_depth_w:
        parsed_args.cc_depth_weight = float(env_cc_depth_w)
    env_cc_relation_w = os.getenv('CC_RELATION_WEIGHT')
    if env_cc_relation_w:
        parsed_args.cc_relation_weight = float(env_cc_relation_w)

    env_sym_weight = os.getenv('SYMMETRIC_PHASE_WEIGHT')
    if env_sym_weight:
        parsed_args.symmetric_phase_weight = float(env_sym_weight)

    env_anti_weight = os.getenv('ANTISYMMETRIC_PHASE_WEIGHT')
    if env_anti_weight:
        parsed_args.antisymmetric_phase_weight = float(env_anti_weight)

    env_inv_weight = os.getenv('INVERSE_PHASE_WEIGHT')
    if env_inv_weight:
        parsed_args.inverse_phase_weight = float(env_inv_weight)

    env_teacher_checkpoint = os.getenv('TEACHER_CHECKPOINT')
    if env_teacher_checkpoint:
        parsed_args.teacher_checkpoint = env_teacher_checkpoint

    env_teacher_type = os.getenv('TEACHER_TYPE')
    if env_teacher_type:
        parsed_args.teacher_type = env_teacher_type

    env_teacher_repo = os.getenv('TEACHER_REPO')
    if env_teacher_repo:
        parsed_args.teacher_repo = env_teacher_repo
    env_murp_repo = os.getenv('MURP_REPO')
    if env_murp_repo:
        parsed_args.murp_repo = env_murp_repo
    env_murp_data = os.getenv('MURP_DATA_DIR')
    if env_murp_data:
        parsed_args.murp_data_dir = env_murp_data
    env_csprom_repo = os.getenv('CSPROM_REPO')
    if env_csprom_repo:
        parsed_args.csprom_repo = env_csprom_repo
    env_csprom_data = os.getenv('CSPROM_DATA_DIR')
    if env_csprom_data:
        parsed_args.csprom_data_dir = env_csprom_data
    env_csprom_dataset = os.getenv('CSPROM_DATASET')
    if env_csprom_dataset:
        parsed_args.csprom_dataset = env_csprom_dataset
    env_csprom_config = os.getenv('CSPROM_CONFIG')
    if env_csprom_config:
        parsed_args.csprom_config = env_csprom_config

    env_teacher_neg = os.getenv('TEACHER_NEGATIVES')
    if env_teacher_neg:
        parsed_args.teacher_negatives = int(env_teacher_neg)

    env_teacher_device = os.getenv('TEACHER_DEVICE')
    if env_teacher_device:
        parsed_args.teacher_device = env_teacher_device

    env_compiled_init = os.getenv('COMPILED_INIT_DIR')
    if env_compiled_init:
        parsed_args.compiled_init_dir = env_compiled_init

    env_kd_lambda = os.getenv('KD_LAMBDA')
    if env_kd_lambda:
        parsed_args.kd_lambda = float(env_kd_lambda)

    env_kd_loss = os.getenv('KD_LOSS')
    if env_kd_loss:
        parsed_args.kd_loss = env_kd_loss
    env_kd_rel_weights = os.getenv('KD_RELATION_WEIGHTS')
    if env_kd_rel_weights:
        parsed_args.kd_relation_weights = env_kd_rel_weights
    env_kd_hyper_weight = os.getenv('KD_HYPER_WEIGHT')
    if env_kd_hyper_weight:
        parsed_args.kd_hyper_weight = float(env_kd_hyper_weight)
    env_hyper_kd_warmup = os.getenv('HYPER_KD_WARMUP_STEPS')
    if env_hyper_kd_warmup:
        parsed_args.hyper_kd_warmup_steps = int(env_hyper_kd_warmup)
    env_teacher_align = os.getenv('TEACHER_ALIGN_WEIGHT')
    if env_teacher_align:
        parsed_args.teacher_align_weight = float(env_teacher_align)
    env_teacher_query_align = os.getenv('TEACHER_QUERY_ALIGN_WEIGHT')
    if env_teacher_query_align:
        parsed_args.teacher_query_align_weight = float(env_teacher_query_align)
    if parsed_args.teacher_query_align_weight > 0:
        parsed_args.expose_query_embedding = True

    if os.getenv('LOG_EVAL_PREDICTIONS', '').lower() in {'1', 'true', 'yes'}:
        parsed_args.log_eval_predictions = True
    env_eval_path = os.getenv('EVAL_PREDICTIONS_PATH')
    if env_eval_path:
        parsed_args.eval_predictions_path = env_eval_path
    env_eval_topk = os.getenv('EVAL_TOPK')
    if env_eval_topk:
        parsed_args.eval_topk = int(env_eval_topk)
    env_relation_metrics = os.getenv('RELATION_METRICS_PATH')
    if env_relation_metrics:
        parsed_args.relation_metrics_path = env_relation_metrics
    if os.getenv('LOG_HARD_NEGATIVES', '').lower() in {'1', 'true', 'yes'}:
        parsed_args.log_hard_negatives = True
    env_hard_neg_path = os.getenv('HARD_NEGATIVE_LOG_PATH')
    if env_hard_neg_path:
        parsed_args.hard_negative_log_path = env_hard_neg_path
    env_hard_limit = os.getenv('HARD_NEGATIVE_LOG_LIMIT')
    if env_hard_limit:
        parsed_args.hard_negative_log_limit = int(env_hard_limit)
    env_teacher_dbg = os.getenv('TEACHER_DEBUG_LOG_PATH')
    if env_teacher_dbg:
        parsed_args.teacher_debug_log_path = env_teacher_dbg
    env_teacher_dbg_limit = os.getenv('TEACHER_DEBUG_LIMIT')
    if env_teacher_dbg_limit:
        parsed_args.teacher_debug_limit = int(env_teacher_dbg_limit)
    env_teacher_dbg_thresh = os.getenv('TEACHER_DEBUG_THRESHOLD')
    if env_teacher_dbg_thresh:
        parsed_args.teacher_debug_threshold = float(env_teacher_dbg_thresh)
    env_hard_fraction = os.getenv('HARD_NEGATIVE_FRACTION')
    if env_hard_fraction:
        parsed_args.hard_negative_fraction = float(env_hard_fraction)
    env_struct_fraction = os.getenv('STRUCTURAL_NEGATIVE_FRACTION')
    if env_struct_fraction:
        parsed_args.structural_negative_fraction = float(env_struct_fraction)
    env_struct_size = os.getenv('STRUCTURAL_NEGATIVE_SIZE')
    if env_struct_size:
        parsed_args.structural_negative_size = int(env_struct_size)
    env_candidate_rules = os.getenv('TRAIN_ANYBURL_RULES')
    if env_candidate_rules:
        parsed_args.train_anyburl_rules = env_candidate_rules
    env_candidate_fraction = os.getenv('CANDIDATE_NEGATIVE_FRACTION')
    if env_candidate_fraction:
        parsed_args.candidate_negative_fraction = float(env_candidate_fraction)
    env_candidate_rule_topk = os.getenv('CANDIDATE_RULE_TOPK')
    if env_candidate_rule_topk:
        parsed_args.candidate_rule_topk = int(env_candidate_rule_topk)
    env_candidate_fallback_topk = os.getenv('CANDIDATE_FALLBACK_TOPK')
    if env_candidate_fallback_topk:
        parsed_args.candidate_fallback_topk = int(env_candidate_fallback_topk)
    env_candidate_cache_max = os.getenv('CANDIDATE_CACHE_MAX')
    if env_candidate_cache_max:
        parsed_args.candidate_cache_max = int(env_candidate_cache_max)
    env_emu_fraction = os.getenv('EMU_NEGATIVE_FRACTION')
    if env_emu_fraction:
        parsed_args.emu_negative_fraction = float(env_emu_fraction)
    env_emu_walks = os.getenv('EMU_NUM_WALKS')
    if env_emu_walks:
        parsed_args.emu_num_walks = int(env_emu_walks)
    env_emu_walk_length = os.getenv('EMU_WALK_LENGTH')
    if env_emu_walk_length:
        parsed_args.emu_walk_length = int(env_emu_walk_length)
    env_emu_cache_size = os.getenv('EMU_CACHE_SIZE')
    if env_emu_cache_size:
        parsed_args.emu_cache_size = int(env_emu_cache_size)
    env_emu_relation_quota = os.getenv('EMU_RELATION_QUOTA')
    if env_emu_relation_quota:
        parsed_args.emu_relation_quota = int(env_emu_relation_quota)
    env_ramp_start = os.getenv('RAMP_START_STEP')
    if env_ramp_start:
        parsed_args.ramp_start_step = int(env_ramp_start)
    env_ramp_end = os.getenv('RAMP_END_STEP')
    if env_ramp_end:
        parsed_args.ramp_end_step = int(env_ramp_end)
    env_phase_scale_t = os.getenv('PHASE_WEIGHT_SCALE_TARGET')
    if env_phase_scale_t:
        parsed_args.phase_weight_scale_target = float(env_phase_scale_t)
    env_phase_sharp_t = os.getenv('PHASE_SHARPNESS_TARGET')
    if env_phase_sharp_t:
        parsed_args.phase_sharpness_target = float(env_phase_sharp_t)
    env_mod_sharp_t = os.getenv('MODULUS_SHARPNESS_TARGET')
    if env_mod_sharp_t:
        parsed_args.modulus_sharpness_target = float(env_mod_sharp_t)
    env_adv_temp_t = os.getenv('ADVERSARIAL_TEMPERATURE_TARGET')
    if env_adv_temp_t:
        parsed_args.adversarial_temperature_target = float(env_adv_temp_t)
    env_neg_size_t = os.getenv('NEGATIVE_SAMPLE_SIZE_TARGET')
    if env_neg_size_t:
        parsed_args.negative_sample_size_target = int(env_neg_size_t)
    env_path_weight_t = os.getenv('PATH_LOSS_WEIGHT_TARGET')
    if env_path_weight_t:
        parsed_args.path_loss_weight_target = float(env_path_weight_t)
    env_rel_weights = os.getenv('RELATION_LOSS_WEIGHTS')
    if env_rel_weights:
        parsed_args.relation_loss_weights = env_rel_weights
    env_valid_steps = os.getenv('VALID_STEPS')
    if env_valid_steps:
        parsed_args.valid_steps = int(env_valid_steps)
    if os.getenv('SKIP_OPTIMIZER_STATE', '').lower() in {'1', 'true', 'yes'}:
        parsed_args.skip_optimizer_state = True
    env_phase_scale = os.getenv('PHASE_WEIGHT_SCALE')
    if env_phase_scale:
        parsed_args.phase_weight_scale = float(env_phase_scale)
    if os.getenv('USE_REGION_HEAD', '').lower() in {'1', 'true', 'yes'}:
        parsed_args.use_region_head = True
    env_region_dim = os.getenv('REGION_DIM')
    if env_region_dim:
        parsed_args.region_dim = int(env_region_dim)
    env_region_blend = os.getenv('REGION_BLEND_WEIGHT')
    if env_region_blend:
        parsed_args.region_blend_weight = float(env_region_blend)
    env_region_blend_final = os.getenv('REGION_BLEND_FINAL_WEIGHT')
    if env_region_blend_final:
        parsed_args.region_blend_final_weight = float(env_region_blend_final)
    env_region_blend_warmup = os.getenv('REGION_BLEND_WARMUP_STEPS')
    if env_region_blend_warmup:
        parsed_args.region_blend_warmup_steps = int(env_region_blend_warmup)
    env_region_volume_penalty = os.getenv('REGION_VOLUME_PENALTY')
    if env_region_volume_penalty:
        parsed_args.region_volume_penalty = float(env_region_volume_penalty)
    env_region_depth_weight = os.getenv('REGION_DEPTH_WEIGHT')
    if env_region_depth_weight:
        parsed_args.region_depth_weight = float(env_region_depth_weight)
    env_region_depth_margin = os.getenv('REGION_DEPTH_MARGIN')
    if env_region_depth_margin:
        parsed_args.region_depth_margin = float(env_region_depth_margin)
    if os.getenv('USE_HYPER_SUBSPACE', '').lower() in {'1', 'true', 'yes'}:
        parsed_args.use_hyper_subspace = True
    env_hyper_dim = os.getenv('HYPER_DIM')
    if env_hyper_dim:
        parsed_args.hyper_dim = int(env_hyper_dim)
    env_hyper_blend_warmup = os.getenv('HYPER_BLEND_WARMUP_STEPS')
    if env_hyper_blend_warmup:
        parsed_args.hyper_blend_warmup_steps = int(env_hyper_blend_warmup)
    env_hyper_radius_weight = os.getenv('HYPER_RADIUS_WEIGHT')
    if env_hyper_radius_weight:
        parsed_args.hyper_radius_weight = float(env_hyper_radius_weight)
    env_hyper_radius_margin = os.getenv('HYPER_RADIUS_MARGIN')
    if env_hyper_radius_margin:
        parsed_args.hyper_radius_margin = float(env_hyper_radius_margin)
    env_hyper_radius_warmup = os.getenv('HYPER_RADIUS_WARMUP_STEPS')
    if env_hyper_radius_warmup:
        parsed_args.hyper_radius_warmup_steps = int(env_hyper_radius_warmup)
    if os.getenv('EXPOSE_QUERY_EMBEDDING', '').lower() in {'1', 'true', 'yes'}:
        parsed_args.expose_query_embedding = True
    if os.getenv('USE_MOS_HEAD', '').lower() in {'1', 'true', 'yes'}:
        parsed_args.use_mos_head = True
    env_mos_components = os.getenv('MOS_COMPONENTS')
    if env_mos_components:
        parsed_args.mos_components = int(env_mos_components)
    env_mos_hidden = os.getenv('MOS_HIDDEN_DIM')
    if env_mos_hidden:
        parsed_args.mos_hidden_dim = int(env_mos_hidden)
    env_mos_entropy = os.getenv('MOS_ENTROPY_WEIGHT')
    if env_mos_entropy:
        parsed_args.mos_entropy_weight = float(env_mos_entropy)
    if os.getenv('USE_HYPERBOLIC_RELATE', '').lower() in {'1', 'true', 'yes'}:
        parsed_args.use_hyperbolic_relate = True
    env_hyper_c = os.getenv('HYPERBOLIC_CURVATURE')
    if env_hyper_c:
        parsed_args.hyperbolic_curvature = float(env_hyper_c)
    if os.getenv('USE_RELATION_GATE', '').lower() in {'1', 'true', 'yes'}:
        parsed_args.use_relation_gate = True
    if os.getenv('USE_TYPE_MOD_NORM', '').lower() in {'1', 'true', 'yes'}:
        parsed_args.use_type_mod_norm = True
    if os.getenv('USE_HIERARCHY_MOD_HEAD', '').lower() in {'1', 'true', 'yes'}:
        parsed_args.use_hierarchy_mod_head = True
    env_comp_phase_w = os.getenv('COMPOSITION_PHASE_WEIGHT')
    if env_comp_phase_w:
        parsed_args.composition_phase_weight = float(env_comp_phase_w)
    env_comp_mod_w = os.getenv('COMPOSITION_MODULUS_WEIGHT')
    if env_comp_mod_w:
        parsed_args.composition_modulus_weight = float(env_comp_mod_w)
    env_use_qa = os.getenv('USE_QUERY_ADAPTIVE')
    if env_use_qa:
        parsed_args.use_query_adaptive = env_use_qa.lower() in {'1', 'true', 'yes', 'y'}
    env_qa_hidden = os.getenv('QA_HIDDEN_DIM')
    if env_qa_hidden:
        parsed_args.qa_hidden_dim = int(env_qa_hidden)
    env_qa_experts = os.getenv('QA_NUM_EXPERTS')
    if env_qa_experts:
        parsed_args.qa_num_experts = int(env_qa_experts)
    env_qa_t_floor = os.getenv('QA_TEMPERATURE_FLOOR')
    if env_qa_t_floor:
        parsed_args.qa_temperature_floor = float(env_qa_t_floor)
    env_qa_t_ceil = os.getenv('QA_TEMPERATURE_CEILING')
    if env_qa_t_ceil:
        parsed_args.qa_temperature_ceiling = float(env_qa_t_ceil)
    env_qa_proto_w = os.getenv('QA_PROTOTYPE_WEIGHT')
    if env_qa_proto_w:
        parsed_args.qa_prototype_weight = float(env_qa_proto_w)
    env_qa_ctr_w = os.getenv('QA_CONTRASTIVE_WEIGHT')
    if env_qa_ctr_w:
        parsed_args.qa_contrastive_weight = float(env_qa_ctr_w)
    env_qa_ctr_t = os.getenv('QA_CONTRASTIVE_TEMP')
    if env_qa_ctr_t:
        parsed_args.qa_contrastive_temp = float(env_qa_ctr_t)
    env_plm_ent = os.getenv('PLM_ENTITY_EMB_PATH')
    if env_plm_ent:
        parsed_args.plm_entity_emb_path = env_plm_ent
    env_plm_rel = os.getenv('PLM_REL_EMB_PATH')
    if env_plm_rel:
        parsed_args.plm_relation_emb_path = env_plm_rel
    env_plm_ent_w = os.getenv('PLM_ENTITY_REG_WEIGHT')
    if env_plm_ent_w:
        parsed_args.plm_entity_reg_weight = float(env_plm_ent_w)
    env_plm_rel_w = os.getenv('PLM_REL_REG_WEIGHT')
    if env_plm_rel_w:
        parsed_args.plm_relation_reg_weight = float(env_plm_rel_w)
    env_plm_teacher = os.getenv('PLM_TEACHER')
    if env_plm_teacher:
        parsed_args.plm_teacher = env_plm_teacher.lower() in {'1', 'true', 'yes', 'y'}
    env_plm_teacher_temp = os.getenv('PLM_TEACHER_TEMPERATURE')
    if env_plm_teacher_temp:
        parsed_args.plm_teacher_temperature = float(env_plm_teacher_temp)
    env_plm_teacher_cos = os.getenv('PLM_TEACHER_COSINE')
    if env_plm_teacher_cos:
        parsed_args.plm_teacher_cosine = env_plm_teacher_cos.lower() in {'1', 'true', 'yes', 'y'}
    env_hier_contrast = os.getenv('HIER_CONTRASTIVE_WEIGHT')
    if env_hier_contrast:
        parsed_args.hier_contrastive_weight = float(env_hier_contrast)
    env_hier_temp = os.getenv('HIER_CONTRASTIVE_TEMP')
    if env_hier_temp:
        parsed_args.hier_contrastive_temp = float(env_hier_temp)
    env_hier_phase = os.getenv('HIER_CONTRASTIVE_PHASE_ONLY')
    if env_hier_phase:
        parsed_args.hier_contrastive_phase_only = env_hier_phase.lower() in {'1', 'true', 'yes', 'y'}
    env_hier_depth_neg = os.getenv('HIER_DEPTH_NEGATIVES')
    if env_hier_depth_neg:
        parsed_args.hier_depth_negatives = int(env_hier_depth_neg)
    env_mixkg = os.getenv('MIXKG_ENABLE')
    if env_mixkg:
        parsed_args.mixkg_enable = env_mixkg.lower() in {'1', 'true', 'yes', 'y'}
    env_mixkg_topk = os.getenv('MIXKG_TOPK')
    if env_mixkg_topk:
        parsed_args.mixkg_topk = int(env_mixkg_topk)
    env_mixkg_mix = os.getenv('MIXKG_MIX_COUNT')
    if env_mixkg_mix:
        parsed_args.mixkg_mix_count = int(env_mixkg_mix)
    env_mixkg_alpha = os.getenv('MIXKG_ALPHA')
    if env_mixkg_alpha:
        parsed_args.mixkg_alpha = float(env_mixkg_alpha)
    env_mixkg_sim = os.getenv('MIXKG_USE_SIMILARITY')
    if env_mixkg_sim:
        parsed_args.mixkg_use_similarity = env_mixkg_sim.lower() in {'1', 'true', 'yes', 'y'}
    env_mixkg_w = os.getenv('MIXKG_SCORE_WEIGHT')
    if env_mixkg_w:
        parsed_args.mixkg_score_weight = float(env_mixkg_w)

    parsed_args.hard_negative_fraction = min(1.0, max(0.0, parsed_args.hard_negative_fraction))
    parsed_args.structural_negative_fraction = min(1.0, max(0.0, parsed_args.structural_negative_fraction))
    parsed_args.candidate_negative_fraction = min(1.0, max(0.0, parsed_args.candidate_negative_fraction))
    parsed_args.emu_negative_fraction = min(1.0, max(0.0, parsed_args.emu_negative_fraction))
    if parsed_args.region_blend_final_weight is None:
        parsed_args.region_blend_final_weight = parsed_args.region_blend_weight

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

def load_entity_depths(depth_path, entity2id):
    if not depth_path or not os.path.exists(depth_path):
        return None
    with open(depth_path, 'r') as fin:
        depth_map_raw = json.load(fin)
    idx_depth = {}
    missing = 0
    for entity, depth in depth_map_raw.items():
        if entity in entity2id:
            idx_depth[entity2id[entity]] = float(depth)
        else:
            missing += 1
    logging.info('Loaded hierarchy depths for %d entities (%d missing).', len(idx_depth), missing)
    return idx_depth

def load_relation_behaviors(path, relation2id):
    if not path or not os.path.exists(path):
        return {}
    with open(path, 'r') as fin:
        data = json.load(fin)
    symmetric = set()
    antisymmetric = set()
    hierarchical = set()
    meronymic = set()
    inverse_pairs = []
    for rel in data.get('symmetric', []):
        if rel in relation2id:
            symmetric.add(relation2id[rel])
    for rel in data.get('antisymmetric', []):
        if rel in relation2id:
            antisymmetric.add(relation2id[rel])
    for rel in data.get('hierarchical', []):
        if rel in relation2id:
            hierarchical.add(relation2id[rel])
    for rel in data.get('meronymic', []):
        if rel in relation2id:
            meronymic.add(relation2id[rel])
    for pair in data.get('inverse_pairs', []):
        if len(pair) != 2:
            continue
        lhs, rhs = pair
        if lhs in relation2id and rhs in relation2id:
            inverse_pairs.append((relation2id[lhs], relation2id[rhs]))
    compositions = []
    for triple in data.get('compositions', []):
        if len(triple) != 3:
            continue
        r1, r2, r3 = triple
        if r1 in relation2id and r2 in relation2id and r3 in relation2id:
            compositions.append((relation2id[r1], relation2id[r2], relation2id[r3]))
    logging.info('Loaded relation behaviors: %d symmetric, %d antisymmetric, %d inverse pairs, %d hierarchical, %d meronymic.',
                 len(symmetric), len(antisymmetric), len(inverse_pairs), len(hierarchical), len(meronymic))
    return {
        'symmetric': symmetric,
        'antisymmetric': antisymmetric,
        'inverse_pairs': inverse_pairs,
        'hierarchical': hierarchical,
        'meronymic': meronymic,
        'compositions': compositions
    }


def load_entity_concepts(concept_map_path, entity2id):
    if not concept_map_path or not os.path.exists(concept_map_path):
        return None, None
    with open(concept_map_path, 'r') as fin:
        raw_map = json.load(fin)
    concept_to_idx = {}
    entity_concepts = {}
    missing = 0
    for entity, concept in raw_map.items():
        if entity not in entity2id:
            missing += 1
            continue
        concept_key = str(concept)
        if concept_key not in concept_to_idx:
            concept_to_idx[concept_key] = len(concept_to_idx)
        entity_concepts[entity2id[entity]] = concept_to_idx[concept_key]
    logging.info('Loaded concepts for %d entities across %d concepts (%d missing entities).',
                 len(entity_concepts), len(concept_to_idx), missing)
    return entity_concepts, concept_to_idx


def load_concept_depths(concept_depth_path, concept_to_idx, entity_concepts=None, entity_depths=None):
    if not concept_to_idx:
        return None
    concept_depths = {}
    if concept_depth_path and os.path.exists(concept_depth_path):
        with open(concept_depth_path, 'r') as fin:
            raw = json.load(fin)
        for concept, depth in raw.items():
            concept_key = str(concept)
            if concept_key in concept_to_idx:
                concept_depths[concept_to_idx[concept_key]] = float(depth)
    elif entity_concepts and entity_depths:
        depth_buckets = defaultdict(list)
        for ent_idx, concept_idx in entity_concepts.items():
            if ent_idx in entity_depths:
                depth_buckets[concept_idx].append(float(entity_depths[ent_idx]))
        for concept_idx, vals in depth_buckets.items():
            if vals:
                concept_depths[concept_idx] = float(sum(vals) / len(vals))
    if concept_depths:
        logging.info('Loaded concept depths for %d concepts.', len(concept_depths))
        return concept_depths
    return None


def _apply_compiled_init(model, compiled_dir, entity_dim, nentity, nrelation):
    if not compiled_dir:
        return
    ent_phase_path = os.path.join(compiled_dir, 'entity_phase.npy')
    rel_phase_path = os.path.join(compiled_dir, 'relation_phase.npy')
    ent_mod_path = os.path.join(compiled_dir, 'entity_modulus.npy')
    rel_mod_path = os.path.join(compiled_dir, 'relation_modulus.npy')
    for path in (ent_phase_path, rel_phase_path, ent_mod_path, rel_mod_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f'Compiled init missing: {path}')

    ent_phase = np.load(ent_phase_path)
    rel_phase = np.load(rel_phase_path)
    ent_mod = np.load(ent_mod_path)
    rel_mod = np.load(rel_mod_path)

    if ent_phase.shape[0] != nentity or ent_mod.shape[0] != nentity:
        raise ValueError('Compiled entity size does not match dataset entities.')
    if rel_phase.shape[0] != nrelation or rel_mod.shape[0] != nrelation:
        if nrelation % 2 == 0 and rel_phase.shape[0] * 2 == nrelation:
            logging.info('Expanding compiled relations to include reciprocals.')
            rel_phase = np.concatenate([rel_phase, rel_phase.copy()], axis=0)
            rel_mod = np.concatenate([rel_mod, rel_mod.copy()], axis=0)
        else:
            raise ValueError('Compiled relation size does not match dataset relations.')

    if entity_dim % 2 != 0:
        raise ValueError('hidden_dim must be even for RelatE compiled init.')
    phase_dim = ent_phase.shape[1]
    expected_half = entity_dim // 2
    if phase_dim != expected_half:
        raise ValueError(
            f'Compiled phase dim {phase_dim} does not match model half-dim {expected_half}.'
        )

    with torch.no_grad():
        model.entity_embedding[:, :phase_dim].copy_(torch.from_numpy(ent_mod))
        model.entity_embedding[:, phase_dim:phase_dim * 2].copy_(torch.from_numpy(ent_phase))
        model.relation_embedding[:, :phase_dim].copy_(torch.from_numpy(rel_mod))
        model.relation_embedding[:, phase_dim:phase_dim * 2].copy_(torch.from_numpy(rel_phase))

def resolve_relation_weight_spec(spec, relation2id):
    if not spec:
        return {}
    if os.path.isfile(spec):
        try:
            with open(spec, 'r') as fin:
                data = json.load(fin)
        except Exception as exc:  # pylint: disable=broad-except
            logging.warning('Failed to read relation weight file %s: %s', spec, exc)
            return {}
        if isinstance(data, dict):
            spec = ' '.join(f'{k}:{v}' for k, v in data.items())
        else:
            logging.warning('Relation weight file %s must contain a JSON object; ignoring.', spec)
            return {}
    weights = {}
    tokens = spec.replace(',', ' ').split()
    for token in tokens:
        if ':' not in token:
            continue
        rel_key, weight_str = token.split(':', 1)
        rel_key = rel_key.strip()
        try:
            weight_val = float(weight_str.strip())
        except ValueError:
            continue
        if rel_key in relation2id:
            rel_idx = relation2id[rel_key]
        else:
            try:
                rel_idx = int(rel_key)
            except ValueError:
                continue
            if rel_idx < 0 or rel_idx >= len(relation2id):
                continue
        weights[rel_idx] = weight_val
    return weights

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

def build_reverse_adjacency(triples):
    reverse_adj = defaultdict(list)
    for h, r, t in triples:
        reverse_adj[t].append((r, h))
    return reverse_adj

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

def build_structural_negative_cache(adjacency, reverse_adjacency, relation_behaviors,
                                    max_candidates=256, seed=0):
    """
    Extends the generic two-hop cache with relation-aware buckets for hierarchical
    and meronymic relations so we can sample siblings/ancestors as negatives.
    """
    cache = build_two_hop_cache(adjacency, max_candidates=max_candidates, seed=seed)
    target_relations = set()
    if relation_behaviors:
        target_relations.update(relation_behaviors.get('hierarchical', []))
        target_relations.update(relation_behaviors.get('meronymic', []))
    if not target_relations:
        return cache
    rng = random.Random(seed + 1)
    for head, neighbors in adjacency.items():
        rel_groups = defaultdict(set)
        for rel, tail in neighbors:
            if rel in target_relations:
                rel_groups[rel].add(tail)
        if not rel_groups:
            continue
        for rel, tails in rel_groups.items():
            candidates = set(tails)
            for tail in list(tails):
                for rel_next, tail_next in adjacency.get(tail, []):
                    if rel_next == rel or rel_next in target_relations:
                        candidates.add(tail_next)
                for rel_prev, parent in reverse_adjacency.get(tail, []):
                    if rel_prev == rel or rel_prev in target_relations:
                        candidates.add(parent)
                        for _, cousin in adjacency.get(parent, []):
                            candidates.add(cousin)
            for rel_prev, parent in reverse_adjacency.get(head, []):
                if rel_prev == rel or rel_prev in target_relations:
                    candidates.add(parent)
                    for _, sibling_tail in adjacency.get(parent, []):
                        candidates.add(sibling_tail)
            if not candidates:
                continue
            cand_list = list(candidates)
            rng.shuffle(cand_list)
            cache[(head, rel)] = cand_list[:max_candidates]
    return cache

def build_emu_negative_cache(train_triples, adjacency, reverse_adjacency, nrelation,
                             max_candidates=512, num_walks=4, walk_length=3,
                             relation_quota=128, seed=0):
    """
    Build EMU-style graph-walk negative caches.
    Returns two caches:
      - tail_cache keyed by (head, relation) for tail-batch negatives
      - head_cache keyed by (relation, tail) for head-batch negatives
    """
    rng = random.Random(seed)
    rel_to_tails = defaultdict(set)
    rel_to_heads = defaultdict(set)
    tails_by_query = defaultdict(set)
    heads_by_query = defaultdict(set)

    for h, r, t in train_triples:
        base_r = r % nrelation if nrelation > 0 else r
        rel_to_tails[base_r].add(t)
        rel_to_heads[base_r].add(h)
        tails_by_query[(h, r)].add(t)
        heads_by_query[(r, t)].add(h)

    def _walk_collect(start_node, graph, reverse_graph):
        candidates = set()
        if walk_length <= 0 or num_walks <= 0:
            return candidates
        for _ in range(num_walks):
            cur = start_node
            for _ in range(walk_length):
                neighbors = graph.get(cur, [])
                if not neighbors:
                    break
                _, nxt = rng.choice(neighbors)
                candidates.add(nxt)
                rev_neighbors = reverse_graph.get(nxt, [])
                if rev_neighbors:
                    _, parent = rng.choice(rev_neighbors)
                    candidates.add(parent)
                cur = nxt
        return candidates

    total_tail_queries = len(tails_by_query)
    total_head_queries = len(heads_by_query)
    tail_log_stride = max(1, total_tail_queries // 10) if total_tail_queries > 0 else 1
    head_log_stride = max(1, total_head_queries // 10) if total_head_queries > 0 else 1

    logging.info(
        'EMU cache build start: triples=%d, tail_queries=%d, head_queries=%d, walks=%d, walk_len=%d, max_candidates=%d',
        len(train_triples),
        total_tail_queries,
        total_head_queries,
        num_walks,
        walk_length,
        max_candidates
    )

    tail_cache = {}
    for idx, ((h, r), true_tails) in enumerate(tails_by_query.items(), start=1):
        base_r = r % nrelation if nrelation > 0 else r
        candidates = set()
        candidates.update(t for _, t in adjacency.get(h, []))
        candidates.update(_walk_collect(h, adjacency, reverse_adjacency))
        rel_tail_pool = list(rel_to_tails.get(base_r, []))
        if rel_tail_pool:
            rng.shuffle(rel_tail_pool)
            candidates.update(rel_tail_pool[:max(0, relation_quota)])
        for t in true_tails:
            for _, parent in reverse_adjacency.get(t, []):
                candidates.add(parent)
                for _, sib in adjacency.get(parent, []):
                    candidates.add(sib)
        candidates.difference_update(true_tails)
        if candidates:
            cand_list = list(candidates)
            rng.shuffle(cand_list)
            tail_cache[(h, r)] = cand_list[:max(1, max_candidates)]
        if idx == 1 or idx % tail_log_stride == 0 or idx == total_tail_queries:
            logging.info('EMU tail cache progress: %d/%d keys processed.', idx, total_tail_queries)

    head_cache = {}
    for idx, ((r, t), true_heads) in enumerate(heads_by_query.items(), start=1):
        base_r = r % nrelation if nrelation > 0 else r
        candidates = set()
        candidates.update(h for _, h in reverse_adjacency.get(t, []))
        candidates.update(_walk_collect(t, reverse_adjacency, adjacency))
        rel_head_pool = list(rel_to_heads.get(base_r, []))
        if rel_head_pool:
            rng.shuffle(rel_head_pool)
            candidates.update(rel_head_pool[:max(0, relation_quota)])
        for h in true_heads:
            for _, parent in reverse_adjacency.get(h, []):
                candidates.add(parent)
                for _, sib in adjacency.get(parent, []):
                    candidates.add(sib)
        candidates.difference_update(true_heads)
        if candidates:
            cand_list = list(candidates)
            rng.shuffle(cand_list)
            head_cache[(r, t)] = cand_list[:max(1, max_candidates)]
        if idx == 1 or idx % head_log_stride == 0 or idx == total_head_queries:
            logging.info('EMU head cache progress: %d/%d keys processed.', idx, total_head_queries)

    logging.info(
        'EMU cache build done: tail_keys=%d, head_keys=%d',
        len(tail_cache),
        len(head_cache)
    )

    return tail_cache, head_cache

def build_anyburl_candidate_cache(train_triples, rules, nrelation,
                                  rule_topk=3000, fallback_topk=200, candidate_max=9000):
    """
    Precompute AnyBURL-driven candidate lists for (head, relation) pairs
    so training can sample negatives from rule-derived candidates.
    """
    if not rules:
        return {}
    tails_by_head, heads_by_tail = build_relation_maps(train_triples, nrelation)
    adjacency = build_adjacency(train_triples)
    rules_by_rel = defaultdict(list)
    for rule in rules:
        rules_by_rel[rule['head_rel']].append(rule)

    cache = {}
    seen = set()
    for h, r, _ in train_triples:
        key = (h, r)
        if key in seen:
            continue
        seen.add(key)
        candidates = generate_candidates_with_rules(
            h,
            r,
            tails_by_head,
            heads_by_tail,
            adjacency,
            train_triples,
            rules_by_rel,
            candidate_max=candidate_max,
            topk_fallback=fallback_topk,
            rule_topk=rule_topk
        )
        if candidates:
            cache[key] = list(candidates.keys())
    return cache

def should_enable_path_loss(step, args):
    if args.path_loss_weight <= 0:
        return False
    if args.path_curriculum_steps is None:
        return True
    start, end = args.path_curriculum_steps
    return (step >= start) and (end <= 0 or step <= end)

def _ramp_value(step, start, end, v0, v1):
    if end <= start:
        return v1
    if step <= start:
        return v0
    if step >= end:
        return v1
    alpha = (step - start) / float(end - start)
    return v0 + (v1 - v0) * alpha

def override_config(args):
    '''
    Override model and data configuration
    '''
    config_root = args.init_checkpoint
    if config_root and os.path.isfile(config_root):
        config_root = os.path.dirname(config_root)
    with open(os.path.join(config_root, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    checkpoint_model = argparse_dict['model']
    requested_model = getattr(args, 'model', checkpoint_model)
    if requested_model != checkpoint_model:
        logging.info('Overriding checkpoint model %s with requested model %s.',
                     checkpoint_model, requested_model)
        args.model = requested_model
    else:
        args.model = checkpoint_model
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']
    args.init_modulus_weight = argparse_dict.get('init_modulus_weight', 3.0) # adding the new modulus weight parameter
    args.phase_weight_scale = argparse_dict.get('phase_weight_scale', 0.65)
    args.use_region_head = argparse_dict.get('use_region_head', False)
    args.region_dim = argparse_dict.get('region_dim', 64)
    args.region_blend_weight = argparse_dict.get('region_blend_weight', 0.5)
    args.use_hyperbolic_relate = argparse_dict.get('use_hyperbolic_relate', False)
    args.hyperbolic_curvature = argparse_dict.get('hyperbolic_curvature', 1.0)


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
    if multiprocessing.current_process().name != 'MainProcess':
        return

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='a'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def _worker_init_logging(_):
    logging.disable(logging.CRITICAL)

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

    args.region_blend_weight_start = args.region_blend_weight
    if args.region_blend_final_weight is None:
        args.region_blend_final_weight = args.region_blend_weight
    
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
    entity_depths = load_entity_depths(args.hierarchy_depth_path, entity2id)
    relation_behaviors = load_relation_behaviors(args.relation_behavior_path, relation2id)
    entity_concepts, concept_to_idx = load_entity_concepts(args.entity_concept_map_path, entity2id)
    concept_depths = load_concept_depths(
        args.concept_depth_map_path,
        concept_to_idx,
        entity_concepts=entity_concepts,
        entity_depths=entity_depths
    )
    hard_relation_ids = set()
    hard_relation_ids.update(relation_behaviors.get('hierarchical', []))
    hard_relation_ids.update(relation_behaviors.get('meronymic', []))
    if args.extra_hard_relations:
        hard_relation_ids.update(args.extra_hard_relations)
        logging.info('Added %d extra hard relations to hard-negative pool.', len(args.extra_hard_relations))
    relation_prompt_tensor = None
    
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
    if args.use_rel_prompt_emb and args.rel_prompt_path:
        prompt_path = os.path.abspath(args.rel_prompt_path)
        if not os.path.exists(prompt_path):
            logging.warning('Relation prompt path %s not found; disabling prompt augmentation.', prompt_path)
            args.use_rel_prompt_emb = False
        else:
            prompt_array = np.load(prompt_path)
            if prompt_array.ndim != 2:
                logging.warning('Relation prompt file %s must be 2-D (got shape %s); disabling.', prompt_path, prompt_array.shape)
                args.use_rel_prompt_emb = False
            elif prompt_array.shape[0] != nrelation:
                logging.warning('Relation prompt count (%d) does not match relations (%d); disabling.', prompt_array.shape[0], nrelation)
                args.use_rel_prompt_emb = False
            else:
                relation_prompt_tensor = torch.from_numpy(prompt_array).float()
                logging.info('Loaded %d relation prompt vectors from %s', prompt_array.shape[0], prompt_path)
    else:
        args.use_rel_prompt_emb = False
    id2entity_map = {idx: ent for ent, idx in entity2id.items()}
    id2relation_map = {idx: rel for rel, idx in relation2id.items()}
    relation_loss_weights_tensor = None
    kd_relation_weights_tensor = None
    args.id2entity = id2entity_map
    args.id2relation = id2relation_map
    base_nrelation = nrelation
    
    args.nentity = nentity
    args.nrelation = nrelation

    plm_entity_vectors = None
    plm_relation_vectors = None
    if args.plm_entity_emb_path:
        plm_entity_vectors = np.load(os.path.abspath(args.plm_entity_emb_path))
        if plm_entity_vectors.shape[0] != nentity:
            logging.warning('PLM entity embeddings size mismatch: %s vs nentity=%d',
                            plm_entity_vectors.shape, nentity)
    if args.plm_relation_emb_path:
        plm_relation_vectors = np.load(os.path.abspath(args.plm_relation_emb_path))
        if plm_relation_vectors.shape[0] != nrelation:
            logging.warning('PLM relation embeddings size mismatch: %s vs nrelation=%d',
                            plm_relation_vectors.shape, nrelation)
    if args.plm_teacher and plm_entity_vectors is None:
        logging.warning('PLM teacher enabled but PLM entity embeddings not provided; disabling PLM teacher.')
        args.plm_teacher = False
    
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
    rel_loss_map = resolve_relation_weight_spec(getattr(args, 'relation_loss_weights', None), relation2id)
    if rel_loss_map:
        relation_loss_weights_tensor = torch.ones(nrelation, dtype=torch.float32)
        for idx, weight in rel_loss_map.items():
            relation_loss_weights_tensor[idx] = float(weight)

    kd_rel_map = resolve_relation_weight_spec(getattr(args, 'kd_relation_weights', None), relation2id)
    if kd_rel_map:
        kd_relation_weights_tensor = torch.ones(nrelation, dtype=torch.float32)
        for idx, weight in kd_rel_map.items():
            kd_relation_weights_tensor[idx] = float(weight)
    sampling_weight_map = resolve_relation_weight_spec(getattr(args, 'relation_sampling_weights', None), relation2id)
    if sampling_weight_map:
        logging.info('Applying relation sampling weights for %d relations.', len(sampling_weight_map))

    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    test_triples  = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)

    train_triples_raw = list(train_triples)
    adjacency = build_adjacency(train_triples_raw)
    reverse_adjacency = build_reverse_adjacency(train_triples_raw)
    structural_cache = build_structural_negative_cache(
        adjacency,
        reverse_adjacency,
        relation_behaviors,
        max_candidates=max(1, args.structural_negative_size)
    )

    candidate_cache = {}
    if args.train_anyburl_rules and args.candidate_negative_fraction > 0:
        logging.info('Loading AnyBURL rules for candidate negatives from %s', args.train_anyburl_rules)
        anyburl_rules = parse_anyburl_rules(args.train_anyburl_rules, entity2id, relation2id)
        candidate_cache = build_anyburl_candidate_cache(
            train_triples_raw,
            anyburl_rules,
            nrelation,
            rule_topk=args.candidate_rule_topk,
            fallback_topk=args.candidate_fallback_topk,
            candidate_max=args.candidate_cache_max
        )
        if candidate_cache:
            avg_size = sum(len(v) for v in candidate_cache.values()) / len(candidate_cache)
            logging.info('Built AnyBURL candidate cache: %d (head,rel) keys, avg size %.1f',
                         len(candidate_cache), avg_size)
        else:
            logging.warning('AnyBURL candidate cache is empty; falling back to random negatives.')

    emu_tail_cache = {}
    emu_head_cache = {}
    if args.emu_negative_fraction > 0:
        emu_tail_cache, emu_head_cache = build_emu_negative_cache(
            train_triples_raw,
            adjacency,
            reverse_adjacency,
            base_nrelation,
            max_candidates=max(1, args.emu_cache_size),
            num_walks=max(1, args.emu_num_walks),
            walk_length=max(1, args.emu_walk_length),
            relation_quota=max(0, args.emu_relation_quota)
        )
        if emu_tail_cache or emu_head_cache:
            tail_avg = (sum(len(v) for v in emu_tail_cache.values()) / len(emu_tail_cache)) if emu_tail_cache else 0.0
            head_avg = (sum(len(v) for v in emu_head_cache.values()) / len(emu_head_cache)) if emu_head_cache else 0.0
            logging.info('Built EMU caches: tail=%d (avg %.1f), head=%d (avg %.1f).',
                         len(emu_tail_cache), tail_avg, len(emu_head_cache), head_avg)
        else:
            logging.warning('EMU cache is empty; falling back to existing negative samplers.')

    teacher_model = None
    teacher_alignment_vectors = None
    teacher_alignment_mask = None
    teacher_entity_dim = None
    if args.teacher_checkpoint and (args.kd_lambda > 0.0 or args.teacher_align_weight > 0.0):
        teacher_device = args.teacher_device
        if teacher_device == 'cuda' and not torch.cuda.is_available():
            teacher_device = 'cpu'
            logging.warning('CUDA not available for teacher, falling back to CPU.')
        teacher_device = torch.device(teacher_device)
        if args.teacher_type == 'simkgc':
            teacher_model = SimKGCTeacher(
                checkpoint_path=args.teacher_checkpoint,
                repo_root=os.path.abspath(args.teacher_repo),
                device=teacher_device,
                max_negatives=args.teacher_negatives,
                id2entity=id2entity_map,
                id2relation=id2relation_map
            )
            logging.info('Loaded SimKGC teacher from %s', args.teacher_checkpoint)
            teacher_entity_dim = teacher_model.entity_vectors.size(1)
            aligned = torch.zeros((nentity, teacher_entity_dim), dtype=torch.float32)
            mask = torch.zeros(nentity, dtype=torch.bool)
            for idx, entity_str in id2entity_map.items():
                teacher_idx = teacher_model.entity_to_idx.get(entity_str)
                if teacher_idx is None:
                    continue
                aligned[idx] = teacher_model.entity_vectors[teacher_idx].float().cpu()
                mask[idx] = True
            teacher_alignment_vectors = aligned
            teacher_alignment_mask = mask
        elif args.teacher_type == 'murp':
            murp_repo = os.path.abspath(args.murp_repo)
            data_dir = args.murp_data_dir
            if not data_dir:
                dataset_name = os.path.basename(os.path.normpath(args.data_path or '')).upper() or 'WN18RR'
                data_dir = os.path.join(murp_repo, 'data', dataset_name)
            teacher_model = MuRPTeacher(
                checkpoint_path=args.teacher_checkpoint,
                repo_root=murp_repo,
                data_dir=data_dir,
                device=teacher_device,
                max_negatives=args.teacher_negatives,
                id2entity=id2entity_map,
                id2relation=id2relation_map
            )
            logging.info('Loaded MuRP teacher from %s', args.teacher_checkpoint)
            teacher_entity_dim = teacher_model.entity_dim
            teacher_alignment_vectors = teacher_model.entity_vectors.cpu()
            teacher_alignment_mask = torch.ones(teacher_alignment_vectors.size(0), dtype=torch.bool)
        elif args.teacher_type == 'csprom':
            csprom_repo = os.path.abspath(args.csprom_repo)
            dataset_name = args.csprom_dataset or os.path.basename(os.path.normpath(args.data_path or ''))
            dataset_key = (dataset_name or '').lower()
            if dataset_key in {'wn18rr', 'wn18rr/'}:
                dataset_name = 'WN18RR'
            elif dataset_key in {'fb15k-237', 'fb15k237'}:
                dataset_name = 'FB15k-237'
            data_dir = args.csprom_data_dir
            if not data_dir:
                data_dir = os.path.join(csprom_repo, 'data', 'processed', dataset_name)
            config_overrides = {}
            if args.csprom_config:
                config_overrides['config_path'] = os.path.abspath(args.csprom_config)
            teacher_model = CSPromTeacher(
                checkpoint_path=args.teacher_checkpoint,
                repo_root=csprom_repo,
                dataset=dataset_name,
                data_dir=data_dir,
                device=teacher_device,
                max_negatives=args.teacher_negatives,
                id2entity=id2entity_map,
                id2relation=id2relation_map,
                config_overrides=config_overrides
            )
            logging.info('Loaded CSProm-KG teacher from %s', args.teacher_checkpoint)
            teacher_entity_dim = teacher_model.entity_vectors.size(1)
            aligned = torch.zeros((nentity, teacher_entity_dim), dtype=torch.float32)
            mask = torch.zeros(nentity, dtype=torch.bool)
            for idx, entity_str in id2entity_map.items():
                teacher_idx = teacher_model.entity_to_idx.get(entity_str)
                if teacher_idx is None:
                    continue
                aligned[idx] = teacher_model.entity_vectors[teacher_idx].float().cpu()
                mask[idx] = True
            teacher_alignment_vectors = aligned
            teacher_alignment_mask = mask
    args.teacher_entity_dim = teacher_entity_dim
    args.teacher_query_dim = getattr(teacher_model, 'query_dim', None) if teacher_model is not None else None
    if getattr(args, 'teacher_query_align_weight', 0.0) > 0 and args.teacher_query_dim is None:
        logging.warning('Teacher query alignment requested but teacher query vectors unavailable; disabling.')
        args.teacher_query_align_weight = 0.0
    if args.use_entity_prompt_emb and teacher_alignment_vectors is None:
        logging.warning('Entity prompt blend requested but teacher entity vectors unavailable; disabling.')
        args.use_entity_prompt_emb = False

    if args.add_reciprocals:
        logging.info('Adding reciprocal triples (RotatE trick).')
        train_triples = add_reciprocal_triples(train_triples, base_nrelation)
        valid_triples = add_reciprocal_triples(valid_triples, base_nrelation)
        test_triples = add_reciprocal_triples(test_triples, base_nrelation)
        nrelation *= 2
        args.nrelation = nrelation
        if relation_prompt_tensor is not None:
            relation_prompt_tensor = torch.cat([relation_prompt_tensor, relation_prompt_tensor.clone()], dim=0)
        if plm_relation_vectors is not None and plm_relation_vectors.shape[0] * 2 == nrelation:
            plm_relation_vectors = np.concatenate([plm_relation_vectors, plm_relation_vectors.copy()], axis=0)
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
        phase_weight_scale=args.phase_weight_scale,
        init_rel_width=args.init_rel_width,
        phase_harmonics=args.phase_harmonics,
        modulus_sharpness=args.modulus_sharpness,
        phase_sharpness=args.phase_sharpness,
        entity_depths=entity_depths,
        relation_behaviors=relation_behaviors,
        relation_prompt_embeddings=relation_prompt_tensor if args.use_rel_prompt_emb else None,
        relation_prompt_weight=args.rel_prompt_weight,
        relation_prompt_warmup_steps=args.rel_prompt_warmup_steps,
        use_entity_prompt=args.use_entity_prompt_emb,
        entity_prompt_weight=args.entity_prompt_weight,
        entity_prompt_warmup_steps=args.entity_prompt_warmup_steps,
        teacher_entity_vectors=teacher_alignment_vectors,
        teacher_alignment_mask=teacher_alignment_mask,
        teacher_entity_dim=teacher_entity_dim,
        teacher_query_dim=args.teacher_query_dim,
        relation_loss_weights=relation_loss_weights_tensor,
        kd_relation_weights=kd_relation_weights_tensor,
        use_mos_head=args.use_mos_head,
        mos_components=args.mos_components,
        mos_hidden_dim=args.mos_hidden_dim,
        mos_entropy_weight=args.mos_entropy_weight,
        expose_query_embedding=args.expose_query_embedding,
        use_region_head=args.use_region_head,
        region_dim=args.region_dim,
        region_blend_weight=args.region_blend_weight,
        use_hyper_subspace=args.use_hyper_subspace,
        hyper_dim=args.hyper_dim,
        hyper_blend_warmup_steps=args.hyper_blend_warmup_steps,
        use_hyperbolic=args.use_hyperbolic_relate,
        hyperbolic_c=args.hyperbolic_curvature,
        use_relation_gate=args.use_relation_gate,
        use_type_mod_norm=args.use_type_mod_norm,
        use_hierarchy_mod_head=args.use_hierarchy_mod_head,
        plm_entity_vectors=plm_entity_vectors,
        plm_relation_vectors=plm_relation_vectors,
        entity_concepts=entity_concepts,
        concept_depths=concept_depths,
        use_query_adaptive=args.use_query_adaptive,
        qa_hidden_dim=args.qa_hidden_dim,
        qa_num_experts=args.qa_num_experts,
        qa_temperature_floor=args.qa_temperature_floor,
        qa_temperature_ceiling=args.qa_temperature_ceiling,
        qa_prototype_weight=args.qa_prototype_weight,
        cc_concept_weight=args.cc_concept_weight,
        cc_depth_weight=args.cc_depth_weight,
        cc_relation_weight=args.cc_relation_weight

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

    base_model_driver = kge_model.module if hasattr(kge_model, 'module') else kge_model

    if args.compiled_init_dir:
        logging.info('Applying RELATE-Compile initialization from %s', args.compiled_init_dir)
        _apply_compiled_init(
            base_model_driver,
            args.compiled_init_dir,
            base_model_driver.entity_embedding.size(1),
            args.nentity,
            args.nrelation
        )
    
    path_iterator = None
    if args.do_train and args.path_loss_weight > 0:
        path_two_hop_cache = build_two_hop_cache(adjacency, max_candidates=max(1, args.path_negative_size * 16))
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
                two_hop_cache=path_two_hop_cache
            )
            path_dataloader = DataLoader(
                path_dataset,
                batch_size=args.path_batch_size,
                shuffle=True,
                num_workers=max(0, args.cpu_num//2),
                worker_init_fn=_worker_init_logging,
                collate_fn=PathDataset.collate_fn
            )
            path_iterator = BidirectionalOneShotIterator.one_shot_iterator(path_dataloader)

    if args.do_train:
        train_samplers = []
        sampling_weights_tensor = None
        if sampling_weight_map:
            rel_weights = np.ones(len(train_triples), dtype=np.float32)
            for idx, (_, rel, _) in enumerate(train_triples):
                rel_weights[idx] = sampling_weight_map.get(rel, 1.0)
            sampling_weights_tensor = torch.as_tensor(rel_weights, dtype=torch.double)
        train_dataset_head = TrainDataset(
            train_triples,
            nentity,
            nrelation,
            args.negative_sample_size,
            'head-batch',
            hard_relation_ids=hard_relation_ids,
            hard_negative_fraction=args.hard_negative_fraction,
            structural_cache=structural_cache,
            structural_negative_fraction=args.structural_negative_fraction,
            candidate_cache=candidate_cache,
            candidate_negative_fraction=args.candidate_negative_fraction,
            emu_tail_cache=emu_tail_cache,
            emu_head_cache=emu_head_cache,
            emu_negative_fraction=args.emu_negative_fraction
        )
        train_dataset_tail = TrainDataset(
            train_triples,
            nentity,
            nrelation,
            args.negative_sample_size,
            'tail-batch',
            hard_relation_ids=hard_relation_ids,
            hard_negative_fraction=args.hard_negative_fraction,
            structural_cache=structural_cache,
            structural_negative_fraction=args.structural_negative_fraction,
            candidate_cache=candidate_cache,
            candidate_negative_fraction=args.candidate_negative_fraction,
            emu_tail_cache=emu_tail_cache,
            emu_head_cache=emu_head_cache,
            emu_negative_fraction=args.emu_negative_fraction
        )
        train_sampler_head = None
        train_sampler_tail = None
        if sampling_weights_tensor is not None:
            train_sampler_head = WeightedRandomSampler(
                sampling_weights_tensor,
                num_samples=len(sampling_weights_tensor),
                replacement=True
            )
            train_sampler_tail = WeightedRandomSampler(
                sampling_weights_tensor,
                num_samples=len(sampling_weights_tensor),
                replacement=True
            )

        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            train_dataset_head,
            batch_size=args.batch_size,
            shuffle=(train_sampler_head is None),
            sampler=train_sampler_head,
            drop_last=True,
            num_workers=max(0, args.cpu_num//2),
            worker_init_fn=_worker_init_logging,
            collate_fn=TrainDataset.collate_fn
        )
        
        train_dataloader_tail = DataLoader(
            train_dataset_tail,
            batch_size=args.batch_size,
            shuffle=(train_sampler_tail is None),
            sampler=train_sampler_tail,
            drop_last=True,
            num_workers=max(0, args.cpu_num//2),
            worker_init_fn=_worker_init_logging,
            collate_fn=TrainDataset.collate_fn
        )
        
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        hard_negative_steps = sorted(args.hard_negative_steps or [])
        hard_negative_idx = 0
        next_hard_negative_step = (
            hard_negative_steps[hard_negative_idx]
            if hard_negative_steps else None
        )
        current_negative_sample_size = args.negative_sample_size
        
        # Set training configuration
        current_learning_rate = args.learning_rate
        base_optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, kge_model.parameters()), 
            lr=current_learning_rate,
            weight_decay=args.weight_decay
        )
        optimizer = base_optimizer
        if args.use_lookahead:
            optimizer = LookaheadOptimizer(
                base_optimizer,
                la_steps=args.lookahead_steps,
                la_alpha=args.lookahead_alpha
            )
            logging.info(
                'Lookahead enabled (steps=%d, alpha=%.3f).',
                args.lookahead_steps,
                args.lookahead_alpha
            )

        scheduler = CosineAnnealingLR(
            base_optimizer,
            T_max=args.lr_t_max or args.max_steps,
            eta_min=args.lr_eta_min
        )
        ema_helper = None
        if args.use_ema:
            ema_helper = EMAHelper(base_model_driver, decay=args.ema_decay)
            logging.info('EMA enabled (decay=%.6f).', args.ema_decay)

        manual_lr_drop_steps = set(args.lr_drop_steps or [])
        applied_lr_drop_steps = set()
        secondary_warmup_applied = False

        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2
        next_warmup_step = args.warm_up_steps

    init_step = 0
    best_val_mrr = 0.0
    best_step = 0
    patience_counter = 0
    plateau_lr_counter = 0

    if args.init_checkpoint:
        # Restore model from checkpoint (directory or explicit file path)
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        ckpt_path = args.init_checkpoint
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, 'checkpoint')
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        init_step = checkpoint['step']
        model_to_load = kge_model.module if hasattr(kge_model, 'module') else kge_model
        model_to_load.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if args.do_train and not args.skip_optimizer_state and 'optimizer_state_dict' in checkpoint:
            current_learning_rate = checkpoint.get('current_learning_rate', current_learning_rate)
            warm_up_steps = checkpoint.get('warm_up_steps', warm_up_steps)
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except ValueError as exc:
                logging.warning('Skipping optimizer state due to mismatch: %s', exc)
        else:
            logging.info('Skipping optimizer state; starting optimizer fresh.')
    else:
        if args.compiled_init_dir:
            logging.info('Using compiled initialization; skipping random init for %s.', args.model)
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
        logging.info('weight_decay = %.2e' % args.weight_decay)

        training_logs = []

        def _plm_teacher_scores(batch, mode, kd_neg_count):
            base_model = base_model_driver
            plm_ent = getattr(base_model, 'plm_entity_vectors', None)
            if plm_ent is None:
                return None
            plm_rel = getattr(base_model, 'plm_relation_vectors', None)
            if plm_rel is None:
                # Fallback: zero relation vectors if not provided.
                plm_rel = torch.zeros((base_model.nrelation, plm_ent.size(1)), device=plm_ent.device)
            positive_sample, negative_sample = batch[0], batch[1]
            heads = positive_sample[:, 0].to(plm_ent.device)
            rels = positive_sample[:, 1].to(plm_ent.device)
            tails = positive_sample[:, 2].to(plm_ent.device)
            if mode == 'tail-batch':
                query = plm_ent.index_select(0, heads) + plm_rel.index_select(0, rels)
                pos_vec = plm_ent.index_select(0, tails)
                neg_ids = negative_sample[:, :kd_neg_count].to(plm_ent.device)
                neg_vec = plm_ent.index_select(0, neg_ids.view(-1)).view(neg_ids.size(0), -1, plm_ent.size(1))
            elif mode == 'head-batch':
                query = plm_ent.index_select(0, tails) - plm_rel.index_select(0, rels)
                pos_vec = plm_ent.index_select(0, heads)
                neg_ids = negative_sample[:, :kd_neg_count].to(plm_ent.device)
                neg_vec = plm_ent.index_select(0, neg_ids.view(-1)).view(neg_ids.size(0), -1, plm_ent.size(1))
            else:
                return None

            if args.plm_teacher_cosine:
                query = F.normalize(query, dim=-1)
                pos_vec = F.normalize(pos_vec, dim=-1)
                neg_vec = F.normalize(neg_vec, dim=-1)

            temp = max(1e-6, float(args.plm_teacher_temperature))
            pos_scores = (query * pos_vec).sum(dim=-1) / temp
            neg_scores = (query.unsqueeze(1) * neg_vec).sum(dim=-1) / temp
            mask = torch.ones(pos_scores.size(0), dtype=torch.bool, device=pos_scores.device)
            return {
                'positive': pos_scores,
                'negative': neg_scores,
                'neg_count': kd_neg_count,
                'mask': mask
            }
        
        grad_steps = max(1, args.gradient_accumulation_steps)
        #Training Loop
        stop_training = False
        ramp_start = args.ramp_start_step or 0
        ramp_end = args.ramp_end_step or 0
        ramp_enabled = ramp_start > 0 and ramp_end > ramp_start
        base_phase_scale = args.phase_weight_scale
        base_phase_sharp = args.phase_sharpness
        base_mod_sharp = args.modulus_sharpness
        base_adv_temp = args.adversarial_temperature
        base_neg_size = args.negative_sample_size
        base_path_weight = args.path_loss_weight
        for step in range(init_step, args.max_steps):
            if ramp_enabled:
                if args.phase_weight_scale_target is not None:
                    args.phase_weight_scale = _ramp_value(
                        step, ramp_start, ramp_end, base_phase_scale, args.phase_weight_scale_target
                    )
                if args.phase_sharpness_target is not None:
                    args.phase_sharpness = _ramp_value(
                        step, ramp_start, ramp_end, base_phase_sharp, args.phase_sharpness_target
                    )
                if args.modulus_sharpness_target is not None:
                    args.modulus_sharpness = _ramp_value(
                        step, ramp_start, ramp_end, base_mod_sharp, args.modulus_sharpness_target
                    )
                if args.adversarial_temperature_target is not None:
                    args.adversarial_temperature = _ramp_value(
                        step, ramp_start, ramp_end, base_adv_temp, args.adversarial_temperature_target
                    )
                if args.path_loss_weight_target is not None:
                    args.path_loss_weight = _ramp_value(
                        step, ramp_start, ramp_end, base_path_weight, args.path_loss_weight_target
                    )
                if args.negative_sample_size_target is not None:
                    new_neg = int(round(_ramp_value(
                        step, ramp_start, ramp_end, base_neg_size, args.negative_sample_size_target
                    )))
                    if new_neg != train_dataset_head.negative_sample_size:
                        train_dataset_head.negative_sample_size = new_neg
                        train_dataset_tail.negative_sample_size = new_neg
            if (base_model_driver.use_region_head
                    and args.region_blend_final_weight is not None
                    and base_model_driver.region_blend_weight != args.region_blend_final_weight):
                if args.region_blend_warmup_steps > 0:
                    progress = min(1.0, max(0, step) / max(1, args.region_blend_warmup_steps))
                else:
                    progress = 1.0
                target = args.region_blend_weight_start + (
                    args.region_blend_final_weight - args.region_blend_weight_start
                ) * progress
                base_model_driver.region_blend_weight = target
            while next_hard_negative_step is not None and step >= next_hard_negative_step:
                prev_size = train_dataset_head.negative_sample_size
                multiplier = max(1.0, args.hard_negative_multiplier)
                boosted_size = int(math.ceil(prev_size * multiplier))
                if boosted_size == prev_size:
                    boosted_size = prev_size + 1
                if args.max_negative_sample_size:
                    boosted_size = min(boosted_size, args.max_negative_sample_size)
                train_dataset_head.negative_sample_size = boosted_size
                train_dataset_tail.negative_sample_size = boosted_size
                current_negative_sample_size = boosted_size
                logging.info(
                    'Hard-negative schedule applied at step %d: negative_sample_size -> %d (prev %d)',
                    step,
                    boosted_size,
                    prev_size
                )
                hard_negative_idx += 1
                next_hard_negative_step = (
                    hard_negative_steps[hard_negative_idx]
                    if hard_negative_idx < len(hard_negative_steps) else None
                )

            micro_logs = []
            for accum_idx in range(grad_steps):
                batch = next(train_iterator)
                path_batch = None
                path_weight = args.path_loss_weight
                if path_iterator and should_enable_path_loss(step, args):
                    path_batch = next(path_iterator)
                else:
                    path_weight = 0.0

                teacher_payload = None
                if teacher_model and args.kd_lambda > 0.0:
                    negative_sample = batch[1]
                    kd_neg_count = min(args.teacher_negatives, negative_sample.size(1))
                    if kd_neg_count > 0:
                        teacher_payload = teacher_model.score_tail_batch(
                            batch[0],
                            negative_sample[:, :kd_neg_count],
                            batch[3]
                        )
                elif args.plm_teacher and args.kd_lambda > 0.0:
                    negative_sample = batch[1]
                    kd_neg_count = min(args.teacher_negatives, negative_sample.size(1))
                    if kd_neg_count > 0:
                        teacher_payload = _plm_teacher_scores(batch, batch[3], kd_neg_count)

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
                    accumulation_steps=grad_steps,
                    batch=batch,
                    teacher_scores=teacher_payload
                )
                if accum_idx == grad_steps - 1 and ema_helper is not None:
                    ema_helper.update(base_model_driver)
                micro_logs.append(micro_log)

            log = {}
            all_keys = set()
            for entry in micro_logs:
                all_keys.update(entry.keys())
            for key in all_keys:
                vals = [m[key] for m in micro_logs if key in m]
                if not vals:
                    continue
                log[key] = sum(vals) / len(vals)

            scheduler.step()   # Smooth cosine update
            current_learning_rate = optimizer.param_groups[0]['lr']

            if (next_warmup_step is not None
                and step >= next_warmup_step
                and args.warm_up_factor < 1.0):
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.warm_up_factor
                scheduler.base_lrs = [base * args.warm_up_factor for base in scheduler.base_lrs]
                if hasattr(scheduler, '_last_lr'):
                    scheduler._last_lr = [lr * args.warm_up_factor for lr in scheduler._last_lr]
                current_learning_rate = optimizer.param_groups[0]['lr']
                logging.info(
                    'Warm-up LR drop applied at step %d: Learning Rate = %.6e',
                    step,
                    current_learning_rate
                )
                if args.warm_up_multiplier > 1.0:
                    next_warmup_step = int(next_warmup_step * args.warm_up_multiplier)
                else:
                    next_warmup_step = None

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
                logging.info(
                    "Step %d: Learning Rate = %.8f | negative_sample_size = %d",
                    step,
                    current_learning_rate,
                    train_dataset_head.negative_sample_size
                )


                # training_logs = []

            # # Add this at the top of the training loop
            # best_val_mrr = 0.0
            # best_step = 0 
                
            if step % args.valid_steps == 0:
                if ema_helper is not None and args.eval_with_ema:
                    ema_helper.apply_shadow(base_model_driver)
                if args.murp_style_reporting:
                    if args.do_test:
                        logging.info('Evaluating on Test Dataset (MuRP-style reporting)...')
                        metrics = KGEModel.test_step(kge_model, test_triples, all_true_triples, args)
                        log_metrics('Test', step, metrics)
                    elif args.do_valid:
                        logging.info('MuRP-style reporting enabled, but --do_test is disabled. Falling back to validation split.')
                        metrics = KGEModel.test_step(kge_model, valid_triples, all_true_triples, args)
                        log_metrics('Valid', step, metrics)
                elif args.do_valid:
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
                        plateau_lr_counter = 0

                        save_variable_list = {
                            'step': step,
                            'current_learning_rate': current_learning_rate,
                            'warm_up_steps': warm_up_steps
                        }
                        save_model(kge_model, optimizer, save_variable_list, args)
                    else:
                        plateau_lr_counter += 1
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

                        should_drop_for_plateau = (
                            args.plateau_lr_factor < 1.0
                            and plateau_lr_counter >= max(1, args.plateau_lr_patience)
                            and step >= args.plateau_lr_start_step
                            and current_learning_rate > args.plateau_lr_min + 1e-12
                        )
                        if should_drop_for_plateau:
                            old_lr = optimizer.param_groups[0]['lr']
                            new_lr = max(old_lr * args.plateau_lr_factor, args.plateau_lr_min)
                            if new_lr < old_lr - 1e-12:
                                scale = new_lr / max(old_lr, 1e-12)
                                for param_group in optimizer.param_groups:
                                    param_group['lr'] = new_lr
                                scheduler.base_lrs = [max(base * scale, args.plateau_lr_min) for base in scheduler.base_lrs]
                                if hasattr(scheduler, '_last_lr'):
                                    scheduler._last_lr = [max(lr * scale, args.plateau_lr_min) for lr in scheduler._last_lr]
                                current_learning_rate = new_lr
                                plateau_lr_counter = 0
                                logging.info(
                                    'Validation plateau triggered LR drop at step %d (MRR %.4f -> %.4f). '
                                    'Learning Rate now %.6e.',
                                    step,
                                    best_val_mrr,
                                    metrics['MRR'],
                                    current_learning_rate
                                )
                if ema_helper is not None and args.eval_with_ema:
                    ema_helper.restore(base_model_driver)


        if stop_training:
            logging.info('Stopping training loop due to early stopping condition.')

        # save_variable_list = {
        #     'step': step, 
        #     'current_learning_rate': current_learning_rate,
        #     'warm_up_steps': warm_up_steps
        # }
        # save_model(kge_model, optimizer, save_variable_list, args)
        
    if args.do_valid and not args.murp_style_reporting:
        logging.info('Evaluating on Valid Dataset...')
        metrics = KGEModel.test_step(kge_model, valid_triples, all_true_triples, args)
        log_metrics('Valid', step, metrics)
    
    # if args.do_test:
    #     logging.info('Evaluating on Test Dataset...')
    #     metrics = kge_model.test_step(kge_model, test_triples, all_true_triples, args)
    #     log_metrics('Test', step, metrics)

    # After training, load the best model before testing
    if args.do_test:
        if args.murp_style_reporting:
            logging.info('MuRP-style reporting: evaluating current model on Test Dataset (no best-valid checkpoint reload).')
            metrics = KGEModel.test_step(kge_model, test_triples, all_true_triples, args)
            log_metrics('Test', step, metrics)
        else:
            logging.info(f" Using best validation model from step {best_step} for final test evaluation.")
            ckpt_path = args.save_path or args.init_checkpoint
            if not ckpt_path:
                raise ValueError('No checkpoint path provided: set --save_path during training or pass --init_checkpoint.')
            if os.path.isdir(ckpt_path):
                ckpt_path = os.path.join(ckpt_path, 'checkpoint')
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            model_to_load = kge_model.module if hasattr(kge_model, 'module') else kge_model
            model_to_load.load_state_dict(checkpoint['model_state_dict'], strict=False)
            metrics = KGEModel.test_step(kge_model, test_triples, all_true_triples, args)
            log_metrics('Test', best_step, metrics)
    
    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics = KGEModel.test_step(kge_model, train_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)
if __name__ == '__main__':
    main(parse_args())
