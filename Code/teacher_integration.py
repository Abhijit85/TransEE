import argparse
import collections
import json
import os
import sys
from types import SimpleNamespace
from typing import Dict, Optional

import torch
from transformers import AutoConfig, BertTokenizer
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()


class SimKGCTeacher:
    """
    Light-weight wrapper around the SimKGC predictor that exposes
    per-triple logits for RelatE's distillation loop.
    """

    def __init__(
        self,
        checkpoint_path: str,
        repo_root: str,
        device: torch.device,
        max_negatives: int = 64,
        id2entity: Optional[Dict[int, str]] = None,
        id2relation: Optional[Dict[int, str]] = None,
    ):
        self.repo_root = os.path.abspath(repo_root)
        self.checkpoint = os.path.abspath(checkpoint_path)
        self.device = torch.device(device)
        self.max_negatives = max(1, max_negatives)
        self.id2entity = id2entity or {}
        self.id2relation = id2relation or {}

        self.ckpt_meta = torch.load(self.checkpoint, map_location='cpu')
        self.teacher_args = self.ckpt_meta.get('args', {})
        self._load_teacher_modules()
        self.predictor = self.BertPredictor()
        self.predictor.load(self.checkpoint, use_data_parallel=False)

        # Respect the requested device even if CUDA is visible.
        if self.device.type == 'cuda' and torch.cuda.is_available():
            self.predictor.model = self.predictor.model.cuda()
            self.predictor.use_cuda = True
        else:
            self.predictor.model = self.predictor.model.cpu()
            self.predictor.use_cuda = False
        self.scale = float(self.predictor.model.log_inv_t.detach().exp().cpu())

        self.entity_dict = self.get_entity_dict()
        self.entity_to_idx = self.entity_dict.entity2idx
        with torch.no_grad():
            entity_vecs = self.predictor.predict_by_entities(self.entity_dict.entity_exs)
        self.entity_vectors = entity_vecs.to(self.device)
        self.query_dim = self.entity_vectors.size(1)

    def _load_teacher_modules(self):
        if self.repo_root not in sys.path:
            sys.path.insert(0, self.repo_root)
        original_argv = sys.argv
        def _resolve_teacher_path(path_value, fallback_name):
            fallback = os.path.join(self.repo_root, 'data', task, fallback_name)
            if not path_value:
                return fallback
            candidate = path_value
            if os.path.exists(candidate):
                return candidate
            if not os.path.isabs(candidate):
                joined = os.path.join(self.repo_root, candidate)
                if os.path.exists(joined):
                    return joined
            # Remap stale absolute roots by preserving suffix from "/data/...".
            marker = f'{os.sep}data{os.sep}'
            if marker in candidate:
                suffix = candidate.split(marker, 1)[1]
                remapped = os.path.join(self.repo_root, 'data', suffix)
                if os.path.exists(remapped):
                    return remapped
            return fallback

        train_path = self.teacher_args.get('train_path')
        valid_path = self.teacher_args.get('valid_path', train_path)
        task = self.teacher_args.get('task', 'WN18RR')
        train_path = _resolve_teacher_path(train_path, 'train.txt.json')
        valid_path = _resolve_teacher_path(valid_path, 'valid.txt.json')
        argv = [
            original_argv[0],
            '--model-dir', os.path.dirname(self.checkpoint),
            '--train-path', train_path,
            '--valid-path', valid_path,
            '--task', task
        ]
        try:
            sys.argv = argv
            from predict import BertPredictor  # type: ignore
            from doc import Example  # type: ignore
            from dict_hub import get_entity_dict  # type: ignore
        finally:
            sys.argv = original_argv

        self.BertPredictor = BertPredictor  # type: ignore[attr-defined]
        self.Example = Example  # type: ignore[attr-defined]
        self.get_entity_dict = get_entity_dict  # type: ignore[attr-defined]

    def _build_examples(self, positive_sample: torch.Tensor, mode: str):
        examples = []
        missing_rows = set()
        for idx in range(positive_sample.size(0)):
            head_id = self.id2entity.get(int(positive_sample[idx, 0].item()))
            relation_str = self.id2relation.get(int(positive_sample[idx, 1].item()))
            tail_id = self.id2entity.get(int(positive_sample[idx, 2].item()))
            if head_id is None or tail_id is None or relation_str is None:
                missing_rows.add(idx)
                continue
            examples.append((idx, self.Example(head_id=head_id, relation=relation_str, tail_id=tail_id)))
        return examples, missing_rows

    def _gather_entity_indices(self, entity_ids: torch.Tensor):
        indices = torch.full(entity_ids.shape, -1, dtype=torch.long, device=entity_ids.device)
        for idx in range(entity_ids.size(0)):
            ent_id = self.id2entity.get(int(entity_ids[idx].item()))
            if ent_id is None:
                continue
            ent_idx = self.entity_to_idx.get(ent_id, -1)
            indices[idx] = ent_idx
        return indices

    def score_tail_batch(self, positive_sample: torch.Tensor, negative_sample: torch.Tensor, mode: str):
        if mode != 'tail-batch' or negative_sample.numel() == 0:
            return None

        # Build examples for the current positives.
        examples, missing_rows = self._build_examples(positive_sample, mode)
        if not examples:
            return None

        ordered_idx = [idx for idx, _ in examples]
        normalized_examples = [ex for _, ex in examples]
        with torch.no_grad():
            hr_vectors, _ = self.predictor.predict_by_examples(normalized_examples)
        hr_vectors = hr_vectors.to(self.device)

        batch_size = positive_sample.size(0)
        neg_limit = min(self.max_negatives, negative_sample.size(1))
        neg_subset = negative_sample[:, :neg_limit].to(positive_sample.device)

        pos_scores = torch.full((batch_size,), 0.0, device=self.device)
        neg_scores = torch.full((batch_size, neg_limit), 0.0, device=self.device)
        mask = torch.zeros(batch_size, dtype=torch.bool)
        query_dim = hr_vectors.size(1)
        query_vectors = torch.zeros(batch_size, query_dim, dtype=hr_vectors.dtype, device=self.device)

        entity_ids = torch.tensor(
            [positive_sample[idx, 2].item() for idx in ordered_idx],
            dtype=torch.long,
            device=positive_sample.device,
        )
        pos_idx_tensor = self._gather_entity_indices(entity_ids.cpu())
        if (pos_idx_tensor < 0).any():
            return None
        pos_embeddings = self.entity_vectors[pos_idx_tensor.to(self.device)]
        pos_vals = torch.sum(hr_vectors * pos_embeddings, dim=1) * self.scale
        for local_idx, global_idx in enumerate(ordered_idx):
            mask[global_idx] = True
            pos_scores[global_idx] = pos_vals[local_idx]
            query_vectors[global_idx] = hr_vectors[local_idx]

        if neg_limit > 0:
            neg_idx_list = []
            for row in range(batch_size):
                if row in missing_rows:
                    neg_idx_list.append(torch.full((neg_limit,), -1, dtype=torch.long))
                    continue
                ent_row = neg_subset[row].cpu()
                mapped = self._gather_entity_indices(ent_row)
                neg_idx_list.append(mapped)
            neg_idx_tensor = torch.stack(neg_idx_list, dim=0)
            valid_mask = (neg_idx_tensor >= 0) & mask.unsqueeze(1)
        neg_indices = neg_idx_tensor.clamp(min=0).to(self.device)
        entity_vecs = self.entity_vectors[neg_indices.view(-1)].view(batch_size, neg_limit, -1)
        hr_for_neg = hr_vectors.new_zeros((batch_size, neg_limit, hr_vectors.size(1)))
        for local_idx, global_idx in enumerate(ordered_idx):
            hr_for_neg[global_idx] = hr_vectors[local_idx].unsqueeze(0).expand(neg_limit, -1)
        neg_vals = torch.sum(hr_for_neg * entity_vecs, dim=-1) * self.scale
        neg_vals = torch.where(valid_mask.to(self.device), neg_vals, torch.zeros_like(neg_vals))
        neg_scores.copy_(neg_vals)

        return {
            'positive': pos_scores,
            'negative': neg_scores,
            'mask': mask,
            'neg_count': neg_limit,
            'query_vectors': query_vectors,
        }


class MuRPTeacher:
    """
    Lightweight inference wrapper around the MuRP/MuRE hyperbolic models.
    Loads a checkpoint saved from teachers/murp and exposes per-triple logits.
    """

    def __init__(
        self,
        checkpoint_path: str,
        repo_root: str,
        data_dir: str,
        device: torch.device,
        max_negatives: int = 64,
        id2entity: Optional[Dict[int, str]] = None,
        id2relation: Optional[Dict[int, str]] = None,
    ):
        self.repo_root = os.path.abspath(repo_root)
        if self.repo_root not in sys.path:
            sys.path.insert(0, self.repo_root)
        from load_data import Data  # type: ignore
        from utils import p_sum, p_log_map, p_exp_map, artanh  # type: ignore

        self.Data = Data
        self.p_sum = p_sum
        self.p_log_map = p_log_map
        self.p_exp_map = p_exp_map
        self.artanh = artanh

        data_dir = os.path.abspath(data_dir)
        if not data_dir.endswith(os.sep):
            data_dir = data_dir + os.sep
        self.data_dir = data_dir
        self.device = torch.device(device)
        self.max_negatives = max(1, max_negatives)
        self.id2entity = id2entity or {}
        self.id2relation = id2relation or {}
        self.ball_eps = 1e-5

        self.dataset = self.Data(data_dir=self.data_dir)
        self.entity_list = self.dataset.entities
        self.relation_list = self.dataset.relations
        self.entity_to_idx = {name: idx for idx, name in enumerate(self.entity_list)}
        self.relation_to_idx = {name: idx for idx, name in enumerate(self.relation_list)}

        ckpt = torch.load(os.path.abspath(checkpoint_path), map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        if 'Eh.weight' in state_dict:
            self.model_type = ckpt.get('model_type', 'poincare')
            entity_weight = state_dict['Eh.weight']
            relation_vec = state_dict['rvh.weight']
            self.mode = 'murp'
        else:
            self.model_type = ckpt.get('model_type', 'euclidean')
            entity_weight = state_dict['E.weight']
            relation_vec = state_dict['rv.weight']
            self.mode = 'mure'
        self.entity_dim = entity_weight.size(1)
        self.Eh = entity_weight.to(torch.double).to(self.device)
        self.Wu = state_dict['Wu'].to(torch.double).to(self.device)
        self.bs = state_dict['bs'].to(torch.double).to(self.device)
        self.bo = state_dict['bo'].to(torch.double).to(self.device)
        if self.mode == 'murp':
            self.rvh = relation_vec.to(torch.double).to(self.device)
        else:
            self.rv = relation_vec.to(torch.double).to(self.device)
        self.entity_vectors = self.Eh.float().cpu()

    def _lookup_entity(self, idx: int) -> int:
        name = self.id2entity.get(idx)
        if name is None:
            return -1
        return self.entity_to_idx.get(name, -1)

    def _lookup_relation(self, idx: int) -> int:
        name = self.id2relation.get(idx)
        if name is None:
            return -1
        rel_idx = self.relation_to_idx.get(name)
        if rel_idx is not None:
            return rel_idx
        reverse_key = f'{name}_reverse' if name else None
        if reverse_key:
            rel_idx = self.relation_to_idx.get(reverse_key)
        return rel_idx if rel_idx is not None else -1

    def _project_ball(self, tensor: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(tensor, dim=-1, keepdim=True)
        mask = norm >= 1.0
        if mask.any():
            tensor = tensor.clone()
            tensor[mask] = tensor[mask] / (norm[mask] - self.ball_eps)
        return tensor

    def _murp_forward(self, head_idx: torch.Tensor, rel_idx: torch.Tensor, tail_idx: torch.Tensor):
        u = self.Eh.index_select(0, head_idx)
        v = self.Eh.index_select(0, tail_idx)
        Ru = self.Wu.index_select(0, rel_idx)
        rv = self.rvh.index_select(0, rel_idx)

        u = self._project_ball(u)
        v = self._project_ball(v)
        rv = self._project_ball(rv)

        u_e = self.p_log_map(u)
        u_W = u_e * Ru
        u_m = self.p_exp_map(u_W)
        v_m = self.p_sum(v, rv)

        u_m = self._project_ball(u_m)
        v_m = self._project_ball(v_m)

        diff = self.p_sum(-u_m, v_m)
        norm = torch.clamp(torch.norm(diff, dim=-1), 1e-10, 1 - self.ball_eps)
        sqdist = (2.0 * self.artanh(norm)) ** 2
        return -sqdist + self.bs.index_select(0, head_idx) + self.bo.index_select(0, tail_idx)

    def _mure_forward(self, head_idx: torch.Tensor, rel_idx: torch.Tensor, tail_idx: torch.Tensor):
        u = self.Eh.index_select(0, head_idx)
        v = self.Eh.index_select(0, tail_idx)
        Ru = self.Wu.index_select(0, rel_idx)
        rv = self.rv.index_select(0, rel_idx)
        u_W = u * Ru
        sqdist = torch.sum((u_W - (v + rv)) ** 2, dim=-1)
        return -sqdist + self.bs.index_select(0, head_idx) + self.bo.index_select(0, tail_idx)

    def _forward_scores(self, head_idx, rel_idx, tail_idx):
        if self.mode == 'murp':
            return self._murp_forward(head_idx, rel_idx, tail_idx)
        return self._mure_forward(head_idx, rel_idx, tail_idx)

    def score_tail_batch(self, positive_sample: torch.Tensor, negative_sample: torch.Tensor, mode: str):
        if mode != 'tail-batch' or negative_sample.numel() == 0:
            return None

        batch_size = positive_sample.size(0)
        neg_limit = min(self.max_negatives, negative_sample.size(1))
        pos_scores = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        neg_scores = torch.zeros(batch_size, neg_limit, dtype=torch.float32, device=self.device)
        mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for row in range(batch_size):
            head_idx = self._lookup_entity(int(positive_sample[row, 0].item()))
            rel_idx = self._lookup_relation(int(positive_sample[row, 1].item()))
            tail_idx = self._lookup_entity(int(positive_sample[row, 2].item()))
            if min(head_idx, rel_idx, tail_idx) < 0:
                continue

            head_tensor = torch.tensor([head_idx], dtype=torch.long, device=self.device)
            rel_tensor = torch.tensor([rel_idx], dtype=torch.long, device=self.device)
            tail_tensor = torch.tensor([tail_idx], dtype=torch.long, device=self.device)
            score = self._forward_scores(head_tensor, rel_tensor, tail_tensor)
            pos_scores[row] = score.float()
            mask[row] = True

            if neg_limit > 0:
                neg_ids = []
                for cand in negative_sample[row, :neg_limit]:
                    mapped = self._lookup_entity(int(cand.item()))
                    neg_ids.append(mapped)
                neg_ids = torch.tensor(neg_ids, dtype=torch.long, device=self.device)
                valid_mask = neg_ids >= 0
                if valid_mask.any():
                    neg_head = head_tensor.expand(valid_mask.sum().item())
                    neg_rel = rel_tensor.expand(valid_mask.sum().item())
                    neg_tail = neg_ids[valid_mask]
                    neg_vals = self._forward_scores(neg_head, neg_rel, neg_tail)
                    neg_scores[row, valid_mask] = neg_vals.float()

        if not mask.any():
            return None

        return {
            'positive': pos_scores,
            'negative': neg_scores,
            'mask': mask,
            'neg_count': neg_limit,
        }


class CSPromTeacher:
    """
    Wrapper around CSProm-KG (Conditional Soft Prompting) to expose per-triple logits
    for RelatE's distillation loop.
    """

    def __init__(
        self,
        checkpoint_path: str,
        repo_root: str,
        dataset: str,
        data_dir: Optional[str],
        device: torch.device,
        max_negatives: int = 64,
        id2entity: Optional[Dict[int, str]] = None,
        id2relation: Optional[Dict[int, str]] = None,
        config_overrides: Optional[Dict[str, object]] = None,
    ):
        self.repo_root = os.path.abspath(repo_root)
        if self.repo_root not in sys.path:
            sys.path.insert(0, self.repo_root)
        self.checkpoint = os.path.abspath(checkpoint_path)
        self.device = torch.device(device)
        self.max_negatives = max(1, max_negatives)
        self.id2entity = id2entity or {}
        self.id2relation = id2relation or {}
        self.dataset = dataset
        self.data_dir = os.path.abspath(data_dir) if data_dir else None
        self.config_overrides = config_overrides or {}

        from models.P_model import KGCPromptTuner  # type: ignore

        self.KGCPromptTuner = KGCPromptTuner
        ckpt_meta = torch.load(self.checkpoint, map_location='cpu', weights_only=False)
        configs = self._build_configs(ckpt_meta)
        self.configs = configs
        self.ent_names, self.rel_names, self.ent_descs = self._load_text_fields()
        self.entity_to_idx = self._load_id_map('entity2id.txt')
        self.relation_to_idx = self._load_id_map('relation2id.txt')
        self.entity_id_map = self._build_student_map(self.id2entity, self.entity_to_idx)
        self.relation_id_map = self._build_student_map(self.id2relation, self.relation_to_idx)
        cache_dir = os.getenv('TRANSFORMERS_CACHE')
        if not cache_dir:
            base_cache = os.getenv('HF_HOME')
            if base_cache:
                cache_dir = os.path.join(base_cache, 'hub')
        self.tokenizer = BertTokenizer.from_pretrained(
            configs.pretrained_model,
            add_prefix_space=False,
            local_files_only=True,
            cache_dir=cache_dir
        )

        text_dict = {
            'ent_names': self.ent_names,
            'rel_names': self.rel_names,
            'ent_descs': self.ent_descs,
        }
        gt = {'all_tail_gt': {}, 'all_head_gt': {}}
        safe_globals = None
        allowed = [argparse.Namespace, collections.defaultdict, list, dict]
        if hasattr(torch.serialization, 'safe_globals'):
            safe_globals = torch.serialization.safe_globals(allowed)
        elif hasattr(torch.serialization, 'add_safe_globals'):
            torch.serialization.add_safe_globals(allowed)
        if safe_globals:
            with safe_globals:
                self.model = self.KGCPromptTuner.load_from_checkpoint(
                    self.checkpoint,
                    strict=False,
                    configs=configs,
                    text_dict=text_dict,
                    gt=gt
                )
        else:
            self.model = self.KGCPromptTuner.load_from_checkpoint(
                self.checkpoint,
                strict=False,
                configs=configs,
                text_dict=text_dict,
                gt=gt
            )
        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            self.entity_vectors = self.model.ent_embed.weight.detach().cpu()
        self.query_dim = self.entity_vectors.size(1)

    def _build_configs(self, ckpt_meta: Dict[str, object]) -> SimpleNamespace:
        config_payload = {}
        hparams = ckpt_meta.get('hyper_parameters') or ckpt_meta.get('hparams') or {}
        if isinstance(hparams, dict):
            config_payload = hparams.get('configs') or hparams.get('config') or {}
            if not config_payload:
                config_payload = hparams
        if hasattr(config_payload, '__dict__'):
            config_payload = vars(config_payload)
        if not isinstance(config_payload, dict):
            config_payload = {}

        config_payload.update(self._load_config_file())
        config_payload.update(self.config_overrides)

        if not self.data_dir:
            raise ValueError('CSProm-KG data directory is required to load entity/relation text fields.')
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f'CSProm-KG data directory not found: {self.data_dir}')

        n_ent = self._read_count('entity2id.txt')
        n_rel = self._read_count('relation2id.txt')

        def _get(name, default):
            return config_payload.get(name, default)

        configs = SimpleNamespace()
        configs.dataset = self.dataset
        configs.n_ent = n_ent
        configs.n_rel = n_rel
        configs.pretrained_model = _get('pretrained_model', 'bert-base-uncased')
        configs.embed_dim = int(_get('embed_dim', 128))
        configs.prompt_length = int(_get('prompt_length', 0))
        configs.prompt_hidden_dim = int(_get('prompt_hidden_dim', max(1, configs.embed_dim // 2)))
        configs.graph_model = _get('graph_model', 'conve')
        configs.desc_max_length = int(_get('desc_max_length', 0))
        configs.text_len = int(_get('text_len', 72))
        configs.n_lar = int(_get('n_lar', 0))
        configs.use_fp16 = bool(_get('use_fp16', False))
        configs.use_speedup = bool(_get('use_speedup', False))
        configs.alpha = float(_get('alpha', 0.0))
        configs.alpha_step = float(_get('alpha_step', 0.0))
        configs.gamma = float(_get('gamma', 1.0))
        configs.hid_drop = float(_get('hid_drop', 0.3))
        configs.hid_drop2 = float(_get('hid_drop2', 0.3))
        configs.feat_drop = float(_get('feat_drop', 0.3))
        configs.k_w = int(_get('k_w', 8))
        configs.k_h = int(_get('k_h', 16))
        configs.num_filt = int(_get('num_filt', 200))
        configs.ker_sz = int(_get('ker_sz', 7))
        configs.bias = bool(_get('bias', False))
        configs.label_smoothing = float(_get('label_smoothing', 0.0))
        configs.loss_gamma = float(_get('loss_gamma', 0.0))
        configs.is_temporal = 'ICEWS' in configs.dataset
        cache_dir = os.getenv('TRANSFORMERS_CACHE')
        if not cache_dir:
            base_cache = os.getenv('HF_HOME')
            if base_cache:
                cache_dir = os.path.join(base_cache, 'hub')
        configs.vocab_size = AutoConfig.from_pretrained(
            configs.pretrained_model,
            local_files_only=True,
            cache_dir=cache_dir
        ).vocab_size
        configs.model_dim = AutoConfig.from_pretrained(
            configs.pretrained_model,
            local_files_only=True,
            cache_dir=cache_dir
        ).hidden_size
        return configs

    def _load_config_file(self) -> Dict[str, object]:
        config_path = self.config_overrides.get('config_path')
        if not config_path:
            return {}
        with open(config_path, 'r', encoding='utf-8') as handle:
            return json.load(handle)

    def _read_count(self, filename: str) -> int:
        with open(os.path.join(self.data_dir, filename), encoding='utf-8') as handle:
            first_line = handle.readline().strip()
        return int(first_line)

    def _load_id_map(self, filename: str) -> Dict[str, int]:
        mapping = {}
        with open(os.path.join(self.data_dir, filename), encoding='utf-8') as handle:
            lines = handle.read().strip().split('\n')
        for line in lines[1:]:
            name, idx = line.split('\t')
            mapping[name] = int(idx)
        return mapping

    def _load_text_fields(self):
        ent_names = self._load_named_file('entityid2name.txt')
        rel_names = self._load_named_file('relationid2name.txt')
        ent_descs = self._load_named_file('entityid2description.txt', keep_spaces=True)
        return ent_names, rel_names, ent_descs

    def _load_named_file(self, filename: str, keep_spaces: bool = False):
        entries = []
        with open(os.path.join(self.data_dir, filename), encoding='utf-8') as handle:
            lines = handle.read().strip('\n').split('\n')
        for line in lines[1:]:
            _, name = line.split('\t')
            if keep_spaces:
                name = ' '.join(name.split(' '))
            entries.append(name)
        return entries

    @staticmethod
    def _build_student_map(id2value: Dict[int, str], teacher_map: Dict[str, int]):
        max_id = max(id2value.keys(), default=-1)
        mapped = [-1] * (max_id + 1)
        for idx, value in id2value.items():
            mapped[idx] = teacher_map.get(value, -1)
        return mapped

    def _construct_input_text(self, entity_idx: int, relation_idx: int):
        src_name = self.ent_names[entity_idx] if entity_idx >= 0 else ''
        if self.dataset == 'WN18RR':
            src_name = ' '.join(src_name.split(' , ')[:-2])
        src_desc = ':' + self.ent_descs[entity_idx] if self.configs.desc_max_length > 0 else ''
        rel_name = self.rel_names[relation_idx] if relation_idx >= 0 else ''
        src = (src_name + ' ' + src_desc).strip()
        return src, rel_name

    def _map_ids(self, ids: torch.Tensor, mapping: list) -> torch.Tensor:
        flat = ids.reshape(-1).tolist()
        mapped = [mapping[val] if val < len(mapping) else -1 for val in flat]
        return torch.tensor(mapped, dtype=torch.long, device=ids.device).view(ids.shape)

    def score_tail_batch(self, positive_sample: torch.Tensor, negative_sample: torch.Tensor, mode: str):
        if mode != 'tail-batch' or negative_sample.numel() == 0:
            return None

        batch_size = positive_sample.size(0)
        neg_limit = min(self.max_negatives, negative_sample.size(1))
        pos_scores = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        neg_scores = torch.zeros(batch_size, neg_limit, dtype=torch.float32, device=self.device)
        mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        head_ids = self._map_ids(positive_sample[:, 0], self.entity_id_map)
        rel_ids = self._map_ids(positive_sample[:, 1], self.relation_id_map)
        tail_ids = self._map_ids(positive_sample[:, 2], self.entity_id_map)
        neg_ids = self._map_ids(negative_sample[:, :neg_limit], self.entity_id_map)

        valid_rows = (head_ids >= 0) & (rel_ids >= 0) & (tail_ids >= 0)
        if not valid_rows.any():
            return None

        valid_idx = valid_rows.nonzero(as_tuple=False).view(-1)
        ent_rel = torch.stack([head_ids[valid_idx], rel_ids[valid_idx]], dim=1).to(self.device)
        text_pairs = [self._construct_input_text(int(head_ids[i].item()), int(rel_ids[i].item())) for i in valid_idx]
        src_texts = [src for src, _ in text_pairs]
        rel_texts = [rel for _, rel in text_pairs]
        tokenized = self.tokenizer(
            src_texts,
            text_pair=rel_texts,
            max_length=self.configs.text_len,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        src_ids = tokenized.input_ids.to(self.device)
        src_mask = tokenized.attention_mask.to(self.device)

        with torch.no_grad():
            logits, pred = self.model(ent_rel, src_ids, src_mask)

        pos_vals = logits.gather(1, tail_ids[valid_idx].unsqueeze(1).to(self.device)).squeeze(1)
        pos_scores[valid_idx] = pos_vals
        mask[valid_idx] = True

        query_vectors = torch.zeros(batch_size, self.query_dim, dtype=logits.dtype, device=self.device)
        query_vectors[valid_idx] = pred.to(self.device)

        if neg_limit > 0:
            neg_subset = neg_ids[valid_idx]
            neg_valid = neg_subset >= 0
            neg_indices = neg_subset.clamp(min=0).to(self.device)
            gathered = logits.gather(1, neg_indices)
            gathered = torch.where(neg_valid.to(self.device), gathered, torch.zeros_like(gathered))
            neg_scores[valid_idx] = gathered

        return {
            'positive': pos_scores,
            'negative': neg_scores,
            'mask': mask,
            'neg_count': neg_limit,
            'query_vectors': query_vectors,
        }
