import os
import logging
import numpy as np
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import BatchType, TestDataset

class KGEModel(nn.Module, ABC):
    """
    Must define
        `self.entity_embedding`
        `self.relation_embedding`
    in the subclasses.
    """

    @abstractmethod
    def func(self, head, rel, tail, batch_type):
        """
        Different tensor shape for different batch types.
        BatchType.SINGLE:
            head: [batch_size, hidden_dim]
            relation: [batch_size, hidden_dim]
            tail: [batch_size, hidden_dim]

        BatchType.HEAD_BATCH:
            head: [batch_size, negative_sample_size, hidden_dim]
            relation: [batch_size, hidden_dim]
            tail: [batch_size, hidden_dim]

        BatchType.TAIL_BATCH:
            head: [batch_size, hidden_dim]
            relation: [batch_size, hidden_dim]
            tail: [batch_size, negative_sample_size, hidden_dim]
        """
        ...

    def forward(self, sample, batch_type=BatchType.SINGLE):
        """
        Given the indexes in `sample`, extract the corresponding embeddings,
        and call func().

        Args:
            batch_type: {SINGLE, HEAD_BATCH, TAIL_BATCH},
                - SINGLE: positive samples in training, and all samples in validation / testing,
                - HEAD_BATCH: (?, r, t) tasks in training,
                - TAIL_BATCH: (h, r, ?) tasks in training.

            sample: different format for different batch types.
                - SINGLE: tensor with shape [batch_size, 3]
                - {HEAD_BATCH, TAIL_BATCH}: (positive_sample, negative_sample)
                    - positive_sample: tensor with shape [batch_size, 3]
                    - negative_sample: tensor with shape [batch_size, negative_sample_size]
        """
        if batch_type == BatchType.SINGLE:
            head = torch.index_select(
                self.entity_modulus_embedding.weight,  # Access the weight tensor
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding.weight,  # Access the weight tensor
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_modulus_embedding.weight,  # Access the weight tensor
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif batch_type == BatchType.HEAD_BATCH:
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_modulus_embedding.weight,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding.weight,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_modulus_embedding.weight,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif batch_type == BatchType.TAIL_BATCH:
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_modulus_embedding.weight,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding.weight,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_modulus_embedding.weight,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('batch_type %s not supported!'.format(batch_type))

        return self.func(head, relation, tail, batch_type)
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propagation and return the loss
        '''
        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_type = next(train_iterator)

        positive_sample = positive_sample.cuda()
        negative_sample = negative_sample.cuda()
        subsampling_weight = subsampling_weight.cuda()

        # negative scores
        negative_score = model((positive_sample, negative_sample), batch_type=batch_type)

        negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                          * F.logsigmoid(-negative_score)).sum(dim=1)

        # positive scores
        positive_score = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        loss.backward()

        optimizer.step()

        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model, data_reader, mode, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        model.eval()

        test_dataloader_head = DataLoader(
            TestDataset(
                data_reader,
                mode,
                BatchType.HEAD_BATCH
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                data_reader,
                mode,
                BatchType.TAIL_BATCH
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataset_list = [test_dataloader_head, test_dataloader_tail]

        logs = []

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, batch_type in test_dataset:
                    positive_sample = positive_sample.cuda()
                    negative_sample = negative_sample.cuda()
                    filter_bias = filter_bias.cuda()

                    batch_size = positive_sample.size(0)

                    score = model((positive_sample, negative_sample), batch_type)
                    score += filter_bias

                    # Explicitly sort all the entities to ensure that there is no test exposure bias
                    argsort = torch.argsort(score, dim=1, descending=True)

                    if batch_type == BatchType.HEAD_BATCH:
                        positive_arg = positive_sample[:, 0]
                    elif batch_type == BatchType.TAIL_BATCH:
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        # Notice that argsort is not ranking
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1

                        # ranking + 1 is the true ranking used in evaluation metrics
                        ranking = 1 + ranking.item()
                        logs.append({
                            'MRR': 1.0 / ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        })

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... ({}/{})'.format(step, total_steps))

                    step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        return metrics

class TransEEnhanced(KGEModel):
    def __init__(self, num_entity, num_relation, hidden_dim, margin, learning_rate=0.001, modulus_weight=1.0, phase_weight=0.5):
        super(TransEEnhanced, self).__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.margin = margin
        self.learning_rate = learning_rate
        self.modulus_weight = modulus_weight
        self.phase_weight = phase_weight

        self.entity_modulus_embedding = nn.Embedding(num_entity, hidden_dim * 2)
        self.relation_embedding = nn.Embedding(num_relation, hidden_dim * 2)  # Use consistent naming here

        nn.init.xavier_uniform_(self.entity_modulus_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)  # Initialize this embedding as well

    def func(self, head, rel, tail, batch_type):
        head_modulus, head_phase = torch.chunk(head, 2, dim=2)
        rel_modulus, rel_phase = torch.chunk(rel, 2, dim=2)
        tail_modulus, tail_phase = torch.chunk(tail, 2, dim=2)

        bias_relation = torch.clamp(rel_modulus, max=1)
        rel_modulus = torch.abs(rel_modulus)
        indicator = (bias_relation < -rel_modulus)
        bias_relation[indicator] = -rel_modulus[indicator]

        phase_score = torch.abs(torch.sin((head_phase + rel_phase - tail_phase) / 2)).sum(dim=2) * self.phase_weight

        modulus_score = torch.norm(
            head_modulus * (rel_modulus + bias_relation) - tail_modulus * (1 - bias_relation), 
            p=1, 
            dim=2
        ) * self.modulus_weight

        return self.margin - (modulus_score + phase_score)

def forward(self, sample, batch_type=BatchType.SINGLE):
    """
    Given the indexes in `sample`, extract the corresponding embeddings,
    and call func().
    """
    if batch_type == BatchType.SINGLE:
        head = torch.index_select(
            self.entity_modulus_embedding,
            dim=0,
            index=sample[:, 0]
        ).unsqueeze(1)

        relation = torch.index_select(
            self.relation_modulus_embedding,
            dim=0,
            index=sample[:, 1]
        ).unsqueeze(1)

        tail = torch.index_select(
            self.entity_modulus_embedding,
            dim=0,
            index=sample[:, 2]
        ).unsqueeze(1)

    elif batch_type == BatchType.HEAD_BATCH:
        tail_part, head_part = sample
        batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

        head = torch.index_select(
            self.entity_modulus_embedding,
            dim=0,
            index=head_part.view(-1)
        ).view(batch_size, negative_sample_size, -1)

        relation = torch.index_select(
            self.relation_modulus_embedding,
            dim=0,
            index=tail_part[:, 1]
        ).unsqueeze(1)

        tail = torch.index_select(
            self.entity_modulus_embedding,
            dim=0,
            index=tail_part[:, 2]
        ).unsqueeze(1)

    elif batch_type == BatchType.TAIL_BATCH:
        head_part, tail_part = sample
        batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

        head = torch.index_select(
            self.entity_modulus_embedding,
            dim=0,
            index=head_part[:, 0]
        ).unsqueeze(1)

        relation = torch.index_select(
            self.relation_modulus_embedding,
            dim=0,
            index=head_part[:, 1]
        ).unsqueeze(1)

        tail = torch.index_select(
            self.entity_modulus_embedding,
            dim=0,
            index=tail_part.view(-1)
        ).view(batch_size, negative_sample_size, -1)

    else:
        raise ValueError('batch_type %s not supported!'.format(batch_type))

    return self.func(head, relation, tail, batch_type)

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        model.train()
        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_type = next(train_iterator)

        positive_sample = positive_sample.cuda()
        negative_sample = negative_sample.cuda()
        subsampling_weight = subsampling_weight.cuda()

        positive_score = model(positive_sample, batch_type=BatchType.SINGLE)
        negative_score = model((positive_sample, negative_sample), batch_type=batch_type)

        positive_loss = -F.logsigmoid(positive_score).squeeze(dim=1)
        negative_loss = F.logsigmoid(-negative_score).mean(dim=1)

        loss = (positive_loss + negative_loss).mean()
        loss.backward()
        optimizer.step()

        log = {
            "positive_loss": positive_loss.mean().item(),
            "negative_loss": negative_loss.mean().item(),
            "loss": loss.item(),
        }

        return log

    @staticmethod
    def test_step(model, data_reader, mode, args):
        model.eval()
        metrics = {}
        with torch.no_grad():
            pass
        return metrics
