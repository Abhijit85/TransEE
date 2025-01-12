import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import math



class TransEEnhanced(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, margin, distance_metric=2, # 1 for L1 and 2 for L2
                 gamma=10.0, phase_weight=1.0, modulus_weight=4.0, epsilon=1.0,reg_coeff=1e-4):
        super(TransEEnhanced, self).__init__()

        # Basic TransE embeddings
        self.entity_modulus = nn.Embedding(num_entities, embedding_dim)
        self.entity_phase = nn.Embedding(num_entities, embedding_dim)
        self.relation_modulus = nn.Embedding(num_relations, embedding_dim)
        self.relation_phase = nn.Embedding(num_relations, embedding_dim)

        # Margin and distance settings
        self.margin = margin
        self.distance_metric = distance_metric


        # Hyperbolic and scaling settings
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + epsilon) / embedding_dim]), requires_grad=False
        )

        # Weights for phase and modulus
        self.phase_weight = phase_weight
        self.modulus_weight = modulus_weight
        self.reg_coeff = reg_coeff

        # Initialization
        nn.init.uniform_(self.entity_modulus.weight, a=-self.embedding_range.item(), b=self.embedding_range.item())
        nn.init.uniform_(self.entity_phase.weight, a=-np.pi, b=np.pi)
        nn.init.uniform_(self.relation_modulus.weight, a=-self.embedding_range.item(), b=self.embedding_range.item())
        nn.init.uniform_(self.relation_phase.weight, a=-np.pi, b=np.pi)

    def forward(self, head, relation, tail):
        h_mod = self.entity_modulus(head)
        h_phase = self.entity_phase(head)
        r_mod = self.relation_modulus(relation)
        r_phase = self.relation_phase(relation)
        t_mod = self.entity_modulus(tail)
        t_phase = self.entity_phase(tail)

        # Modulus scoring: hyperbolic-inspired adjustment
        modulus_score = torch.norm(h_mod * r_mod - t_mod, p=self.distance_metric, dim=-1)

        # Phase scoring: advanced angular consistency
        phase_diff = torch.abs(torch.sin((h_phase + r_phase - t_phase) / 2))
        phase_score = torch.sum(phase_diff, dim=-1)

        # Weighted combined score
        score = self.modulus_weight * modulus_score + self.phase_weight * phase_score
        return score

    def compute_loss(self, positive_score, negative_score):
        # Margin-based ranking loss
        base_loss = F.relu(self.margin + positive_score - negative_score)

        # Regularization terms for Elastic modulus and phase (combining L1 and L2)
        modulus_regularization_L1 = torch.sum(torch.norm(self.entity_modulus.weight, p=1, dim=-1))
        modulus_regularization_L2 = torch.sum(torch.norm(self.entity_modulus.weight, p=2, dim=-1))
        phase_regularization_L1 = torch.sum(torch.norm(self.entity_phase.weight, p=1, dim=-1))
        phase_regularization_L2 = torch.sum(torch.norm(self.entity_phase.weight, p=2, dim=-1))

        # Total loss with combined regularization
        # Introduce alpha to control the mix between L1 and L2
        alpha = 0.5  # Example value, you can adjust this
        total_loss = base_loss.mean() + reg_coeff * (
            alpha * (modulus_regularization_L1 + phase_regularization_L1) +
            (1 - alpha) * (modulus_regularization_L2 + phase_regularization_L2)
    )


        # # Regularization terms for modulus and phase
        # modulus_regularization = torch.sum(torch.norm(self.entity_modulus.weight, p=self.distance_metric, dim=-1))
        # phase_regularization = torch.sum(torch.norm(self.entity_phase.weight, p=self.distance_metric, dim=-1))

        # # Total loss with regularization
        # # total_loss = base_loss.mean() + 1e-5 * (modulus_regularization + phase_regularization)
        # # total_loss = base_loss.mean() + 1e-3 * (modulus_regularization + phase_regularization)
        # total_loss = base_loss.mean() + 1e-4 * (modulus_regularization + phase_regularization)
        # # total_loss = base_loss.mean() + 1e-2 * (modulus_regularization + phase_regularization)
        return total_loss

# Helper function to create corrupted edges
def create_corrupted_edge_index(edge_index, edge_type, num_entities,negative_rate =50):
      """
    Creates corrupted edge indices for negative sampling.

    Args:
        edge_index (torch.Tensor): The original edge indices.
        edge_type (torch.Tensor): The edge types.
        num_entities (int): The total number of entities.
        negative_rate (int, optional): The number of negative samples per positive sample. Defaults to 1.

    Returns:
        torch.Tensor: The corrupted edge indices.
    """

    # Repeat the original edge indices based on the negative rate
      num_positive_edges = edge_index.shape[1]
      repeated_edge_index = edge_index.repeat(1, negative_rate)
      repeated_edge_type = edge_type.repeat(negative_rate)

    # Create corrupted edges
      corrupt_head_or_tail = torch.randint(high=2, size=(num_positive_edges * negative_rate,),
                                         device=edge_index.device)
      random_entities = torch.randint(high=num_entities,
                                     size=(num_positive_edges * negative_rate,), device=edge_index.device)

    # Corrupt either head or tail based on corrupt_head_or_tail
      heads = torch.where(corrupt_head_or_tail == 1, random_entities,
                          repeated_edge_index[0, :])
      tails = torch.where(corrupt_head_or_tail == 0, random_entities,
                          repeated_edge_index[1, :])

      return torch.stack([heads, tails], dim=0)