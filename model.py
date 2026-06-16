"""
Model definitions for MTDriverGNN.

Contains:
- ResidualGCNEncoder: GCN encoder with residual connection
- GCN_Residual_TwoHeads: multi-task model with two prediction heads (Y1, Y2)
- LearnableAlpha: learnable weighting factor between the two task losses
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class ResidualGCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int],
                 dropout: float = 0.5, use_layernorm: bool = False):
        super().__init__()
        assert len(hidden_dims) >= 1
        self.convs = nn.ModuleList()
        last = in_dim
        for h in hidden_dims:
            self.convs.append(GCNConv(last, h))
            last = h
        self.res_proj = nn.Linear(in_dim, hidden_dims[-1], bias=False) if in_dim != hidden_dims[-1] else nn.Identity()
        self.ln = nn.LayerNorm(hidden_dims[-1]) if use_layernorm else nn.Identity()
        self.dropout = dropout

        for conv in self.convs:
            nn.init.xavier_uniform_(conv.lin.weight)
        if isinstance(self.res_proj, nn.Linear):
            nn.init.xavier_uniform_(self.res_proj.weight)

    def forward(self, x, edge_index):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + (self.res_proj(x) if isinstance(self.res_proj, nn.Linear) else x)
        h = self.ln(h)
        return h


class GCN_Residual_TwoHeads(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int],
                 dropout: float = 0.5, use_layernorm: bool = False,
                 head_hidden: int = None):
        super().__init__()
        self.encoder = ResidualGCNEncoder(in_dim, hidden_dims,
                                          dropout=dropout,
                                          use_layernorm=use_layernorm)
        hd = hidden_dims[-1]
        if head_hidden is None:
            head_hidden = hd
        self.shared = nn.Sequential(
            nn.Linear(hd, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out_y1 = nn.Linear(head_hidden, 1)
        self.out_y2 = nn.Linear(head_hidden, 1)

        for m in self.shared:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.out_y1.weight, nonlinearity="sigmoid")
        nn.init.kaiming_uniform_(self.out_y2.weight, nonlinearity="sigmoid")

    def forward(self, x, edge_index, return_h: bool = False):
        h = self.encoder(x, edge_index)
        h = self.shared(h)
        logit1 = self.out_y1(h).squeeze(-1)
        logit2 = self.out_y2(h).squeeze(-1)
        if return_h:
            return logit1, logit2, h
        return logit1, logit2


class LearnableAlpha(nn.Module):
    def __init__(self, init_alpha: float = 0.7):
        super().__init__()
        init_logit = torch.logit(torch.tensor(float(init_alpha)))
        self.logit_alpha = nn.Parameter(init_logit)

    def forward(self):
        return torch.sigmoid(self.logit_alpha)
