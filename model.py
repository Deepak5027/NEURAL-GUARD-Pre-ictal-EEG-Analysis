"""
Stage 3 — Temporal Graph Neural Network (STGCN-inspired)
Architecture:
  1. Temporal Conv1D  — learns local patterns along time axis per node
  2. GCN layers       — propagates spatial (inter-channel) information
  3. Temporal Conv1D  — refines after spatial mixing
  4. Global pooling   — collapses graph to fixed vector
  5. MLP classifier   — pre-ictal vs interictal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch


class TemporalConvBlock(nn.Module):
    """1-D depthwise temporal convolution applied per node feature."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 9):
        super().__init__()
        pad = (kernel - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=pad, bias=False)
        self.bn   = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B*N, C, T)
        return F.gelu(self.bn(self.conv(x)))


class SpatioTemporalBlock(nn.Module):
    """One ST block: temporal → spatial (GCN) → temporal."""
    def __init__(self, in_feat: int, hidden: int, out_feat: int, kernel: int = 9):
        super().__init__()
        self.temp1 = TemporalConvBlock(in_feat, hidden, kernel)
        self.gcn   = GCNConv(hidden, hidden)
        self.temp2 = TemporalConvBlock(hidden, out_feat, kernel)
        self.skip  = nn.Linear(in_feat, out_feat) if in_feat != out_feat else nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor, batch_size: int, n_nodes: int) -> torch.Tensor:
        """
        x           : (B*N, C, T)   node signal features
        edge_index  : (2, E)
        edge_weight : (E,)
        """
        B_N, C, T = x.shape

        # ── temporal pass ──
        h = self.temp1(x)                                  # (B*N, hidden, T)

        # ── spatial pass: average over time → GCN → broadcast back ──
        h_mean = h.mean(dim=-1)                            # (B*N, hidden)
        h_mean = F.gelu(self.gcn(h_mean, edge_index, edge_weight))  # (B*N, hidden)
        h = h + h_mean.unsqueeze(-1)                       # broadcast (B*N, hidden, T)

        # ── second temporal pass ──
        h = self.temp2(h)                                  # (B*N, out_feat, T)

        # ── skip connection ──
        skip = self.skip(x.mean(-1)).unsqueeze(-1).expand_as(h)
        return h + skip


class SeizureTGNN(nn.Module):
    """
    Full Temporal GNN for pre-seizure binary classification.

    Input per graph:
      x           : (N_nodes, T)          raw EEG signal as node features
      edge_index  : (2, E)
      edge_attr   : (E,)                  PLV weights
      batch       : (N_nodes,)

    Output:
      logits      : (B, 2)                pre-ictal vs interictal
    """
    def __init__(self,
                 n_channels: int  = 23,
                 n_samples:  int  = 7680,
                 hidden:     int  = 64,
                 n_blocks:   int  = 3,
                 dropout:    float = 0.3):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples  = n_samples

        # Project raw signal → feature channels
        self.input_proj = TemporalConvBlock(1, hidden, kernel=9)

        # Stack of ST blocks
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.blocks.append(
                SpatioTemporalBlock(hidden, hidden, hidden, kernel=9)
            )

        # Classifier
        self.pool    = global_mean_pool                     # (B*N, D) → (B, D)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x           = data.x            # (B*N, T)
        edge_index  = data.edge_index   # (2, E)
        edge_weight = data.edge_attr    # (E,)
        batch       = data.batch        # (B*N,)

        B_N, T = x.shape

        # (B*N, T) → (B*N, 1, T) → (B*N, hidden, T)
        h = self.input_proj(x.unsqueeze(1))

        for block in self.blocks:
            h = block(h, edge_index, edge_weight,
                      batch_size=int(batch.max().item()) + 1,
                      n_nodes=self.n_channels)

        # Average over time → (B*N, hidden)
        h = h.mean(dim=-1)
        h = self.dropout(h)

        # Graph-level pooling → (B, hidden)
        h = self.pool(h, batch)

        return self.head(h)             # (B, 2)


if __name__ == "__main__":
    from torch_geometric.data import Batch
    N, T = 23, 7680
    graphs = []
    for _ in range(4):
        x  = torch.randn(N, T)
        ei = torch.randint(0, N, (2, 60))
        ea = torch.rand(60)
        graphs.append(Data(x=x, edge_index=ei, edge_attr=ea, y=torch.tensor([0])))

    batch = Batch.from_data_list(graphs)
    model = SeizureTGNN(n_channels=N, n_samples=T)
    logits = model(batch)
    print(f"Logits shape : {logits.shape}")   # (4, 2)
    print(f"Parameters   : {sum(p.numel() for p in model.parameters()):,}")
