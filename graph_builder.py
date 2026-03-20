"""
Stage 2 — Dynamic Brain Graph Construction
Treats each EEG channel as a node. Edge weights = Phase Locking Value (PLV)
between channel pairs, computed per epoch. Threshold to create sparse graph.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from itertools import combinations


N_CHANNELS  = 23
PLV_THRESH  = 0.3          # keep edges with PLV >= threshold


# ── Standard CHB-MIT 3-D electrode positions (simplified unit sphere coords) ──
ELECTRODE_POS = {
    "FP1":  (-0.3,  0.9,  0.3),  "FP2":  ( 0.3,  0.9,  0.3),
    "F7":   (-0.7,  0.5,  0.5),  "F3":   (-0.5,  0.6,  0.6),
    "FZ":   ( 0.0,  0.7,  0.7),  "F4":   ( 0.5,  0.6,  0.6),
    "F8":   ( 0.7,  0.5,  0.5),  "T7":   (-1.0,  0.0,  0.0),
    "C3":   (-0.7,  0.0,  0.7),  "CZ":   ( 0.0,  0.0,  1.0),
    "C4":   ( 0.7,  0.0,  0.7),  "T8":   ( 1.0,  0.0,  0.0),
    "P7":   (-0.7, -0.5,  0.5),  "P3":   (-0.5, -0.6,  0.6),
    "PZ":   ( 0.0, -0.7,  0.7),  "P4":   ( 0.5, -0.6,  0.6),
    "P8":   ( 0.7, -0.5,  0.5),  "O1":   (-0.3, -0.9,  0.3),
    "O2":   ( 0.3, -0.9,  0.3),  "F1":   (-0.25, 0.65, 0.65),
    "F2":   ( 0.25, 0.65, 0.65), "FC1":  (-0.35, 0.35, 0.87),
    "FC2":  ( 0.35, 0.35, 0.87),
}
CHANNEL_NAMES = list(ELECTRODE_POS.keys())[:N_CHANNELS]


def phase_locking_value(x: np.ndarray, y: np.ndarray) -> float:
    """PLV between two time series x and y (1-D)."""
    hx    = np.angle(np.fft.fft(x))
    hy    = np.angle(np.fft.fft(y))
    return float(np.abs(np.mean(np.exp(1j * (hx - hy)))))


def build_adjacency(epoch: np.ndarray, threshold: float = PLV_THRESH) -> np.ndarray:
    """
    epoch : (N_channels, N_samples)
    Returns: adjacency matrix (N, N) with PLV weights, zeros below threshold.
    """
    N   = epoch.shape[0]
    adj = np.zeros((N, N), dtype=np.float32)
    for i, j in combinations(range(N), 2):
        plv = phase_locking_value(epoch[i], epoch[j])
        if plv >= threshold:
            adj[i, j] = plv
            adj[j, i] = plv
    return adj


def adj_to_edge_index(adj: np.ndarray):
    """Convert dense adjacency to COO edge_index + edge_attr tensors."""
    src, dst = np.nonzero(adj)
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    edge_attr  = torch.tensor(adj[src, dst], dtype=torch.float)
    return edge_index, edge_attr


def epoch_to_graph(epoch: np.ndarray, label: int = 0) -> Data:
    """
    epoch  : (N_channels, N_samples)  preprocessed
    label  : 0 = interictal, 1 = pre-ictal
    Returns: torch_geometric.data.Data
    """
    adj        = build_adjacency(epoch)
    edge_index, edge_attr = adj_to_edge_index(adj)
    x          = torch.tensor(epoch, dtype=torch.float)  # node features = raw signal
    y          = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def build_graph_dataset(epochs: np.ndarray, labels: np.ndarray) -> list:
    """
    epochs  : (N_epochs, N_channels, N_samples)
    labels  : (N_epochs,)
    Returns : list of Data objects
    """
    return [epoch_to_graph(epochs[i], int(labels[i])) for i in range(len(epochs))]


if __name__ == "__main__":
    dummy_epoch  = np.random.randn(N_CHANNELS, 7680).astype(np.float32)
    graph        = epoch_to_graph(dummy_epoch, label=0)
    print(f"Nodes        : {graph.num_nodes}")
    print(f"Edges        : {graph.num_edges}")
    print(f"Node feature : {graph.x.shape}")
    print(f"Label        : {graph.y}")
