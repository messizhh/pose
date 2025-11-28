# modules/pose/st_gcn.py
import copy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Graph:
    """
    Graph helper to construct adjacency A for a skeleton.
    Default: MediaPipe body topology with 33 nodes (partial, as example).
    You can pass custom num_node and edge list for hands/face.
    Produces self.A: torch.Tensor shape (K, V, V) where K is number of subsets (here 2: hop0 & hop1).
    """
    def __init__(self, num_node: int = 33, edge_list: Optional[List[Tuple[int,int]]] = None, strategy: str = "spatial"):
        self.num_node = int(num_node)
        self.strategy = strategy
        self.edge = None
        self._build_default_edges(edge_list)
        self._get_adjacency()

    def _build_default_edges(self, edge_list: Optional[List[Tuple[int,int]]]):
        if edge_list is not None:
            self.edge = [(i, i) for i in range(self.num_node)] + list(edge_list)
            return

        # default MediaPipe-like partial topology for 33 points (you can replace with precise mapping)
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_link = [
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10), (11, 12),
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
            (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
        ]
        self.edge = self_link + neighbor_link

    def _get_adjacency(self):
        # build hop distance matrix
        V = self.num_node
        hop_dis = np.full((V, V), np.inf)
        for i in range(V):
            hop_dis[i, i] = 0
        for i, j in self.edge:
            if i < V and j < V:
                hop_dis[i, j] = hop_dis[j, i] = 1

        valid_hop = 1
        if self.strategy == "spatial":
            A_list = []
            for hop in range(valid_hop + 1):
                a_root = np.zeros((V, V), dtype=np.float32)
                a_close = np.zeros((V, V), dtype=np.float32)
                for i in range(V):
                    for j in range(V):
                        if hop_dis[j, i] == hop:
                            if hop == 0:
                                a_root[j, i] = 1.0
                            else:
                                a_close[j, i] = 1.0
                A_list.append(a_root + a_close)
            A = np.stack(A_list, axis=0)  # (K, V, V)
            self.A = torch.tensor(A, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown graph strategy: {self.strategy}")


class STGCNBlock(nn.Module):
    """
    Single ST-GCN block:
      - spatial: 1x1 conv producing out_channels * K, split per subset and multiply with adjacency
      - temporal: Conv2d over (T,1) to capture temporal context
    Input: x (N, in_channels, T, V)
    Output: (N, out_channels, T, V_out) where V_out == V unless adjacency changes
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 t_kernel_size: int = 9,
                 stride: int = 1,
                 dropout: float = 0.0,
                 A: torch.Tensor = None,
                 learnable_A: bool = False):
        super().__init__()
        assert A is not None, "Adjacency A (K,V,V) must be provided"
        # register adjacency as buffer or parameter
        if learnable_A:
            self.A = nn.Parameter(A.clone(), requires_grad=True)
        else:
            # register as buffer so it moves with .to(device) and is saved in state_dict
            self.register_buffer('A', A.clone())

        self.num_subsets = A.shape[0]
        self.out_channels = out_channels

        # spatial conv: 1x1 to produce channels = out_channels * K
        self.gcn = nn.Conv2d(in_channels, out_channels * self.num_subsets, kernel_size=1, bias=False)

        # temporal conv (TCN)
        padding_t = (t_kernel_size - 1) // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(t_kernel_size, 1),
                      stride=(stride, 1), padding=(padding_t, 0), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

        # residual
        if (in_channels != out_channels) or (stride != 1):
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C, T, V)
        """
        res = self.residual(x)  # (N, out_c, T, V) or (N, C, T, V) if identity
        N, C, T, V = x.shape

        # spatial conv
        x = self.gcn(x)  # (N, out_c * K, T, V)
        # reshape to (N, K, out_c, T, V)
        x = x.view(N, self.num_subsets, self.out_channels, T, V)
        # einsum over adjacency: (N, K, out_c, T, V) * (K, V, W) -> (N, out_c, T, W)
        # here W == V (no change in node count)
        x = torch.einsum('nkctv,kvw->nctw', x, self.A)

        # temporal conv and return with residual
        x = self.tcn(x)  # (N, out_c, T, V)
        return self.relu(x + res)


class STGCN(nn.Module):
    """
    Stack multiple STGCNBlock to form an encoder.
    Example usage:
      graph = Graph(num_node=33)
      A = graph.A  # (K, V, V)
      model = STGCN(in_channels=6, channels=[64, 128, 256], A=A, t_kernel=9, dropout=0.1)
    Input to forward: (N, C, T, V)
    Output: (N, C_out, T, V)
    """
    def __init__(self,
                 in_channels: int,
                 channels: List[int],
                 A: torch.Tensor,
                 t_kernel: int = 9,
                 dropout: float = 0.0,
                 residual: bool = True):
        super().__init__()
        self.layers = nn.ModuleList()
        C_in = in_channels
        for i, C_out in enumerate(channels):
            stride = 1
            block = STGCNBlock(in_channels=C_in,
                               out_channels=C_out,
                               t_kernel_size=t_kernel,
                               stride=stride,
                               dropout=dropout,
                               A=A,
                               learnable_A=False)
            self.layers.append(block)
            C_in = C_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V)
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------- Example / test helper ----------------
if __name__ == "__main__":
    # quick sanity test
    V = 33
    graph = Graph(num_node=V)
    A = graph.A  # (K, V, V)
    net = STGCN(in_channels=6, channels=[64, 128, 256], A=A, t_kernel=9, dropout=0.1)
    # dummy input: N=2, C=6 (coords + motion), T=50, V=33
    x = torch.randn(2, 6, 50, V)
    y = net(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)  # expect (2, 256, 50, 33)
