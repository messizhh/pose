# modules/pose/pose_encoder.py

import torch
import torch.nn as nn
from typing import List, Tuple

from .st_gcn import STGCN, Graph


# -------------------------------------------------------
# Helper: Build Combined Skeleton Graph
# -------------------------------------------------------
def get_combined_graph_params(parts: List[str]) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Construct a unified skeleton graph for ['body','left','right'].
    face 不建议加，因为太大（468 nodes）。
    """
    PART_SIZES = {
        'body': 33,
        'left': 21,
        'right': 21
    }

    def get_body_edges():
        return [
            (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
            (9,10),(11,12),
            (11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
            (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
            (11,23),(12,24),(23,24),
            (23,25),(25,27),(27,29),(27,31),(29,31),
            (24,26),(26,28),(28,30),(28,32),(30,32)
        ]

    def get_hand_edges():
        edges = []
        edges += [(0,1),(1,2),(2,3),(3,4)]
        edges += [(0,5),(5,6),(6,7),(7,8)]
        edges += [(0,9),(9,10),(10,11),(11,12)]
        edges += [(0,13),(13,14),(14,15),(15,16)]
        edges += [(0,17),(17,18),(18,19),(19,20)]
        return edges

    all_edges = []
    global_indices = {}
    offset = 0

    for part in parts:
        size = PART_SIZES[part]

        # internal edges
        if part == "body":
            edges = get_body_edges()
            global_indices["l_wrist"] = offset + 15
            global_indices["r_wrist"] = offset + 16

        elif part == "left":
            edges = get_hand_edges()
            global_indices["left_root"] = offset + 0

        elif part == "right":
            edges = get_hand_edges()
            global_indices["right_root"] = offset + 0

        # shift edges
        for u, v in edges:
            all_edges.append((u+offset, v+offset))
            all_edges.append((v+offset, u+offset))  # make undirected

        offset += size

    total_nodes = offset

    # cross-part stitching
    if "body" in parts and "left" in parts:
        all_edges.append((global_indices["l_wrist"], global_indices["left_root"]))
        all_edges.append((global_indices["left_root"], global_indices["l_wrist"]))

    if "body" in parts and "right" in parts:
        all_edges.append((global_indices["r_wrist"], global_indices["right_root"]))
        all_edges.append((global_indices["right_root"], global_indices["r_wrist"]))

    # add self links
    all_edges += [(i, i) for i in range(total_nodes)]

    return total_nodes, all_edges


# -------------------------------------------------------
# Pose Encoder
# -------------------------------------------------------
class PoseEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: List[int] = [64,128,256],
        parts: List[str] = ['body','left','right'],
        t_kernel: int = 9,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.parts = parts

        # 1. build graph
        self.num_nodes, edge_list = get_combined_graph_params(parts)
        self.graph = Graph(num_node=self.num_nodes, edge_list=edge_list)

        # 2. input normalization: BN over (C*V)
        self.data_bn = nn.BatchNorm1d(in_channels * self.num_nodes)

        # 3. ST-GCN backbone
        self.stgcn = STGCN(
            in_channels=in_channels,
            channels=hidden_channels,
            A=self.graph.A,
            t_kernel=t_kernel,
            dropout=dropout
        )

        self.out_dim = hidden_channels[-1]

    def forward(self, x):
        """
        x: (N, C, T, V)
        """
        N, C, T, V = x.shape
        if V != self.num_nodes:
            raise ValueError(f"PoseEncoder expected V={self.num_nodes} nodes, but got {V} (parts={self.parts})")

        # BN: (N, C*T*V) → reshape-friendly
        x = x.reshape(N, C * V, T)
        x = self.data_bn(x)
        x = x.reshape(N, C, T, V)

        feat = self.stgcn(x)   # (N, C_out, T, V)
        return feat


# -------------------------------------------------------
# Pretrain Wrapper
# -------------------------------------------------------
class PretrainModel(nn.Module):
    def __init__(self, encoder: PoseEncoder, in_channels: int = 6):
        super().__init__()
        self.encoder = encoder
        dim = encoder.out_dim

        self.recon_head = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, in_channels, 1)
        )

        self.pred_head = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, in_channels, 1)
        )

    def forward(self, x):
        feat = self.encoder(x)
        return self.recon_head(feat), self.pred_head(feat)
