import torch
import torch.nn as nn

class CrossModalAdapter(nn.Module):
    """
    Cross-Modal Adapter with Vector Gating.
    Integrates Pose features into Visual features via Cross-Attention.
    """
    def __init__(self, dim_rgb: int, dim_pose: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        # If pose dim != rgb dim, project
        self.pose_proj = nn.Linear(dim_pose, dim_rgb) if dim_pose != dim_rgb else nn.Identity()

        # Pre-Norm improves stability
        self.norm_rgb  = nn.LayerNorm(dim_rgb)
        self.norm_pose = nn.LayerNorm(dim_rgb)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim_rgb,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Vector gates (zero init)
        self.gate_attn = nn.Parameter(torch.zeros(dim_rgb))
        self.gate_ffn  = nn.Parameter(torch.zeros(dim_rgb))

        hidden_dim = dim_rgb // 4
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim_rgb),
            nn.Linear(dim_rgb, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim_rgb),
            nn.Dropout(dropout)
        )

    def forward(self, x_rgb, x_pose, pose_pad_mask=None):
        """
        x_rgb:  (BT, L, Drgb)
        x_pose: (BT, V, Dp)
        pose_pad_mask: (BT, V)   # True = padding
        """
        B, L, Drgb = x_rgb.shape
        B2, V, Dp = x_pose.shape
        assert B == B2, "Batch mismatch between RGB and Pose streams"

        # Project pose → rgb dimension
        pose = self.pose_proj(x_pose)

        q = self.norm_rgb(x_rgb)
        k = v = self.norm_pose(pose)

        attn_out, _ = self.cross_attn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=pose_pad_mask   # True = IGNORE those positions
        )

        # Gate must broadcast over sequence
        x = x_rgb + attn_out * self.gate_attn

        x = x + self.ffn(x) * self.gate_ffn

        return x
