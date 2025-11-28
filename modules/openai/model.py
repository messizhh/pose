# modules/openai/model.py
# Fully revised version with Cross-Modal Fusion support (ViT-focused, compatible with ModifiedResNet path)
# Key fixes:
# - positional embedding broadcasting fixed
# - class token broadcasting fixed
# - pose_dim passed through Transformer/Fusion
# - pose_feat reshape + detach + key_padding_mask handling
# - minimal invasive changes to preserve original CLIP compatibility

from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# CrossModalAdapter must accept (x_rgb, x_pose, pose_pad_mask=None)
from modules.pose.fusion import CrossModalAdapter


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        # Use an nn.MultiheadAttention for readability/compatibility
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        # x: (N, C, H, W)
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # -> (HW, N, C)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1, N, C)
        # positional embedding -> (1, L, C) then broadcasting
        pos = self.positional_embedding.to(x.dtype).unsqueeze(0)  # (1, L, C)
        x = x + pos.squeeze(0)[:, None, :]  # result shape (L, N, C) but broadcasting adapted
        # Use MultiheadAttention (batch_first=False default here since x is (L,N,C))
        out, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return out.squeeze(0)


class ModifiedResNet(nn.Module):
    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Adapter(nn.Module):
    def __init__(self, d_model: int, skip_connect=True):
        super().__init__()
        r = 1 / 4
        self.mlp = nn.Sequential(OrderedDict([
            ("c_in", nn.Linear(d_model, int(d_model * r))),
            ("gelu", QuickGELU()),
            ("c_out", nn.Linear(int(d_model * r), d_model))
        ]))
        self.skip_connect = skip_connect
        # zero-init last layer to start as identity
        nn.init.zeros_(self.mlp[-1].weight)
        if self.mlp[-1].bias is not None:
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor):
        if self.skip_connect:
            return self.mlp(x) + x
        else:
            return self.mlp(x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, checkpointing=False):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=False)  # original uses (L,N,D)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        # adapters (zero init inside)
        self.S_Adapter = Adapter(d_model, skip_connect=False)
        self.MLP_Adapter = Adapter(d_model, skip_connect=False)

        scale = d_model ** -0.5
        self.prefix_length = 8
        self.prefix_embedding_k = nn.Parameter(scale * torch.randn(self.prefix_length, 1, d_model))
        self.prefix_embedding_v = nn.Parameter(scale * torch.randn(self.prefix_length, 1, d_model))
        self.checkpointing = checkpointing

    def attention(self, x: torch.Tensor):
        # x shape: (L, N, D)
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        # concatenate prefix embeddings to key/value (dim 0)
        k = torch.cat([x, self.prefix_embedding_k.repeat(1, x.shape[1], 1)], dim=0)
        v = torch.cat([x, self.prefix_embedding_v.repeat(1, x.shape[1], 1)], dim=0)
        return self.attn(x, k, v, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        if self.checkpointing:
            x = x + checkpoint(self.attention, checkpoint(self.ln_1, x)) + self.S_Adapter(x)
        else:
            x = x + self.attention(self.ln_1(x)) + self.S_Adapter(x)

        if self.checkpointing:
            x = x + checkpoint(self.mlp, checkpoint(self.ln_2, x)) + self.MLP_Adapter(x)
        else:
            x = x + self.mlp(self.ln_2(x)) + self.MLP_Adapter(x)
        return x


class AggregationBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=False)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 1)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 1, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        # cls attends to other tokens: x shape (L,N,D) -> take 0 as cls + rest
        return self.attn(x[:1], x[1:], x[1:], need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, cls):
        cls = cls + self.attention(self.ln_1(torch.cat((cls, x), dim=0)))
        cls = cls + self.mlp(self.ln_2(cls))
        return cls


class UnfoldTemporalWindows(nn.Module):
    def __init__(self, window_size=5, window_stride=1, window_dilation=1):
        super().__init__()
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_dilation = window_dilation

        self.padding = (window_size + (window_size - 1) * (window_dilation - 1) - 1) // 2
        self.unfold = nn.Unfold(kernel_size=(self.window_size, 1),
                                dilation=(self.window_dilation, 1),
                                stride=(self.window_stride, 1),
                                padding=(self.padding, 0))

    def forward(self, x, T):
        NT, C, P = x.shape
        x = x.view(-1, T, C, P).permute(0, 2, 1, 3)
        x = self.unfold(x)
        x = x.view(-1, C, self.window_size, T, P).permute(0, 3, 1, 2, 4).reshape(NT, C, -1)
        return x


class Correlation_Module(nn.Module):
    def __init__(self, neighbors=5):
        super().__init__()

    def forward(self, x, upfold):
        L, N, D = x.shape
        import math
        affinities = torch.einsum('lnd,ond->lon', x, upfold) / math.sqrt(D)
        features = torch.einsum('lon,ond->lnd', torch.sigmoid(affinities) - 0.5, upfold)
        return features


class TemporalAggregationBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = Correlation_Module()
        self.ln_1 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.upfold = UnfoldTemporalWindows(5)

    def attention(self, x: torch.Tensor, T):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        x_upfold = self.upfold(x[1:].permute(1, 2, 0), T).permute(2, 0, 1)
        return self.attn(x[:1], x_upfold)

    def forward(self, x: torch.Tensor, T):
        cls = self.attention(self.ln_1(x), T)
        return cls


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, checkpointing=True,
                 pose_dim: int = None):
        super().__init__()
        self.width = width
        self.checkpointing = checkpointing
        self.layers = layers
        if self.checkpointing:
            self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask, checkpointing=(i >= 0)) for i in range(layers)])
        else:
            self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for i in range(layers)])

        # Fusion blocks: pass pose_dim so CrossModalAdapter can project pose -> rgb dim
        self.pose_dim = pose_dim if pose_dim is not None else width
        self.fusion_blocks = nn.ModuleList([
            CrossModalAdapter(dim_rgb=width, dim_pose=self.pose_dim, n_heads=heads)
            for _ in range(layers)
        ])

        self.query = nn.Parameter(torch.rand(1, 1, width), requires_grad=True)
        self.aggblocks = nn.ModuleList([AggregationBlock(width, heads, attn_mask) for _ in range(layers)])
        self.taggblocks = TemporalAggregationBlock(width, heads, attn_mask)
        self.temporal_ada_weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x: torch.Tensor, T=None, pose_feat=None, pose_mask=None):
        """
        x: (L, N, D)  # L = seq_len (patches+1), N = batch_first in original CLIP = batch*frames
        pose_feat: either (B, T, V, Dp) or (N, V, Dp) where N == batch*T
        pose_mask: either (B, T, V) or (N, V) with 1=valid,0=pad
        """
        L, N, D = x.shape
        query = self.query.repeat(1, N, 1)  # (1, N, D)

        pose_feat_flat = None
        key_pad = None

        if pose_feat is not None:
            # Detach to prevent gradients back to pose encoder (Dual-Frozen strategy)
            pose_feat = pose_feat.detach()

            # If pose is (B, T, V, Dp) -> flatten to (B*T, V, Dp)
            if pose_feat.dim() == 4:
                B, Time, V, Dp = pose_feat.shape
                assert B * Time == N, f"Mismatch batch/frame: Visual N={N}, Pose B*T={B*Time}"
                pose_feat_flat = pose_feat.reshape(B * Time, V, Dp)
            else:
                pose_feat_flat = pose_feat  # assume already (N, V, Dp)

            # handle mask: convert to key_padding_mask format (True==PAD)
            if pose_mask is not None:
                if pose_mask.dim() == 3:
                    B, Time, V = pose_mask.shape
                    assert B * Time == N, f"Mask size mismatch vs visual N: mask {pose_mask.shape} vs N={N}"
                    key_pad = pose_mask.reshape(B * Time, V)
                else:
                    key_pad = pose_mask
                # current mask semantics: 1=valid, 0=pad -> MHA expects True for pad positions
                key_pad = (key_pad == 0).to(dtype=torch.bool)

        # iterate layers
        for i in range(len(self.resblocks)):
            # original CLIP block (ResidualAttentionBlock expects/returns (L,N,D))
            x = self.resblocks[i](x)

            # cross-modal fusion inserted after attention+residual (as you designed)
            if pose_feat_flat is not None:
                x_nld = x.permute(1, 0, 2)  # (N, L, D)
                # Fusion adapter expects (N, L, D) and (N, V, Dp)
                x_nld = self.fusion_blocks[i](x_rgb=x_nld, x_pose=pose_feat_flat, pose_pad_mask=key_pad)
                # back to (L, N, D)
                x = x_nld.permute(1, 0, 2)

            # aggregation block update for cls
            if self.checkpointing:
                query = checkpoint(self.aggblocks[i], x, query)
            else:
                query = self.aggblocks[i](x, query)

        # temporal aggregation
        if self.checkpointing:
            x = x + checkpoint(self.taggblocks, x, T) * self.temporal_ada_weight
        else:
            x = x + self.taggblocks(x, T) * self.temporal_ada_weight

        return x, query


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int,
                 output_dim: int, pose_dim: int = None):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        # class token as (1, D)
        self.class_embedding = nn.Parameter(scale * torch.randn(1, width))
        # positional embedding shape (L, D) -> we'll unsqueeze in forward to (1, L, D)
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        # Pass pose_dim to Transformer so fusion adapters know pose dimension
        self.transformer = Transformer(width, layers, heads, pose_dim=pose_dim)

        self.ln_post = LayerNorm(width)
        self.ln_post_cls = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.ada_weight = nn.Parameter(torch.tensor([0.5, 0.5]), requires_grad=True)

    def forward(self, x: torch.Tensor, T, pose_feat=None, pose_mask=None):
        # x: (Batch*Time, 3, H, W)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # Broadcast class token: (1, D) -> (B, 1, D)
        cls_token = self.class_embedding.to(x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        # positional embedding: make (1, L, D) for safe broadcasting
        pos = self.positional_embedding.to(x.dtype).unsqueeze(0)  # (1, L, D)
        x = x + pos  # (B, L, D)
        x = self.ln_pre(x)

        # transpose to CLIP internal format LND (L, N, D)
        x = x.permute(1, 0, 2)

        # pass pose down to transformer; pose_feat could be None
        x, new_cls = self.transformer(x, T, pose_feat=pose_feat, pose_mask=pose_mask)

        # back to NLD
        x = x.permute(1, 0, 2)

        # combine cls and pooled image token as in original design
        cls_out = self.ln_post(x[:, 0, :]) * self.ada_weight[0]
        cls_from_query = self.ln_post_cls(new_cls.permute(1, 0, 2))[:, 0, :] * self.ada_weight[1]
        x = cls_out + cls_from_query

        if self.proj is not None:
            x = x @ self.proj
        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 pose_dim: int = None,  # IMPORTANT: pass the pose encoder output dim here
                 ):
        super().__init__()

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                pose_dim=pose_dim
            )

        self.initialize_parameters()

    def initialize_parameters(self):
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, T, pose_feat=None, pose_mask=None):
        if pose_feat is None:
            # keep compatibility
            return self.visual(image.type(self.dtype), T)
        return self.visual(image.type(self.dtype), T, pose_feat=pose_feat, pose_mask=pose_mask)

    def forward(self, image):
        image_features = self.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, pose_dim: int = 256):
    """
    Build CLIP visual model from a state_dict.
    pose_dim: the output dim of your pretrained PoseEncoder (default 256) — adjust to your checkpoint
    """
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        pose_dim=pose_dim
    )

    # keep only visual params for loading visual weights
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    for key in list(state_dict.keys()):
        if not 'visual' in key:
            del state_dict[key]

    model.load_state_dict(state_dict, strict=False)
    return model
