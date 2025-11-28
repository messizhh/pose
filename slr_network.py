# slr_network.py
# Full replacement: SLR model that fuses CLIP visual stream with a frozen PoseEncoder.
# Expects:
#   - clip_model: an instance compatible with your modified modules/openai/model.py
#                 in particular, clip.encode_image(image_batch, T, pose_feat=..., pose_mask=...)
#   - pose_encoder_checkpoint: path to pretrained PoseEncoder (optional)
#   - dataloader must return: padded_video, video_length, padded_label, label_length, info, padded_pose, pose_mask
#     where padded_pose shape is (B, C_pose, T, V) and pose_mask is (B, T, V) or (B, V) or None
#
# The forward returns:
#   - logits: (B, T, vocab_size) (time-major padded to max T in batch)
#   - enc_out: encoded features (for debugging / contrastive use)

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# adjust imports to your repo layout
from modules.pose.pose_encoder import PoseEncoder
from modules.pose.fusion import CrossModalAdapter  # optionally used if CLIP doesn't support pose injection


class TemporalAggregator(nn.Module):
    """Simple BiLSTM temporal aggregator producing per-frame features."""
    def __init__(self, in_dim: int, hidden_dim: int = 512, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=True,
                            batch_first=True)
        self.out_dim = hidden_dim * 2

    def forward(self, x, lengths=None):
        # x: (B, T, D)
        # optionally pack_padding for efficiency
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            out, _ = self.lstm(x)  # (B, T, 2*H)
        return out


class SLRNet(nn.Module):
    """
    Sign Language Recognition model integrating CLIP + Frozen PoseEncoder + Pose->Visual fusion.

    Usage:
        model = SLRNet(clip_model=clip, pose_pretrain_path="work_dir/pose_epoch_50.pth",
                       pose_cfg=dict(in_channels=6, hidden_channels=[64,128,256], parts=['body','left','right']),
                       temporal_cfg=dict(hidden_dim=512))
    Forward:
        logits, enc_out = model(batch_video, video_lengths, padded_pose, pose_mask)
    """

    def __init__(self,
                 clip_model,
                 vocab_size: int,
                 pose_pretrain_path: Optional[str] = None,
                 pose_cfg: dict = None,
                 freeze_pose: bool = True,
                 aggregator_cfg: dict = None,
                 aggregate_with_lstm: bool = True,
                 adaptor_cfg: Optional[dict] = None,
                 device: Optional[torch.device] = None):
        """
        clip_model: pre-instantiated CLIP/visual backbone (modified to accept pose_feat, preferably)
        vocab_size: number of output tokens for CTC / classifier
        pose_pretrain_path: optional checkpoint path that contains 'encoder_state_dict' or full state
        pose_cfg: dict to instantiate PoseEncoder if needed (in_channels, hidden_channels, parts, t_kernel, dropout)
        freeze_pose: freeze pose encoder params
        aggregator_cfg: dict passed to TemporalAggregator
        adaptor_cfg: dict for CrossModalAdapter if we need to do fusion here (fallback)
        """
        super().__init__()

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ------------- CLIP -------------
        self.clip = clip_model  # expect .encode_image(image, T, pose_feat=None, pose_mask=None)

        # CLIP's output embedding dim (assumes clip has .proj or similar)
        # Try to auto infer embed_dim from clip internals
        try:
            self.clip_dim = self.clip.proj.shape[0] if hasattr(self.clip, 'proj') else None
        except Exception:
            self.clip_dim = None

        # Fallback: request embed dim from visual module if possible
        if self.clip_dim is None:
            # some CLIP variants return embedding dim via .visual.attnpool.c_proj.in_features or similar
            try:
                self.clip_dim = getattr(self.clip.visual, 'output_dim', None)
            except Exception:
                self.clip_dim = None

        # if still unknown, user must set aggregator_cfg accordingly
        if self.clip_dim is None:
            # default to 512
            self.clip_dim = 512

        # ------------- Pose Encoder -------------
        pose_cfg = pose_cfg or {}
        pose_in_channels = pose_cfg.get('in_channels', 6)
        hidden_channels = pose_cfg.get('hidden_channels', [64, 128, 256])
        parts = pose_cfg.get('parts', ['body', 'left', 'right'])
        t_kernel = pose_cfg.get('t_kernel', 9)
        dropout = pose_cfg.get('dropout', 0.2)

        # Instantiate PoseEncoder
        self.pose_encoder = PoseEncoder(in_channels=pose_in_channels,
                                        hidden_channels=hidden_channels,
                                        parts=parts,
                                        t_kernel=t_kernel,
                                        dropout=dropout).to(self.device)

        # load pretrained checkpoint if provided
        if pose_pretrain_path:
            ckpt = torch.load(pose_pretrain_path, map_location='cpu')
            # ckpt may contain different keys; prefer 'encoder_state_dict' or full model
            if isinstance(ckpt, dict):
                if 'encoder_state_dict' in ckpt:
                    self.pose_encoder.load_state_dict(ckpt['encoder_state_dict'], strict=False)
                else:
                    # attempt to load matching keys
                    try:
                        self.pose_encoder.load_state_dict(ckpt, strict=False)
                    except Exception:
                        # try to find matching subkeys
                        for k in ckpt:
                            if 'encoder' in k and 'state' in k:
                                try:
                                    self.pose_encoder.load_state_dict(ckpt[k], strict=False)
                                    break
                                except Exception:
                                    pass
            else:
                # assume ckpt is state_dict
                self.pose_encoder.load_state_dict(ckpt, strict=False)

        # Freeze pose encoder if requested
        if freeze_pose:
            for p in self.pose_encoder.parameters():
                p.requires_grad = False
            self.pose_encoder.eval()

        # Pose output dimension: last hidden_channels
        self.pose_out_dim = hidden_channels[-1]

        # ------------- Adapter (fallback) -------------
        # If CLIP visual already accepts pose via encode_image, you may not need this adapter.
        # We still prepare one in case we need to fuse externally (adapter fusion).
        adaptor_cfg = adaptor_cfg or {}
        adapter_heads = adaptor_cfg.get('n_heads', 8)
        adapter_dropout = adaptor_cfg.get('dropout', 0.1)
        self.external_adapter = CrossModalAdapter(dim_rgb=self.clip_dim,
                                                  dim_pose=self.pose_out_dim,
                                                  n_heads=adapter_heads,
                                                  dropout=adapter_dropout).to(self.device)

        # ------------- Temporal aggregator and head -------------
        aggregator_cfg = aggregator_cfg or {}
        hidden_dim = aggregator_cfg.get('hidden_dim', 512)
        num_layers = aggregator_cfg.get('num_layers', 2)
        dropout = aggregator_cfg.get('dropout', 0.2)

        if aggregate_with_lstm:
            self.aggregator = TemporalAggregator(in_dim=self.clip_dim,
                                                 hidden_dim=hidden_dim,
                                                 num_layers=num_layers,
                                                 dropout=dropout)
            enc_dim = self.aggregator.out_dim
        else:
            # simple linear projection if not using LSTM
            self.aggregator = nn.Identity()
            enc_dim = self.clip_dim

        # CTC / classification head: map per-frame features to vocab logits
        self.ctc_head = nn.Sequential(
            nn.Linear(enc_dim, enc_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(enc_dim // 2, vocab_size)
        )

        # initialization helpers
        self._init_weights()

    def _init_weights(self):
        # zero-init adapter gates (if any)
        try:
            if hasattr(self.external_adapter, 'gate_attn'):
                nn.init.zeros_(self.external_adapter.gate_attn)
            if hasattr(self.external_adapter, 'gate_ffn'):
                nn.init.zeros_(self.external_adapter.gate_ffn)
        except Exception:
            pass

    # ------------------ utility helpers ------------------
    @staticmethod
    def _ensure_images_shape(images: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Accept video input in either:
          - (B, T, C, H, W)
          - (B*T, C, H, W)
        Return:
          images_reshaped (B*T, C, H, W), B, T
        """
        if images.dim() == 5:
            B, T, C, H, W = images.shape
            images_reshaped = images.view(B * T, C, H, W)
            return images_reshaped, B, T
        elif images.dim() == 4:
            # ambiguous: assume (B*T, C, H, W); user must provide T in call
            BxT, C, H, W = images.shape
            return images, None, None
        else:
            raise ValueError("Unsupported image tensor shape: must be (B,T,C,H,W) or (B*T,C,H,W)")

    @staticmethod
    def _process_pose_tensor(pose: torch.Tensor, pose_mask: Optional[torch.Tensor] = None):
        """
        pose: (B, C_pose, T, V)
        Return:
          - pose_input: (B*T, V, Dp) after pose_encoder and projection
          - pose_mask_flat: (B*T, V) boolean mask (True for pad) OR None
        """
        # validate dims
        if pose is None:
            return None, None, None, None

        # ensure float
        pose = pose.float()
        B, C, T, V = pose.shape
        return pose, B, C, T, V

    # ------------------ forward ------------------
    def forward(self, images: torch.Tensor,
                video_lengths: Optional[torch.Tensor] = None,
                poses: Optional[torch.Tensor] = None,
                pose_mask: Optional[torch.Tensor] = None):
        
        device = next(self.parameters()).device
        images = images.to(device)
        if poses is not None: poses = poses.to(device)
        if pose_mask is not None: pose_mask = pose_mask.to(device)
        if video_lengths is not None: video_lengths = video_lengths.to(device)

        # 1) Prepare images (reshape to B*T if needed)
        images_bt, B, T = self._ensure_images_shape(images)
        if B is None:
            if video_lengths is None:
                raise ValueError("Must provide 'video_lengths' to infer B and T.")
            T = int(video_lengths.max().item())
            B = images_bt.shape[0] // T

        # 2) Compute pose features
        pose_feat_for_fusion = None
        pose_key_pad = None
        
        if poses is not None:
            # === [Fix 1] Auto-permute (B, T, V, C) -> (B, C, T, V) ===
            # C_pose 应该是 pose_encoder 的 in_channels (比如 6)
            if poses.shape[-1] == self.pose_encoder.in_channels: 
                poses = poses.permute(0, 3, 1, 2).contiguous()
            
            # 确保现在是 (B, C, T, V)
            assert poses.dim() == 4 and poses.shape[1] == self.pose_encoder.in_channels, \
                f"Pose shape mismatch. Expected (B, {self.pose_encoder.in_channels}, T, V), got {poses.shape}"

            # Pass through PoseEncoder
            with torch.no_grad() if not any(p.requires_grad for p in self.pose_encoder.parameters()) else torch.enable_grad():
                pose_enc = self.pose_encoder(poses)  # Out: (B, Dp, T, V)
            
            # Check logic: PoseEncoder usually maintains T if stride=1.
            # If not, you must assert T match or resize pose_enc.
            assert pose_enc.shape[2] == T, f"PoseEncoder time dim {pose_enc.shape[2]} != Image T {T}. Check PoseEncoder stride."

            # Permute to (B, T, V, Dp)
            pose_enc = pose_enc.permute(0, 2, 3, 1).contiguous()
            _, _, Vp, Dp = pose_enc.shape

            # Flatten to (B*T, V, Dp) for CLIP interaction
            pose_feat_for_fusion = pose_enc.view(B * T, Vp, Dp)

            # Prepare Mask
            if pose_mask is not None:
                # 假设 0 代表 padding/无效点
                is_padding = (pose_mask == 0) 
                
                if pose_mask.dim() == 3: # (B, T, V)
                     pose_key_pad = is_padding.view(B * T, Vp)
                elif pose_mask.dim() == 2: # (B, V) -> broadcast to T
                     pose_key_pad = is_padding.unsqueeze(1).repeat(1, T, 1).view(B * T, Vp)
        
        # ... 后续调用 clip.encode_image ... (确保 model.py 已修改)
        # 3) Visual forward
        # ...

        # 3) Visual forward: call CLIP.encode_image (we rely on your modified CLIP)
        #    CLIP is expected to accept (image_batch, T, pose_feat=..., pose_mask=...)
        try:
            # images_bt: (B*T, C, H, W)
            # pass pose features and mask to CLIP when available
            if pose_feat_for_fusion is not None:
                # clip.encode_image should return (B*T, clip_dim)
                visual_emb = self.clip.encode_image(images_bt, T, pose_feat=pose_feat_for_fusion, pose_mask=pose_key_pad)
            else:
                visual_emb = self.clip.encode_image(images_bt, T)
        except TypeError:
            # Fallback: CLIP doesn't accept pose params — do external fusion
            #  - call clip.encode_image(images_bt) to get per-frame features and then apply external adapter
            visual_emb = self.clip.encode_image(images_bt, T)  # (B*T, clip_dim)
            if pose_feat_for_fusion is not None:
                # Need to obtain token-level visual embeddings (patch tokens), but typical CLIP returns pooled embedding.
                # We use the external adapter in a simplified manner: project pose to clip_dim, then add to pooled feature.
                # WARNING: this is a rough fallback — better to use a CLIP version that supports token-level fusion.
                pose_proj = self.external_adapter.pose_proj(pose_feat_for_fusion)  # (B*T, V, clip_dim)
                # aggregate pose tokens to single vector (mean)
                pose_ctx = pose_proj.mean(dim=1)  # (B*T, clip_dim)
                visual_emb = visual_emb + pose_ctx * 0.5  # small fusion factor

        # visual_emb: (B*T, clip_dim)  -> reshape to (B, T, clip_dim)
        visual_emb = visual_emb.view(B, T, -1)

        # 4) Temporal aggregation
        enc = self.aggregator(visual_emb, video_lengths)  # (B, T, enc_dim) if LSTM, else identity

        # 5) Classification head per time-step
        logits = self.ctc_head(enc)  # (B, T, vocab_size)

        return logits, enc
