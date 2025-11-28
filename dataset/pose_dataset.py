# dataset/pose_dataset.py
import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Optional, Tuple

class PosePretrainDataset(Dataset):
    """
    Pose pretraining dataset.

    Each sample is a .npz produced by preprocess/pose_extractor_mediapipe.py.
    This dataset can concatenate multiple parts (body / left / right / face)
    along the node dimension V, and optionally include motion channels.

    Output per item:
      dict {
        'feat': Tensor (C, T, V_total)  # dtype=torch.float32
        'mask_time': Tensor (T,)        # 1.0 for valid frames else 0.0
        'mask_nodes': Tensor (T, V_total) # 1/0 per node
        'orig_len': int                 # original number of frames before cropping/pad
        'sample_name': str
      }

    Args:
      root_path: root folder containing split subfolder (train/dev/test) with .npz files.
      split: "train"/"dev"/"test"
      max_length: number of frames to pad/crop to (T_out)
      parts: list of parts to use, e.g. ['body','left','right']
      use_motion: whether to load *_motion channels and concatenate them to coords
      coords_priority: list, order to pick coords per part, e.g. ['rel','raw']
      random_crop: if True (typically for train) use random crop when sequence longer than max_length
      pad_value: value to pad with (default 0.0)
    """

    def __init__(self,
                 root_path: str,
                 split: str = "train",
                 max_length: int = 300,
                 parts: List[str] = ("body",),
                 use_motion: bool = True,
                 coords_priority: List[str] = ("rel", "raw"),
                 random_crop: bool = True,
                 pad_value: float = 0.0):
        self.root = Path(root_path) / split
        self.files = sorted(glob.glob(str(self.root / "*.npz")))
        self.max_length = int(max_length)
        self.parts = list(parts)
        self.use_motion = bool(use_motion)
        self.coords_priority = list(coords_priority)
        self.random_crop = bool(random_crop)
        self.pad_value = float(pad_value)

        if len(self.files) == 0:
            raise RuntimeError(f"No .npz files found in {self.root}")

        # Determine expected V_total by loading first file (assume consistent across dataset)
        sample0 = np.load(self.files[0], allow_pickle=True)
        V_total = 0
        channels_per_point = None
        for p in self.parts:
            # find coords key
            for pr in self.coords_priority:
                key = f"{p}_{pr}"
                if key in sample0:
                    V_total += sample0[key].shape[1]
                    channels_per_point = sample0[key].shape[2]  # usually 2 or 3
                    break
            else:
                # fallback to raw
                key = f"{p}_raw"
                if key in sample0:
                    V_total += sample0[key].shape[1]
                    channels_per_point = sample0[key].shape[2]
                else:
                    raise RuntimeError(f"Part {p} not found in sample {self.files[0]}")
            # if motion enabled, channels double (coords + motion)
        if self.use_motion:
            self.channels_per_point = (channels_per_point * 2)
        else:
            self.channels_per_point = channels_per_point
        self.V_total = V_total  # number of nodes after concatenation

        print(f"PosePretrainDataset(split={split}) found {len(self.files)} files, "
              f"max_length={self.max_length}, parts={self.parts}, V_total={self.V_total}, "
              f"channels_per_point={self.channels_per_point}")

    def __len__(self):
        return len(self.files)

    def _load_part_arrays(self, data: np.lib.npyio.NpzFile, part: str):
        """
        Load arrays for a given part.
        Returns:
          coords: ndarray (T, V_part, C)  (C usually 2 or 3)
          motion: ndarray (T, V_part, C) or zeros if not present or use_motion False
          mask_nodes: ndarray (T, V_part) (0/1 uint8)
        """
        coords = None
        for pr in self.coords_priority:
            key = f"{part}_{pr}"
            if key in data:
                coords = data[key]
                break
        if coords is None:
            # fallback to raw
            key = f"{part}_raw"
            if key in data:
                coords = data[key]
            else:
                raise KeyError(f"Neither {part}_rel nor {part}_raw found in file.")

        # motion
        if self.use_motion:
            motion_key = f"{part}_motion"
            if motion_key in data:
                motion = data[motion_key]
            else:
                # zeros same shape as coords
                motion = np.zeros_like(coords)
        else:
            motion = None

        # try with conventional naming
        mask_key = f"mask_{part}"
        mask_nodes = data[mask_key] if mask_key in data else (np.ones((coords.shape[0], coords.shape[1]), dtype=np.uint8))

        return coords.astype(np.float32), (motion.astype(np.float32) if motion is not None else None), mask_nodes.astype(np.uint8)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        data = np.load(path, allow_pickle=True)
        sample_name = Path(path).stem

        # Load and concat parts along V dimension
        part_feats = []
        part_masks = []
        orig_len = None
        for part in self.parts:
            coords, motion, mask_nodes = self._load_part_arrays(data, part)
            if orig_len is None:
                orig_len = coords.shape[0]
            else:
                # sanity check lengths
                if coords.shape[0] != orig_len:
                    raise ValueError(f"Part {part} has length {coords.shape[0]} != {orig_len} in {path}")

            if self.use_motion:
                # concat coords and motion along channel dim per point
                # coords (T, V, C), motion (T, V, C) -> (T, V, 2C)
                feat_part = np.concatenate([coords, motion], axis=-1)
            else:
                feat_part = coords  # (T, V, C)

            part_feats.append(feat_part)         # list of (T, V_part, channels_per_point)
            part_masks.append(mask_nodes)        # list of (T, V_part)

        # concatenate along node dimension V
        feat = np.concatenate(part_feats, axis=1)      # (T, V_total, channels_per_point)
        mask_nodes = np.concatenate(part_masks, axis=1)  # (T, V_total)

        # sanitize NaN/inf
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

        T_orig = feat.shape[0]
        V_total = feat.shape[1]
        C_per_point = feat.shape[2]

        # cropping / padding in time dimension
        if T_orig > self.max_length:
            if self.random_crop:
                start = np.random.randint(0, T_orig - self.max_length + 1)
            else:
                # center crop
                start = (T_orig - self.max_length) // 2
            feat_cropped = feat[start:start + self.max_length]
            mask_nodes_cropped = mask_nodes[start:start + self.max_length]
            valid_len = self.max_length
        else:
            # pad at end
            pad_len = self.max_length - T_orig
            if pad_len > 0:
                pad_feat = np.full((pad_len, V_total, C_per_point), fill_value=self.pad_value, dtype=feat.dtype)
                feat_cropped = np.concatenate([feat, pad_feat], axis=0)
                pad_mask = np.zeros((pad_len, V_total), dtype=mask_nodes.dtype)
                mask_nodes_cropped = np.concatenate([mask_nodes, pad_mask], axis=0)
            else:
                feat_cropped = feat
                mask_nodes_cropped = mask_nodes
            valid_len = min(T_orig, self.max_length)

        # transpose to (C, T, V)
        # currently feat_cropped: (T, V, channels_per_point)
        feat_tensor = torch.from_numpy(feat_cropped).float().permute(2, 0, 1).contiguous()
        mask_time = torch.zeros(self.max_length, dtype=torch.float32)
        mask_time[:valid_len] = 1.0
        mask_nodes_tensor = torch.from_numpy(mask_nodes_cropped).float()  # (T, V_total)

        return {
            'feat': feat_tensor,             # (C, T, V)
            'mask_time': mask_time,          # (T,)
            'mask_nodes': mask_nodes_tensor, # (T, V)
            'orig_len': int(T_orig),
            'sample_name': sample_name
        }
