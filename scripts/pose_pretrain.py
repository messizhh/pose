#!/usr/bin/env python3
# scripts/pose_pretrain.py
"""
Self-supervised pretraining for Pose Encoder.

- Uses PosePretrainDataset (returns dict with 'feat', 'mask_time', 'mask_nodes', 'orig_len', 'sample_name')
- Model: PoseEncoder + PretrainModel (reconstruction head + prediction head)
- Loss:
    recon_loss = MSE over valid frames/nodes (weighted by mask_time & mask_nodes)
    pred_loss  = MSE over masked frames only (weighted)
- Saves best checkpoint by val loss.
"""

import os
import sys
import random
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# ensure project root is importable
sys.path.append(os.getcwd())

from dataset.pose_dataset import PosePretrainDataset
from modules.pose.pose_encoder import PoseEncoder, PretrainModel


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mask_random_contiguous(T: int, mask_portion: float = 0.2) -> Tuple[int, int]:
    """
    Return start, length of a contiguous mask segment of length int(T*mask_portion)
    If mask_portion == 0 -> (0,0)
    """
    if mask_portion <= 0.0:
        return 0, 0
    length = max(1, int(T * mask_portion))
    if length >= T:
        return 0, T
    start = random.randint(0, T - length)
    return start, length


def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor, node_time_mask: torch.Tensor, eps: float = 1e-6):
    """
    pred, target: (N, C, T, V)
    node_time_mask: (N, T, V) with 1 for valid, 0 for invalid (padding / missing detection)
    Weighted MSE averaged over valid elements.
    """
    assert pred.shape == target.shape
    N, C, T, V = pred.shape
    # expand mask to channels
    mask = node_time_mask.unsqueeze(1)  # (N,1,T,V)
    sq = (pred - target) ** 2
    sq_masked = sq * mask  # invalid positions zeroed out
    num_valid = mask.sum() * C  # each channel counts
    if num_valid.item() < 1.0:
        # fallback safe mean
        return sq_masked.mean()
    loss = sq_masked.sum() / (num_valid + eps)
    return loss


def train_one_epoch(model: nn.Module, optimizer: optim.Optimizer, loader: DataLoader, device: torch.device,
                    mask_portion: float = 0.2):
    model.train()
    total_loss = 0.0
    total_samples = 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        feat = batch['feat'].to(device)            # (N, C, T, V)
        mask_time = batch['mask_time'].to(device)  # (N, T)
        mask_nodes = batch['mask_nodes'].to(device)  # (N, T, V)

        N, C, T, V = feat.shape

        # create mask_time_masked: per-sample random contiguous segment masked
        mask_time_mask = torch.ones((N, T), dtype=torch.bool, device=device)
        for i in range(N):
            s, l = mask_random_contiguous(T, mask_portion)
            if l > 0:
                mask_time_mask[i, s:s + l] = False  # False means masked (we will zero input there)

        # create masked input
        # create masked input
        masked_input = feat.clone()
        # masked frames set to zero (both coords and motion)
        # mask_time_mask: (N, T) -> expand to (N, 1, T, 1) to broadcast over C and V
        mask_condition = (~mask_time_mask).unsqueeze(1).unsqueeze(-1)
        masked_input = masked_input.masked_fill(mask_condition, 0.0)

        # forward
        recon_out, pred_out = model(masked_input)  # both (N, C, T, V)

        # prepare node_time mask for loss (valid frames * valid nodes)
        # mask_time: (N, T) float (1/0), convert to (N, T, 1) to broadcast
        node_time_mask_full = (mask_nodes * mask_time.unsqueeze(-1)).float()  # (N, T, V)

        # recon loss: over all valid frames/nodes
        loss_recon = weighted_mse_loss(recon_out, feat, node_time_mask_full)

        # pred loss: only on masked frames (and valid nodes)
        # masked frames boolean per-sample mask_time_mask False -> masked
        masked_time_bool = (~mask_time_mask).float()  # (N, T) 1 for masked frames
        if masked_time_bool.sum() > 0:
            masked_node_time = node_time_mask_full * masked_time_bool.unsqueeze(-1)  # (N,T,V)
            loss_pred = weighted_mse_loss(pred_out, feat, masked_node_time)
        else:
            loss_pred = torch.tensor(0.0, device=device)

        loss = loss_recon + 0.5 * loss_pred

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * N
        total_samples += N
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / max(1, total_samples)
    return avg_loss


def validate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="Valid", leave=False)
        for batch in pbar:
            feat = batch['feat'].to(device)
            mask_time = batch['mask_time'].to(device)
            mask_nodes = batch['mask_nodes'].to(device)

            # no masking during validation — just evaluate reconstruction quality
            recon_out, pred_out = model(feat)  # pass full input
            node_time_mask_full = (mask_nodes * mask_time.unsqueeze(-1)).float()
            loss_recon = weighted_mse_loss(recon_out, feat, node_time_mask_full)
            total_loss += loss_recon.item() * feat.shape[0]
            total_samples += feat.shape[0]
            pbar.set_postfix({'val_loss': f'{loss_recon.item():.4f}'})
    avg = total_loss / max(1, total_samples)
    return avg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data/phoenix2014/pose_features")
    p.add_argument("--save_dir", type=str, default="./work_dir/pose_pretrain")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mask_portion", type=float, default=0.2, help="portion of frames to mask per sample")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    os.makedirs(args.save_dir, exist_ok=True)

    # dataset
    train_ds = PosePretrainDataset(args.data_root, split="train", max_length=300, parts=['body', 'left', 'right'],
                                   use_motion=True, random_crop=True)
    val_ds = PosePretrainDataset(args.data_root, split="dev", max_length=300, parts=['body', 'left', 'right'],
                                 use_motion=True, random_crop=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # model (match PoseEncoder signature from modules)
    # choose hidden channels list according to depth desired
    hidden_channels = [64, 128, 256]  # you can modify to [128,256,256] for bigger model
    encoder = PoseEncoder(in_channels=6, hidden_channels=hidden_channels, parts=['body', 'left', 'right'],
                          t_kernel=9, dropout=0.2)
    model = PretrainModel(encoder=encoder, in_channels=6).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()  # kept for sanity, we use weighted_mse_loss for main

    best_val = float('inf')
    start_epoch = 0

    print("Start Pose Pre-training...")
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = train_one_epoch(model, optimizer, train_loader, device, mask_portion=args.mask_portion)
        val_loss = validate(model, val_loader, device)

        print(f"Epoch {epoch+1} TrainLoss={train_loss:.4f} ValLoss={val_loss:.4f}")

        # save checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'hidden_channels': hidden_channels
        }
        ckpt_path = Path(args.save_dir) / f"pose_epoch_{epoch+1}.pth"
        torch.save(ckpt, ckpt_path)

        # save best
        if val_loss < best_val:
            best_val = val_loss
            best_path = Path(args.save_dir) / "pose_best.pth"
            torch.save(ckpt, best_path)
            print(f"Saved best checkpoint to {best_path} (val_loss={val_loss:.4f})")

    print("Pretraining finished.")


if __name__ == "__main__":
    main()
