#!/usr/bin/env python3
# main.py - training entry for AdaptSign + Pose fusion (Corrected Version)

import os
import sys
import time
import yaml
import argparse
import numpy as np
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# add project root
ROOT = Path(__file__).resolve().parents[0]
sys.path.append(str(ROOT))

# imports
from dataset.dataloader_video_pose import VideoPoseDataset
from slr_network import SLRNet
# Import original clip module for loading standard weights
from modules.openai import clip as clip_module
from modules.openai import model as clip_model_module

try:
    build_clip_from_state = getattr(clip_model_module, "build_model", None)
except Exception:
    build_clip_from_state = None

# Decode helper for simple validation check
from utils.decode import Decode

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="path to yaml config")
    p.add_argument("--gpu", type=str, default="0", help="GPU id(s), e.g. '0'")
    p.add_argument("--resume", type=str, default=None, help="checkpoint to resume")
    return p.parse_args()

def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_clip_model(cfg, device):
    """
    Robust CLIP builder:
    1. Try loading from checkpoint state_dict (if provided).
    2. Fallback to standard clip.load() (automatic download/cache).
    """
    model_args = cfg.get("model_args", {})
    ckpt_path = model_args.get("clip_checkpoint", None)
    c2d_type = model_args.get("c2d_type", "ViT-B/16")

    # Method 1: Load from specific checkpoint file (e.g. finetuned weights)
    if ckpt_path and os.path.exists(ckpt_path) and build_clip_from_state:
        print(f"[CLIP] Loading custom state_dict from {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" in sd: sd = sd["model_state_dict"]
        model = build_clip_from_state(sd).to(device)
        return model

    # Method 2: Standard Load (Original AdaptSign way)
    print(f"[CLIP] Loading standard {c2d_type} weights...")
    # jit=False is important to allow modification (injection of adapters)
    model = clip_module.load(c2d_type, device=device, jit=False)
    return model

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    iters = 0
    pbar = tqdm(dataloader, desc=f"Train Ep {epoch}", ncols=100)
    
    for batch in pbar:
        # Unpack batch (aligned with VideoPoseDataset.collate_fn)
        # video, v_lens, pose, p_mask, labels, l_lens, ids
        images, video_lengths, poses, pose_mask, labels, label_lens, ids = batch
        
        images = images.to(device)         # (B, T, C, H, W)
        video_lengths = video_lengths.to(device)
        poses = poses.to(device)           # (B, 6, T, 75)
        pose_mask = pose_mask.to(device)   # (B, T, 75)
        labels = labels.to(device)
        label_lens = label_lens.to(device)

        optimizer.zero_grad()

        # Forward
        logits, enc = model(images, video_lengths, poses=poses, pose_mask=pose_mask)
        # logits: (B, T, vocab)
        
        # Prepare for CTCLoss (LogSoftmax & Permute T,B,C)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        log_probs_tbc = log_probs.permute(1, 0, 2)

        # Flatten targets for CTC
        flat_targets = []
        for i in range(labels.size(0)):
            L = int(label_lens[i].item())
            if L > 0:
                flat_targets.append(labels[i, :L])
        
        if not flat_targets:
            continue
            
        targets_concat = torch.cat(flat_targets).to(device)
        
        # CTC Loss
        # input_lengths should match the time dimension of logits
        # Note: If your visual backbone does downsampling (e.g. stride), you must adjust input_lengths here.
        # SLRNet aggregator assumes stride=1 or handles it. If stride=1:
        loss = criterion(log_probs_tbc, targets_concat, video_lengths, label_lens)

        if torch.isnan(loss):
            print("[WARN] NaN loss detected, skipping step")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        iters += 1
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return total_loss / max(1, iters)

def validate(model, dataloader, criterion, device, decoder=None):
    model.eval()
    total_loss = 0.0
    iters = 0
    
    # Store first batch predictions for visualization
    sample_preds = []
    sample_gts = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Eval", ncols=100)
        for batch_idx, batch in enumerate(pbar):
            images, video_lengths, poses, pose_mask, labels, label_lens, ids = batch
            
            images = images.to(device)
            video_lengths = video_lengths.to(device)
            poses = poses.to(device)
            pose_mask = pose_mask.to(device)
            labels = labels.to(device)
            label_lens = label_lens.to(device)

            logits, _ = model(images, video_lengths, poses=poses, pose_mask=pose_mask)
            
            # Loss Calc
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            log_probs_tbc = log_probs.permute(1, 0, 2)
            
            flat_targets = []
            for i in range(labels.size(0)):
                L = int(label_lens[i].item())
                if L > 0: flat_targets.append(labels[i, :L])
            
            if flat_targets:
                targets_concat = torch.cat(flat_targets).to(device)
                loss = criterion(log_probs_tbc, targets_concat, video_lengths, label_lens)
                total_loss += loss.item()
            
            iters += 1
            
            # Simple Decoding for the first batch to check progress
            if batch_idx == 0 and decoder is not None:
                # Greedy decode
                # logits: (B, T, V)
                decoded_sentences = decoder.decode(logits, video_lengths, batch_first=True, probs=False)
                for i in range(min(2, len(decoded_sentences))):
                    sample_preds.append(decoded_sentences[i])
                    # Get GT string (helper needed, simplified here)
                    # sample_gts.append("GT_Placeholder") 
                    
    print(f"    [Sample Pred]: {sample_preds[0] if sample_preds else 'N/A'}")
    
    return total_loss / max(1, iters)

def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    # Setup Device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MAIN] Device: {device}")
    
    set_seed(cfg.get("random_seed", 42))

    # --- 1. Load Dictionary (The Fix) ---
    # Phoenix2014 dict is usually a .npy file
    # We look for 'dict_path' in dataset_info or assume a standard path
    dataset_info_path = f"./configs/{cfg['dataset']}.yaml"
    if os.path.exists(dataset_info_path):
        with open(dataset_info_path, 'r') as f:
            ds_info = yaml.safe_load(f)
            dict_path = ds_info.get('dict_path', './preprocess/phoenix2014/gloss_dict.npy')
    else:
        dict_path = './preprocess/phoenix2014/gloss_dict.npy'

    print(f"[MAIN] Loading Gloss Dict from: {dict_path}")
    if os.path.exists(dict_path):
        # Load numpy object
        gloss_dict = np.load(dict_path, allow_pickle=True).item()
    else:
        raise FileNotFoundError(f"Gloss dict not found at {dict_path}")
    
    vocab_size = len(gloss_dict) + 1 # +1 for blank
    print(f"[MAIN] Vocab Size: {vocab_size}")

    # --- 2. Build Datasets ---
    # Config compatibility: check 'feeder_args' first (from baseline.yaml), else 'data_args'
    data_cfg = cfg.get("feeder_args", cfg.get("data_args", {}))
    
    # Ensure Pose Root is set
    pose_root = data_cfg.get("pose_root", "./data/phoenix2014/pose_features_aligned")
    prefix = data_cfg.get("prefix", "./dataset/phoenix2014/phoenix-2014-multisigner")
    
    train_set = VideoPoseDataset(
        prefix=prefix,
        gloss_dict=gloss_dict,
        pose_root=pose_root,
        dataset=cfg.get("dataset", "phoenix2014"),
        mode="train",
        input_size=data_cfg.get("input_size", 224),
        frame_interval=data_cfg.get("frame_interval", 1),
        transform_mode=True
    )
    
    dev_set = VideoPoseDataset(
        prefix=prefix,
        gloss_dict=gloss_dict,
        pose_root=pose_root,
        dataset=cfg.get("dataset", "phoenix2014"),
        mode="dev",
        input_size=data_cfg.get("input_size", 224),
        transform_mode=False
    )
    
    train_loader = DataLoader(
        train_set, 
        batch_size=cfg.get("batch_size", 2), 
        shuffle=True, 
        num_workers=cfg.get("num_worker", 4),
        collate_fn=VideoPoseDataset.collate_fn,
        drop_last=True
    )
    
    dev_loader = DataLoader(
        dev_set, 
        batch_size=cfg.get("test_batch_size", 2), 
        shuffle=False, 
        num_workers=cfg.get("num_worker", 4),
        collate_fn=VideoPoseDataset.collate_fn
    )

    # --- 3. Build Model ---
    # Build CLIP first
    clip_model = build_clip_model(cfg, device)
    
    # Build SLRNet
    model_args = cfg.get("model_args", {})
    # Override vocab_size in config just in case
    model_args['vocab_size'] = vocab_size 
    
    model = SLRNet(
        clip_model=clip_model,
        vocab_size=vocab_size,
        pose_pretrain_path=model_args.get("pose_pretrain_path", None),
        pose_cfg=model_args.get("pose_cfg", {}),
        freeze_pose=model_args.get("freeze_pose", True),
        aggregator_cfg=model_args.get("aggregator_cfg", {}),
        adaptor_cfg=model_args.get("adaptor_cfg", {})
    ).to(device)

    # Freeze CLIP?
    if model_args.get("freeze_clip", True):
        for p in model.clip.parameters():
            p.requires_grad = False
    
    # --- 4. Setup Training ---
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.get("optimizer_args", {}).get("base_lr", 1e-4)
    )
    
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True).to(device)
    
    # Helper for decoding string
    from utils.decode import Decode
    decoder = Decode(gloss_dict, vocab_size, method='beam', beam_width=5, blank_index=0)

    # Output Dir
    work_dir = Path(cfg.get("work_dir", "./work_dir/pose_experiment"))
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config for reproducibility
    with open(work_dir / "config_run.yaml", "w") as f:
        yaml.dump(cfg, f)

    start_epoch = 0
    best_loss = float("inf")

    # --- 5. Loop ---
    num_epochs = cfg.get("num_epoch", 40)
    print(f"[MAIN] Start training for {num_epochs} epochs...")
    
    for epoch in range(start_epoch, num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, ctc_loss, device, epoch)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
        
        val_loss = validate(model, dev_loader, ctc_loss, device, decoder)
        print(f"Epoch {epoch}: Val Loss   = {val_loss:.4f}")
        
        # Save Checkpoint
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            torch.save(model.state_dict(), work_dir / "best_model.pth")
            print("  [*] Saved Best Model")
        
        if epoch % 5 == 0:
            torch.save(model.state_dict(), work_dir / f"checkpoint_ep{epoch}.pth")

if __name__ == "__main__":
    main()