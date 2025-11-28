#!/usr/bin/env python3
"""
scripts/gen_phoenix_info.py

根据帧文件夹结构自动生成 preprocess/phoenix2014/{train,dev,test}_info.npy
Assumes frames root layout:
  <frames_root>/
    train/<sample_folder>/
    dev/<sample_folder>/
    test/<sample_folder>/

Each entry will be a dict:
  idx: {
    'folder': '<mode>/<sample_folder>',
    'fileid': '<sample_folder>',
    'label': '',               # placeholder (no gloss info)
    'original_info': ''        # placeholder
  }
"""
import os
import argparse
import numpy as np

def gen_info(frames_root, out_dir, modes=('train','dev','test')):
    os.makedirs(out_dir, exist_ok=True)
    for mode in modes:
        mode_dir = os.path.join(frames_root, mode)
        if not os.path.isdir(mode_dir):
            print(f"[WARN] mode dir not found: {mode_dir}, skipping")
            continue
        folders = sorted([d for d in os.listdir(mode_dir) if os.path.isdir(os.path.join(mode_dir, d))])
        info = {}
        for i, fname in enumerate(folders):
            # folder stored as "mode/foldername" to match BaseFeeder join logic
            info[i] = {
                'folder': f"{mode}/{fname}",
                'fileid': fname,
                'label': '',
                'original_info': ''
            }
        out_path = os.path.join(out_dir, f"{mode}_info.npy")
        np.save(out_path, info)
        print(f"Saved {len(info)} entries to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_root', required=True, help='path to fullFrame-256x256px folder')
    parser.add_argument('--out_preprocess', default='./preprocess/phoenix2014', help='where to write *_info.npy')
    parser.add_argument('--modes', nargs='+', default=['train','dev','test'])
    args = parser.parse_args()
    gen_info(args.frames_root, args.out_preprocess, modes=args.modes)
