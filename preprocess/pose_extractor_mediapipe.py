#!/usr/bin/env python3
"""
preprocess/pose_extractor_mediapipe.py (Multiprocessing Version)
"""

import os
import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
import cv2
from tqdm import tqdm
from functools import partial
import multiprocessing

try:
    import mediapipe as mp
except Exception as e:
    raise ImportError("mediapipe not installed. Install via `pip install mediapipe`.") from e

# ----------------- 之前的工具函数保持不变 -----------------
# 为了节省篇幅，这里复用了核心逻辑，但为了多进程安全，必须把 holistic 的创建放进 worker 里

def sorted_frame_files(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not files:
        for sub in folder.iterdir():
            if sub.is_dir():
                sub_files = [p for p in sub.iterdir() if p.is_file() and p.suffix.lower() in exts]
                files.extend(sub_files)
    files = sorted(files, key=lambda p: p.name)
    return files

def load_video_frames(video_path: Path, sample_rate: int = 1) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if idx % sample_rate == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames, fps

def safe_mediapipe_landmarks_to_array(landmarks, img_w: int, img_h: int, return_z: bool, coords_mode: str):
    if landmarks is None: return None
    arr = []
    for lm in landmarks:
        x, y = lm.x, lm.y
        z = getattr(lm, "z", 0.0)
        if coords_mode == "pixel":
            x *= img_w; y *= img_h
            z = z * img_w if z is not None else 0.0
        arr.append([x, y, z] if return_z else [x, y])
    return np.asarray(arr, dtype=np.float32)

def pad_or_fill(arr: Optional[np.ndarray], num_points: int, dims: int):
    if arr is None: return np.zeros((num_points, dims), dtype=np.float32)
    V = arr.shape[0]
    if V == num_points: return arr.astype(np.float32)
    elif V < num_points:
        pad = np.zeros((num_points - V, dims), dtype=np.float32)
        return np.vstack([arr.astype(np.float32), pad])
    else: return arr[:num_points, :dims].astype(np.float32)

def extract_sequence(frames: List[np.ndarray], holistic, return_z: bool, coords_mode: str, 
                     face_keep_indices: Optional[List[int]], body_num=33, hands_num=21, face_num=468):
    T = len(frames)
    dims = 3 if return_z else 2
    data = {
        "body_raw": np.zeros((T, body_num, dims), dtype=np.float32),
        "left_raw": np.zeros((T, hands_num, dims), dtype=np.float32),
        "right_raw": np.zeros((T, hands_num, dims), dtype=np.float32),
        "face_raw": np.zeros((T, len(face_keep_indices) if face_keep_indices else face_num, dims), dtype=np.float32),
        "mask_body": np.zeros((T, body_num), dtype=np.uint8),
        "mask_left": np.zeros((T, hands_num), dtype=np.uint8),
        "mask_right": np.zeros((T, hands_num), dtype=np.uint8),
        "mask_face": np.zeros((T, len(face_keep_indices) if face_keep_indices else face_num), dtype=np.uint8)
    }
    
    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Helper to process component
        def process_comp(landmarks, key_prefix, num_pts, keep_idxs=None):
            arr = safe_mediapipe_landmarks_to_array(landmarks, w, h, return_z, coords_mode)
            if arr is not None:
                if keep_idxs:
                    # Face subset
                    picked = []
                    for pid, idx in enumerate(keep_idxs):
                        if idx < arr.shape[0]:
                            picked.append(arr[idx])
                            data[f"mask_{key_prefix}"][i, pid] = 1
                        else:
                            picked.append([0]*dims)
                    data[f"{key_prefix}_raw"][i] = np.array(picked, dtype=np.float32)
                else:
                    data[f"{key_prefix}_raw"][i] = pad_or_fill(arr, num_pts, dims)
                    data[f"mask_{key_prefix}"][i, :min(arr.shape[0], num_pts)] = 1
            else:
                pass # Already zeros
                
        process_comp(res.pose_landmarks.landmark if res.pose_landmarks else None, "body", body_num)
        process_comp(res.left_hand_landmarks.landmark if res.left_hand_landmarks else None, "left", hands_num)
        process_comp(res.right_hand_landmarks.landmark if res.right_hand_landmarks else None, "right", hands_num)
        process_comp(res.face_landmarks.landmark if res.face_landmarks else None, "face", face_num, keep_idxs=face_keep_indices)
        
    return data

def compute_root_center(seq_body):
    # 23=left_hip, 24=right_hip
    lh, rh = 23, 24
    T = seq_body.shape[0]
    return (seq_body[:, lh] + seq_body[:, rh]) / 2.0

def center_and_scale(all_raw, scale_by="shoulder"):
    body = all_raw["body_raw"]
    root = compute_root_center(body)
    
    if scale_by == "shoulder":
        # 11=left_shoulder, 12=right_shoulder
        dists = np.linalg.norm(body[:, 11, :] - body[:, 12, :], axis=-1)
    else: # hip
        dists = np.linalg.norm(body[:, 23, :] - body[:, 24, :], axis=-1)
    
    dists[dists == 0] = 1.0
    scale = dists[:, None]
    
    out = {}
    for k in ["body_raw", "left_raw", "right_raw", "face_raw"]:
        out[k.replace("_raw", "_rel")] = (all_raw[k] - root[:, None, :]) / scale[:, None, :]
    return out

def compute_motion(all_arrays, mode="normalized"):
    suffix = "_rel" if mode == "normalized" else "_raw"
    out = {}
    for k in ["body", "left", "right", "face"]:
        arr = all_arrays.get(f"{k}{suffix}", all_arrays.get(f"{k}_raw"))
        if arr is None: continue
        motion = np.zeros_like(arr)
        motion[1:] = arr[1:] - arr[:-1]
        out[f"{k}_motion"] = motion
    return out

# ----------------- WORKER FUNCTION (多进程核心) -----------------
def worker_process_item(item_info, args_dict):
    """
    单个进程处理单个视频的函数
    """
    name, path_str = item_info
    path = Path(path_str)
    out_dir = Path(args_dict['output_dir'])
    out_file = out_dir / f"{name}.npz"
    
    # 如果已存在且不覆盖，跳过
    if out_file.exists() and not args_dict['overwrite']:
        return "skipped"

    # 在进程内初始化 MediaPipe (不能跨进程共享)
    mp_holistic = mp.solutions.holistic
    # 注意：这里开启 static_image_mode=True 以提高准确率，虽然慢一点但在多进程下可以接受
    with mp_holistic.Holistic(static_image_mode=True, model_complexity=1, 
                              min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        # 加载数据
        if args_dict['mode'] == "frames":
            frame_paths = sorted_frame_files(path)
            if not frame_paths: return f"failed_no_frames: {name}"
            # 简单采样
            frames = []
            for i, p in enumerate(frame_paths):
                if i % args_dict['sample_rate'] == 0:
                    img = cv2.imread(str(p))
                    if img is not None: frames.append(img)
            fps = 25.0 # 默认
        else:
            frames, fps = load_video_frames(path, args_dict['sample_rate'])
            
        if not frames: return f"failed_empty: {name}"

        # 提取
        try:
            raw = extract_sequence(frames, holistic, args_dict['return_z'], args_dict['coords_mode'], 
                                   args_dict['face_keep_indices'])
        except Exception as e:
            return f"error_extract: {name} {e}"

        # 后处理
        to_save = raw.copy()
        
        if args_dict['save_relative']:
            rel = center_and_scale(raw, scale_by=args_dict['normalize'])
            to_save.update(rel)
            
        if args_dict['save_motion']:
            # 如果有 relative 优先用 relative 计算 motion
            motion_base = rel if args_dict['save_relative'] and args_dict['motion_mode'] == 'normalized' else raw
            motion = compute_motion(motion_base, args_dict['motion_mode'])
            to_save.update(motion)

        # 保存
        meta = {
            "sample_name": name,
            "orig_num_frames": len(frames), # 近似
            "saved_keys": list(to_save.keys())
        }
        np.savez_compressed(str(out_file), meta=meta, **to_save)
        
    return "success"

# ----------------- MAIN -----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--mode", choices=["frames", "videos"], default="frames")
    p.add_argument("--sample_rate", type=int, default=1)
    p.add_argument("--coords", choices=["normalized", "pixel"], default="normalized")
    p.add_argument("--no_z", action="store_true")
    p.add_argument("--face_keep_indices", type=str, default=None)
    p.add_argument("--normalize", choices=["none", "shoulder", "hip"], default="shoulder")
    p.add_argument("--save_relative", action="store_true")
    p.add_argument("--save_motion", action="store_true")
    p.add_argument("--motion_mode", default="normalized")
    p.add_argument("--overwrite", action="store_true")
    # 新增多进程参数
    p.add_argument("--num_workers", type=int, default=8, help="Number of CPU processes")
    args = p.parse_args()

    # 准备参数字典 (传给 worker)
    face_indices = None
    if args.face_keep_indices:
        with open(args.face_keep_indices, "r") as f: face_indices = json.load(f)
        
    args_dict = {
        "output_dir": args.output_dir,
        "overwrite": args.overwrite,
        "mode": args.mode,
        "sample_rate": args.sample_rate,
        "return_z": not args.no_z,
        "coords_mode": args.coords,
        "face_keep_indices": face_indices,
        "normalize": args.normalize,
        "save_relative": args.save_relative,
        "save_motion": args.save_motion,
        "motion_mode": args.motion_mode
    }

    in_dir = Path(args.input_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 扫描文件
    items = []
    if args.mode == "frames":
        for sub in sorted(in_dir.iterdir()):
            if sub.is_dir(): items.append((sub.name, str(sub)))
    else:
        for pth in sorted(in_dir.iterdir()):
            if pth.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
                items.append((pth.stem, str(pth)))
    
    print(f"Found {len(items)} samples. Using {args.num_workers} workers.")
    
    # 开启多进程池
    # 自动根据 CPU 核心数调整，AutoDL 上一般核心很多
    num_procs = args.num_workers
    if num_procs < 1: num_procs = multiprocessing.cpu_count()
    
    with multiprocessing.Pool(processes=num_procs) as pool:
        # 使用 imap_unordered 并显示进度条
        # partial 用于固定 args_dict 参数
        func = partial(worker_process_item, args_dict=args_dict)
        
        results = list(tqdm(pool.imap_unordered(func, items), total=len(items)))
        
    # 简单的统计
    success = results.count("success")
    skipped = results.count("skipped")
    print(f"Done. Success: {success}, Skipped: {skipped}, Total: {len(items)}")

if __name__ == "__main__":
    main()