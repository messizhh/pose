#!/usr/bin/env python3
import os
import argparse
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

def load_info(info_path):
    d = np.load(info_path, allow_pickle=True).item()
    # d = { idx: {"folder": "...", "fileid": "...", "label": "...", "original_info": "..."} }
    return d

def load_pose_file(pose_dir, fileid):
    """Load pose file with fallback matching."""

    exact = os.path.join(pose_dir, fileid + ".npz")
    if os.path.exists(exact):
        return exact

    # fallback: match prefix (for cases where frame uses -0, but pose starts at -1)
    prefix = fileid.split("-")[0] + "-" + fileid.split("-")[1] + "-" + fileid.split("-")[2] + "-" + fileid.split("-")[3]

    # shorter fallback
    prefix = fileid.rsplit("-", 1)[0]  # "01April_2010_Thursday_heute_default"

    cand = [n for n in os.listdir(pose_dir) if n.startswith(prefix)]
    if len(cand) == 0:
        return None

    # pick closest id
    # fileid ends with "xxx-default-0"
    try:
        target_num = int(fileid.split("-")[-1])
    except:
        target_num = 0

    def get_num(name):
        try:
            return int(name.split("-")[-1].replace(".npz", ""))
        except:
            return 99999

    cand_sorted = sorted(cand, key=lambda x: abs(get_num(x) - target_num))
    return os.path.join(pose_dir, cand_sorted[0])

def process_one(args):
    fi, pose_in_dir, pose_out_dir, target_v = args

    fileid = fi["fileid"]
    pose_file = load_pose_file(pose_in_dir, fileid)

    if pose_file is None:
        return False, f"[MISS] {fileid}"

    try:
        data = np.load(pose_file, allow_pickle=True)
        
        # ============ 修改开始 ============
        # 1. 既然没有 "pose" 键，我们需要把各个部位拼起来
        # 常见的 SLR 做法是：躯干 + 左手 + 右手
        # 注意：这里假设数据的形状是 (T, V, C)，我们沿着 V (axis=1) 拼接
        
        p_body = data['body_raw']
        p_left = data['left_raw']
        p_right = data['right_raw']
        
        # 拼接成一个大数组
        pose = np.concatenate([p_body, p_left, p_right], axis=1)
        
        # (如果你还想要人脸，可以把 data['face_raw'] 也就加进去，但通常不需要)
        # ============ 修改结束 ============

    except Exception as e:
        # 这里把具体错误 e 打印出来，方便调试
        return False, f"[BROKEN] {pose_file} | Error: {e}"

    # ... 下面保持原样 ...
    # If number of nodes mismatch, pad or truncate
    T, V, C = pose.shape
    if V != target_v:
        if V < target_v:
            pad = np.zeros((T, target_v - V, C), dtype=pose.dtype)
            pose = np.concatenate([pose, pad], axis=1)
        else:
            pose = pose[:, :target_v]

    # save
    out_path = os.path.join(pose_out_dir, fileid + ".npz")
    np.savez_compressed(out_path, pose=pose)

    return True, f"[OK] {fileid}"

def process_mode(prefix, dataset, mode, in_pose_dir, out_pose_dir, target_v):
    info_path = f"./preprocess/{dataset}/{mode}_info.npy"
    info = load_info(info_path)

    os.makedirs(os.path.join(out_pose_dir, mode), exist_ok=True)
    pose_in = os.path.join(in_pose_dir, mode)
    pose_out = os.path.join(out_pose_dir, mode)

    tasks = []
    for idx in info:
        tasks.append((
            info[idx],
            pose_in,
            pose_out,
            target_v
        ))

    pool = mp.Pool(mp.cpu_count())
    ok_cnt = 0
    fail_cnt = 0

    for ok, msg in tqdm(pool.imap(process_one, tasks), total=len(tasks)):
        if ok:
            ok_cnt += 1
        else:
            if fail_cnt < 5:  # 只打印前5个错误，避免刷屏
                print(f"\nError: {msg}")
            fail_cnt += 1
    pool.close()
    pool.join()

    print(f"[{mode}] Done. Total={len(tasks)}, OK={ok_cnt}, Failed={fail_cnt}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="phoenix2014")
    parser.add_argument("--modes", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--in_pose_dir", type=str, required=True)
    parser.add_argument("--out_pose_dir", type=str, required=True)
    parser.add_argument("--target_v", type=int, default=75)
    args = parser.parse_args()

    print("=== Pose Alignment with Fallback Matching ===")
    for mode in args.modes:
        print(f"\n=== Processing mode: {mode} ===")
        process_mode(args.prefix, args.dataset, mode,
                     args.in_pose_dir, args.out_pose_dir, args.target_v)

if __name__ == "__main__":
    main()
