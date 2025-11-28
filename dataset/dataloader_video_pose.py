import os
import cv2
import glob
import numpy as np
import torch
import warnings
import torch.utils.data as data
from utils import video_augmentation

warnings.simplefilter(action='ignore', category=FutureWarning)

class VideoPoseDataset(data.Dataset):
    def __init__(
        self,
        prefix,
        gloss_dict,
        pose_root="./data/phoenix2014/pose_features_aligned",
        dataset="phoenix2014",
        mode="train",
        input_size=224,
        frame_interval=1,
        transform_mode=True,
    ):
        self.prefix = prefix
        self.dataset = dataset
        self.mode = mode
        self.pose_root = os.path.join(pose_root, mode)
        self.gloss_dict = gloss_dict
        self.frame_interval = frame_interval
        self.input_size = input_size

        # Load info list
        self.info_list = np.load(
            f"./preprocess/{dataset}/{mode}_info.npy", 
            allow_pickle=True
        ).item()
        
        # Clean up list structure if needed (handle numerical keys)
        self.info_list = [v for k, v in self.info_list.items() if isinstance(k, int)]
        
        print(f"[VideoPoseDataset:{mode}] Loaded {len(self.info_list)} samples")

        # Transforms
        self.transform_mode = "train" if transform_mode else "test"
        self.data_aug = self._build_transform()

    def __len__(self):
        return len(self.info_list)

    def _build_transform(self):
        if self.transform_mode == "train":
            print("Apply training transforms (RGB only)...")
            # 注意：RandomCrop 和 Flip 仅作用于 RGB。
            # Pose 坐标不会随之变换，这在某些情况下是可以接受的（关注局部特征），
            # 但如果追求严格对齐，应关闭 Flip 或手动对齐 Pose。
            return video_augmentation.Compose([
                video_augmentation.RandomCrop(self.input_size),
                video_augmentation.RandomHorizontalFlip(0.5), # 警告：这会导致 RGB 是右手，Pose 还是左手
                video_augmentation.Resize(1.0),
                video_augmentation.ToTensor(),
                video_augmentation.TemporalRescale(0.2, self.frame_interval),
            ])
        else:
            print("Apply testing transforms...")
            return video_augmentation.Compose([
                video_augmentation.CenterCrop(self.input_size),
                video_augmentation.Resize(1.0),
                video_augmentation.ToTensor(),
            ])

    def _read_rgb(self, fi):
        # 兼容原来的路径逻辑
        if 'phoenix' in self.dataset:
            # AdaptSign 默认路径结构
            folder = os.path.join(self.prefix, "features/fullFrame-256x256px/" + fi['folder'])
            img_list = sorted(glob.glob(folder + "/*.png")) # Phoenix 通常是 png
        elif self.dataset == 'CSL':
            folder = os.path.join(self.prefix, "features/fullFrame-256x256px/" + fi['folder'])
            img_list = sorted(glob.glob(folder + "/*.jpg"))
        elif self.dataset == 'CSL-Daily':
            folder = os.path.join(self.prefix, fi['folder'])
            img_list = sorted(glob.glob(folder)) # CSL-Daily info 中通常已包含 *.jpg
        else:
            # Fallback
            folder = os.path.join(self.prefix, fi['folder'])
            img_list = sorted(glob.glob(folder + "/*"))

        # [重要] 视频抽帧
        img_list = img_list[::self.frame_interval]

        video = [
            cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            for path in img_list
        ]
        return video

    def _read_pose(self, fi, target_length):
        fileid = fi["fileid"]
        path = os.path.join(self.pose_root, fileid + ".npz")
        
        if not os.path.exists(path):
            # Fallback: 全 0
            # 使用 target_length 确保与 RGB 长度一致
            pose = np.zeros((target_length, 75, 6), dtype=np.float32)
            mask = np.zeros((target_length, 75), dtype=np.float32)
            return pose, mask

        try:
            data = np.load(path, allow_pickle=True)
            pose = data["pose"] # (T_full, 75, 6)
            
            # [重要] Pose 也要同步抽帧，否则长度不匹配
            pose = pose[::self.frame_interval]
            
            # 再次校验长度：因为 video_augmentation 可能会进一步丢帧或 rescale
            # 这里我们尽量返回原始抽帧后的长度，由 collate_fn 或 __getitem__ 再次截断
        except Exception:
            pose = np.zeros((target_length, 75, 6), dtype=np.float32)
        
        T, V, C = pose.shape
        mask = np.ones((T, V), dtype=np.float32) # 1 表示有效
        
        return pose, mask

    def __getitem__(self, index):
        fi = self.info_list[index]

        # 1. Read RGB
        raw_video = self._read_rgb(fi)
        
        # 2. Transform RGB
        # video_augmentation 可能会改变帧数 (TemporalRescale)
        video, _ = self.data_aug(raw_video, [], fi["fileid"]) 
        video = video.float() / 127.5 - 1
        
        rgb_T = video.shape[0]

        # 3. Read Pose
        # 先读取，期望长度是 rgb_T (大概)
        pose, pose_mask = self._read_pose(fi, rgb_T)
        
        # 4. 强制对齐长度 (Truncate or Pad)
        # 因为 TemporalRescale 是随机的，Pose 很难完美同步，这里做硬性对齐
        pose_T = pose.shape[0]
        if pose_T > rgb_T:
            pose = pose[:rgb_T]
            pose_mask = pose_mask[:rgb_T]
        elif pose_T < rgb_T:
            # Pad with zeros if pose is shorter
            pad_len = rgb_T - pose_T
            pose_pad = np.zeros((pad_len, 75, 6), dtype=np.float32)
            mask_pad = np.zeros((pad_len, 75), dtype=np.float32)
            pose = np.concatenate([pose, pose_pad], axis=0)
            pose_mask = np.concatenate([pose_mask, mask_pad], axis=0)
            
        # 5. Convert to Tensor & Permute
        pose = torch.from_numpy(pose).float()          # (T, 75, 6)
        pose_mask = torch.from_numpy(pose_mask).float() # (T, 75)
        
        # Permute to (C, T, V) for PoseEncoder
        # 注意：这里我们输出 (C, T, V)，这符合常见习惯
        # 如果你的 slr_network 里写的是接收 (B, T, V, C)，请在这里改为不 permute，或者在 network 里 permute
        # 按照之前给你的 network 代码，我们在 forward 里加了自动 permute，所以这里输出什么都可以兼容。
        # 但为了规范，输出 (C, T, V) 比较好。
        pose = pose.permute(2, 0, 1) # (6, T, 75)

        # 6. Label
        label_list = []
        for token in fi["label"].split(" "):
            if token in self.gloss_dict:
                label_list.append(self.gloss_dict[token][0])
        label = torch.LongTensor(label_list)

        return video, video.shape[0], pose, pose_mask, label, len(label), fi["fileid"]

    @staticmethod
    def collate_fn(batch):
        # Sort by video length (descending)
        batch = sorted(batch, key=lambda x: x[1], reverse=True)
        
        videos, v_lens, poses, pose_masks, labels, lab_lens, ids = zip(*batch)
        
        max_T = v_lens[0]
        
        # --- Pad Video ---
        padded_video = []
        for vid in videos:
            T = vid.shape[0]
            if T < max_T:
                # Video padding: usually repeat last frame
                pad = vid[-1:].expand(max_T - T, -1, -1, -1)
                vid = torch.cat([vid, pad], dim=0)
            padded_video.append(vid)
        padded_video = torch.stack(padded_video) # (B, T, C, H, W)
        
        # --- Pad Pose (6, T, V) ---
        padded_pose = []
        padded_mask = []
        for p, m, curT in zip(poses, pose_masks, v_lens):
            # p: (6, T, 75), m: (T, 75)
            if curT < max_T:
                # Pose padding: zero pad
                pad_len = max_T - curT
                
                # Pad pose with 0
                pad_p = torch.zeros((6, pad_len, 75), dtype=p.dtype)
                p = torch.cat([p, pad_p], dim=1)
                
                # [重要] Pad mask with 0 (Invalid)
                pad_m = torch.zeros((pad_len, 75), dtype=m.dtype)
                m = torch.cat([m, pad_m], dim=0)
                
            padded_pose.append(p)
            padded_mask.append(m)
            
        padded_pose = torch.stack(padded_pose) # (B, 6, T, 75)
        padded_mask = torch.stack(padded_mask) # (B, T, 75)
        
        # --- Pad Label ---
        max_L = max(lab_lens)
        padded_labels = []
        for lab in labels:
            if len(lab) < max_L:
                pad = torch.zeros(max_L - len(lab), dtype=torch.long)
                lab = torch.cat([lab, pad], dim=0)
            padded_labels.append(lab)
        padded_labels = torch.stack(padded_labels)
        
        return padded_video, torch.LongTensor(v_lens), padded_pose, padded_mask, padded_labels, torch.LongTensor(lab_lens), ids