import glob
import os
import numpy as np
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset

class ASVSpoofDataset(Dataset):
    TARGET_T = 600

    def __init__(self, processed_dir: str, protocol_path: str, mode: str, transform=None):
        self.mode = mode
        self.transform = transform
        if mode == "train" and transform is None:
            self.transform = torch.nn.Sequential(
                T.FrequencyMasking(freq_mask_param=30),
                T.TimeMasking(time_mask_param=50)
            )
        all_paths = sorted(glob.glob(os.path.join(processed_dir, "*.npy")))
        if not all_paths:
            raise FileNotFoundError(f"No .npy files found in {processed_dir}")
        mapping = {}
        if mode in ("train", "dev"):
            if not os.path.exists(protocol_path):
                raise FileNotFoundError(f"Protocol file {protocol_path} not found")
            with open(protocol_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    mapping[parts[1]] = 1 if parts[-1] == "bonafide" else 0
        self.items = []
        for path in all_paths:
            utt_id = os.path.basename(path).rsplit(".", 1)[0]
            if mode in ("train", "dev"):
                if utt_id in mapping:
                    self.items.append((path, mapping[utt_id]))
                else:
                    print(f"Warning: {utt_id} not found in protocol {protocol_path}")
            else:
                self.items.append((path, utt_id))
        if not self.items:
            raise ValueError(f"No valid items found for {mode} dataset in {processed_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, val = self.items[index]
        try:
            arr = np.load(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            raise
        feats = torch.from_numpy(arr).float()
        if feats.ndim == 2:
            feats = feats.unsqueeze(0)
        if feats.shape != (1, 257, self.TARGET_T):
            print(f"Warning: Unexpected shape {feats.shape} for {path}")
        _, _, t = feats.shape
        if t < self.TARGET_T:
            pad_amt = self.TARGET_T - t
            feats = torch.nn.functional.pad(feats, (0, pad_amt))
        else:
            feats = feats[:, :, :self.TARGET_T]
        if self.mode == "train" and self.transform:
            feats = self.transform(feats)
        if self.mode in ("train", "dev"):
            return {"features": feats, "labels": val}
        return {"features": feats, "utt_id": val}