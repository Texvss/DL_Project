import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

class ASVSpoofDataset(Dataset):
    TARGET_T = 600

    def __init__(self, processed_dir: str, protocol_path: str, mode: str, transform=None):
        self.mode = mode
        self.transform = transform
        all_paths = sorted(glob.glob(os.path.join(processed_dir, "*.npy")))

        mapping = {}
        if mode in ("train", "dev"):
            with open(protocol_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    mapping[parts[1]] = 0 if parts[-1] == "bonafide" else 1

        self.items = []
        for path in all_paths:
            utt_id = os.path.basename(path).rsplit(".", 1)[0]
            if mode in ("train", "dev"):
                if utt_id in mapping:
                    self.items.append((path, mapping[utt_id]))
            else:
                self.items.append((path, utt_id))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, val = self.items[index]
        arr = np.load(path)
        feats = torch.from_numpy(arr).float()
        if feats.ndim == 2:
            feats = feats.unsqueeze(0)

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
        else:
            return {"features": feats, "utt_id": val}
