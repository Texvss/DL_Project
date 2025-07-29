import logging
import random
from typing import List
import glob
import os

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

class ASVSpoofDataset(Dataset):
    def __init__(self, processed_dir, protocol_path, mode):
        all_paths = sorted(glob.glob(os.path.join(processed_dir, "*.npy")))
        
        with open(protocol_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            mapping = {}
            for line in lines:
                parts = line.strip().split()
                utt_id = parts[1]
                label_str = parts[-1]
                if label_str == "bonafide":
                    label = 0
                else:
                    label = 1
                mapping[utt_id] = label
        self.items = []
        self.mode = mode
        for path in all_paths:
            utt_id = os.path.basename(path).rsplit(".", 1)[0]
            if utt_id in mapping:
                self.items.append((path, mapping[utt_id]))
            else:
                continue

    
    def __getitem__(self, index):
        path, value = self.items[index]
        features_raw = np.load(path)
        features = torch.from_numpy(features_raw).float()
        if features.ndim == 2:
            features = features.unsqueeze(0)
        return {"features": features, "labels": value}
        
    def __len__(self):
        return len(self.items)
