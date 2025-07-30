import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from collections import Counter

from src.datasets.asvspoof import ASVSpoofDataset
from src.datasets.collate import collate_fn
# from src.model.lcnn import LCNN
# from src.trainer.trainer import Trainer

train_ds = ASVSpoofDataset("data/processed/train", "data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt", "train")
eval_ds = ASVSpoofDataset("data/processed/eval", "data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt", "eval")
dev_ds = ASVSpoofDataset("data/processed/dev", "data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt", "dev")

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
eval_loader = DataLoader(eval_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)

labels = [lbl for _, lbl in train_ds.items]
counts = Counter(labels)
print("Class distribution:", counts)


shapes = []
for path, _ in train_ds.items[:10]:
    arr = np.load(path)
    print(f"{os.path.basename(path)} → {arr.shape}")

batch = next(iter(train_loader))
# print(batch["features"].shape, batch["labels"].shape) 