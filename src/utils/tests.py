import torch
import os
from torch.utils.data import DataLoader

from src.datasets.asvspoof import ASVSpoofDataset
from src.datasets.collate import collate_fn
from src.model.lcnn import LCNNModel4


train_ds = ASVSpoofDataset("data/processed/train", "data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt", "train")
eval_ds = ASVSpoofDataset("data/processed/eval", "data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt", "eval")
dev_ds = ASVSpoofDataset("data/processed/dev", "data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt", "dev")

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
eval_loader = DataLoader(eval_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)

batch = next(iter(train_loader))
# print(batch["features"].shape, batch["labels"].shape)

model = LCNNModel4(
    input_channels=1,
    channels_list=[96,192,384,256],
    kernel_sizes=[9,5,5,4],
    steps=[1,1,1,1],
    kernel_pool=2,
    step_pool=2,
    dropout=0.3,
    FLayer_size=512,
    n_classes=2
)

dummy = torch.randn(8, 1, 257, 528)

out = model(dummy)
print(out.shape)