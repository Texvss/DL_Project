import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from src.datasets.asvspoof import ASVSpoofDataset
from src.datasets.collate import collate_fn
from src.model.lcnn import LCNN
from src.metrics.calculate_eer import compute_eer

full_ds = ASVSpoofDataset(
    processed_dir='data/processed/train',
    protocol_path='data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt',
    mode='train',
    transform=None
)
subset_idxs = list(range(10))
train_ds = Subset(full_ds, subset_idxs)
loader = DataLoader(train_ds, batch_size=5, shuffle=True, collate_fn=collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LCNN(num_classes=2).to(device)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

# 3) Цикл обучения
for epoch in range(1, 101):
    model.train()
    for batch in loader:
        feats  = batch['features'].to(device)
        labels = batch['labels'].to(device)
        opt.zero_grad()
        logits = model(feats)
        loss   = loss_fn(logits, labels)
        loss.backward()
        opt.step()

    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            feats  = batch['features'].to(device)
            labels = batch['labels'].to(device).cpu().numpy()
            probs  = torch.softmax(model(feats), dim=1)[:, 1].cpu().numpy()
            all_scores.append(probs)
            all_labels.append(labels)
    y_scores = np.concatenate(all_scores)
    y_true   = np.concatenate(all_labels)
    bona     = y_scores[y_true == 1]
    spoof    = y_scores[y_true == 0]
    eer, _   = compute_eer(bona, spoof)

    print(f"Epoch {epoch:3d}: overfit EER = {eer*100:5.2f}%")
    if eer < 1e-3:
        print("Модель зафитилась — EER≈0, проблем с архитектурой/лосc-ом нет.")
        break

# def check_protocol(protocol_path):
#     bonafide_count = 0
#     spoof_count = 0
#     with open(protocol_path, "r") as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) < 5:
#                 continue
#             key = parts[-1]
#             if key == "bonafide":
#                 bonafide_count += 1
#             elif key == "spoof":
#                 spoof_count += 1
#     print(f"Protocol {protocol_path}: {bonafide_count} bonafide, {spoof_count} spoof, total {bonafide_count + spoof_count}")

# check_protocol("data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt")
