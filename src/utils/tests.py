# overfit_test.py
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from src.datasets.asvspoof import ASVSpoofDataset
from src.datasets.collate import collate_fn
from src.model.lcnn import LCNN
from src.metrics.calculate_eer import compute_eer

# 1) Собираем маленький датасет из 10 примеров без аугментации
full_ds = ASVSpoofDataset(
    processed_dir='data/processed/train',
    protocol_path='data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt',
    mode='train',
    transform=None  # отключаем masking
)
subset_idxs = list(range(10))              # первые 10 файлов
train_ds = Subset(full_ds, subset_idxs)
loader = DataLoader(train_ds, batch_size=5, shuffle=True, collate_fn=collate_fn)

# 2) Модель, оптимизатор, loss
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

    # 4) Проверяем EER на том же самом подмножестве
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
    bona     = y_scores[y_true == 1]  # bonafide=1
    spoof    = y_scores[y_true == 0]
    eer, _   = compute_eer(bona, spoof)

    print(f"Epoch {epoch:3d}: overfit EER = {eer*100:5.2f}%")
    if eer < 1e-3:
        print("Модель зафитилась — EER≈0, проблем с архитектурой/лосc-ом нет.")
        break
