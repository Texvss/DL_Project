import os
import numpy as np
from torch.utils.data import Dataset
import torch

class ASVSpoofDataset(Dataset):
    def __init__(self, data_dir, protocol_path, split):
        self.data_dir = data_dir
        self.items = []
        seen_utt_ids = set()

        protocol_labels = {}
        with open(protocol_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                utt_id = parts[1]
                label = parts[-1]
                if label in ['bonafide', 'spoof']:
                    protocol_labels[utt_id] = 1 if label == 'bonafide' else 0

        for filename in os.listdir(data_dir):
            if not filename.endswith('.npy'):
                continue
            utt_id = filename.replace('.npy', '')
            file_path = os.path.join(data_dir, filename)
            if utt_id in protocol_labels:
                if utt_id not in seen_utt_ids:
                    self.items.append((file_path, protocol_labels[utt_id]))
                    seen_utt_ids.add(utt_id)
            else:
                print(f"Extra file: {utt_id} - Assigning default label (spoof=0)")
                self.items.append((file_path, 0))

        print(f"Loaded {len(self.items)} items for {split}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        features = np.load(path)  # Загружаем как NumPy
        features = torch.from_numpy(features).float()  # Преобразуем в тензор
        utt_id = os.path.basename(path).replace('.npy', '')
        return {
            'features': features,
            'labels': label,
            'utt_id': utt_id
        }

if __name__ == '__main__':
    train_cfg = {
        "train_dir": "/kaggle/input/processed/processed/train",
        "train_protocol": "/kaggle/input/raw-data/ASVspoof2019_LA_cm_protocols/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
        "dev_dir": "/kaggle/input/processed/processed/dev",
        "dev_protocol": "/kaggle/input/raw-data/ASVspoof2019_LA_cm_protocols/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
        "eval_dir": "/kaggle/input/processed/processed/eval",
        "eval_protocol": "/kaggle/input/raw-data/ASVspoof2019_LA_cm_protocols/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
    }
    for split, dir_path, proto_path in [('train', train_cfg['train_dir'], train_cfg['train_protocol']),
                                        ('dev', train_cfg['dev_dir'], train_cfg['dev_protocol']),
                                        ('eval', train_cfg['eval_dir'], train_cfg['eval_protocol'])]:
        ds = ASVSpoofDataset(dir_path, proto_path, split)
        labels = [label for _, label in ds.items]
        counts = np.bincount(labels)
        print(f"{split.capitalize()}: spoof={counts[0]}, bonafide={counts[1]}")