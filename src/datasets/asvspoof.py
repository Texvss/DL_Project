import os
import numpy as np
from torch.utils.data import Dataset

class ASVSpoofDataset(Dataset):
    def __init__(self, data_dir, protocol_path, split):
        self.data_dir = data_dir
        self.items = []
        with open(protocol_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                utt_id = parts[1]
                label = parts[-1]
                if split == 'train' and 'trn' in protocol_path:
                    if label in ['bonafide', 'spoof']:
                        self.items.append((
                            os.path.join(data_dir, f"{utt_id}.npy"),
                            1 if label == 'bonafide' else 0
                        ))
                elif split == 'dev' and 'dev' in protocol_path:
                    if label in ['bonafide', 'spoof']:
                        self.items.append((
                            os.path.join(data_dir, f"{utt_id}.npy"),
                            1 if label == 'bonafide' else 0
                        ))
                elif split == 'eval' and 'eval' in protocol_path:
                    if label in ['bonafide', 'spoof']:
                        self.items.append((
                            os.path.join(data_dir, f"{utt_id}.npy"),
                            1 if label == 'bonafide' else 0
                        ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        features = np.load(path)
        utt_id = os.path.basename(path).replace('.npy', '')
        return {
            'features': features,
            'labels':   label,
            'utt_id':   utt_id
        }