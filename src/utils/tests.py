import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from collections import Counter
from src.utils.preprocess import compute_spectrogram

# from src.datasets.asvspoof import ASVSpoofDataset
# from src.datasets.collate import collate_fn
# from src.model.lcnn import LCNN
# from src.trainer.trainer import Trainer

# train_ds = ASVSpoofDataset("data/processed/train", "data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt", "train")
# eval_ds = ASVSpoofDataset("data/processed/eval", "data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt", "eval")
# dev_ds = ASVSpoofDataset("data/processed/dev", "data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt", "dev")

# train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
# eval_loader = DataLoader(eval_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
# dev_loader = DataLoader(dev_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)

# labels = [lbl for _, lbl in eval_ds.items]
# counts = Counter(labels)
# print("Class distribution:", counts)


# # # shapes = []
# # # for path, _ in train_ds.items[:10]:
# # #     arr = np.load(path)
# # #     print(f"{os.path.basename(path)} → {arr.shape}")

# # # batch = next(iter(train_loader))
# # # # print(batch["features"].shape, batch["labels"].shape)

# # def check_files(processed_dir, protocol_path):
# #     npy_files = set(os.path.basename(f).rsplit(".", 1)[0] for f in glob.glob(os.path.join(processed_dir, "*.npy")))
# #     with open(protocol_path, "r") as f:
# #         protocol_ids = set(line.strip().split()[1] for line in f)
# #     missing_in_npy = protocol_ids - npy_files
# #     extra_in_npy = npy_files - protocol_ids
# #     print(f"Processed {len(npy_files)} .npy files, {len(protocol_ids)} protocol entries")
# #     print(f"Missing in .npy (in protocol but not in processed): {len(missing_in_npy)}")
# #     print(f"Extra in .npy (in processed but not in protocol): {len(extra_in_npy)}")
# #     if missing_in_npy:
# #         print(f"Missing in .npy: {missing_in_npy}")
# #     if extra_in_npy:
# #         print(f"Extra in .npy: {extra_in_npy}")

# # check_files(
# #     "data/processed/dev",
# #     "data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
# # )
# # check_files(
# #     "data/processed/train",
# #     "data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trl.txt"
# # )
# # check_files(
# #     "data/processed/eval",
# #     "data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
# # )

spect = compute_spectrogram("data/raw/ASVspoof2019_LA_dev/flac/LA_D_1000265.flac")
print("shape:", spect.shape)
# ожидаем: (1, n_fft//2+1, T) == (1, 1724//2+1, ≥600)
assert spect.shape[1] == 1724//2+1

_, _, T = spect.shape
print("time frames:", T)  
# assert T == 600


import librosa
y, sr = librosa.load("data/raw/ASVspoof2019_LA_dev/flac/LA_D_1000265.flac", sr=16000)
S = librosa.stft(y, n_fft=1724, hop_length=int(0.0081*sr), win_length=1724, window="blackman", center=False)
logS = 20*np.log10(np.abs(S)+1e-6)
logS = (logS - logS.mean())/logS.std()
# обрежем/pad до 600 кадров:
logS = logS[:, :600] if logS.shape[1]>=600 else np.pad(logS, ((0,0),(0,600-logS.shape[1])))
diff = np.max(np.abs(logS - spect[0]))
print("max abs difference:", diff)
assert diff < 1e-3  # или другой маленький порог
