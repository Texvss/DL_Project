import torch
from typing import List, Dict

def collate_fn(batch: List[Dict]) -> Dict:
    TARGET_T = 600
    if not batch:
        raise ValueError("Batch is empty")
    times = [item["features"].shape[-1] for item in batch]
    max_time = min(max(times), TARGET_T)
    padded = []
    for item in batch:
        f = item["features"]
        if f.shape[0] != 1 or f.shape[1] != 257:
            raise ValueError(f"Unexpected feature shape {f.shape} in item {item.get('utt_id', 'unknown')}")
        t = f.shape[-1]
        if t < max_time:
            pad_amt = max_time - t
            f = torch.nn.functional.pad(f, (0, pad_amt))
        elif t > max_time:
            f = f[:, :, :max_time]
        padded.append(f)
    features_batch = torch.stack(padded, dim=0)
    out = {"features": features_batch}
    if "labels" in batch[0]:
        out["labels"] = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    if "utt_id" in batch[0]:
        out["utt_id"] = [item["utt_id"] for item in batch]
    return out