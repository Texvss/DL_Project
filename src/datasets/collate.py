import torch

def collate_fn(batch: list[dict]) -> dict:
    times = [item["features"].shape[-1] for item in batch]
    max_time = max(times)

    padded = []
    for item in batch:
        f = item["features"]
        pad_amt = max_time - f.shape[-1]
        if pad_amt > 0:
            f = torch.nn.functional.pad(f, (0, pad_amt))
        padded.append(f)
    features_batch = torch.stack(padded, dim=0)

    out = {"features": features_batch}

    if "labels" in batch[0]:
        out["labels"] = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    if "utt_id" in batch[0]:
        out["utt_id"] = [item["utt_id"] for item in batch]

    return out
