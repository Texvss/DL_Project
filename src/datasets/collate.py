import torch

def collate_fn(batch: list[dict]):
    times = [item["features"].shape[-1] for item in batch]
    max_time = max(times)

    padded = []
    for item in batch:
        features = item["features"]
        pad_amount = max_time - features.shape[-1]
        if pad_amount > 0:
            features = torch.nn.functional.pad(features, (0, pad_amount), value=0.0)
        padded.append(features)
        features_batch = torch.stack(padded, dim=0)

        result_batch = {"features": features_batch}

        if "labels" in batch[0]:
            labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
            result_batch["labels"] = labels

        if "utt_id" in batch[0]:
            utt_id = [[item["utt_id"] for item in batch]]
            result_batch["utt_id"] = utt_id
    
    return result_batch
