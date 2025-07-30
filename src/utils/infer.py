# scripts/infer.py
import argparse
import os
import csv

import torch
from torch.utils.data import DataLoader
from src.datasets.asvspoof    import ASVSpoofDataset
from src.datasets.collate     import collate_fn
from src.model.lcnn           import LCNN

def main():
    p = argparse.ArgumentParser(description="Inference for ASVspoof LCNN-4")
    p.add_argument("--checkpoint",   required=True, help="Путь к best_lcnn4.pt")
    p.add_argument("--npy-dir",      required=True, help="Папка с .npy для eval")
    p.add_argument("--protocol",     required=True, help="eval-протокол txt")
    p.add_argument("--batch-size",   type=int, default=16)
    p.add_argument("--output-csv",   required=True, help="mgergokov@edu.csv")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_ds = ASVSpoofDataset(
        processed_dir = args.npy_dir,
        protocol_path = args.protocol,
        mode = "eval"
    )
    loader = DataLoader(
        eval_ds,
        batch_size = args.batch_size,
        shuffle = False,
        collate_fn = collate_fn,
        pin_memory = True,
        num_workers = 4
    )

    dummy_cfg = {
        "input_channels":1, "channels_list":[96,192,384,256],
        "kernel_sizes":[9,5,5,4], "steps":[1,1,1,1],
        "kernel_pool":2, "step_pool":2,
        "dropout":0.3, "FLayer_size":512, "n_classes":2
    }
    model = LCNN(**dummy_cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    results = []
    with torch.no_grad():
        for batch in loader:
            feats = batch["features"].to(device)
            utts  = batch["utt_id"]
            logits = model(feats)
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            for utt, score in zip(utts, probs):
                results.append((utt, float(score)))

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        for utt, score in results:
            writer.writerow([utt, score])

    print(f"Saved {len(results)} scores to {args.output_csv}")

if __name__ == "__main__":
    main()
