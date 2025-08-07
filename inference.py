import os
import csv
import torch
from torch.utils.data import DataLoader
from src.datasets.asvspoof import ASVSpoofDataset
from src.datasets.collate import collate_fn
from src.model.lcnn import LCNN
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device(cfg.device if cfg.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    model = hydra.utils.instantiate(cfg.model, _recursive_=False).to(device)
    model.load_state_dict(torch.load(cfg.checkpoint_path, map_location=device))
    model.eval()

    eval_ds = hydra.utils.instantiate(
        cfg.dataset.eval,
        _recursive_=False
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    out_csv = os.path.join(cfg.output_dir, cfg.output_file)
    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        for batch in eval_loader:
            feats = batch["features"].to(device)
            utts = batch["utt_id"]
            with torch.no_grad():
                logits = model(feats)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            for u, s in zip(utts, probs):
                wr.writerow([u, f"{s:.6f}"])

    print(f"Predictions saved to {out_csv}")

if __name__ == '__main__':
    main()