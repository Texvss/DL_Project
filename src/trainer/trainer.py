import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from comet_ml import Experiment
from src.datasets.asvspoof import ASVSpoofDataset
from src.datasets.collate import collate_fn
from src.model.lcnn import LCNN
from src.metrics.calculate_eer import compute_eer
from sklearn.metrics import accuracy_score

torch._dynamo.config.suppress_errors = True

class Trainer:
    def __init__(self, train_config: dict, run_config: dict):
        self.experiment = Experiment(
            api_key=run_config["comet_api_key"],
            project_name=run_config.get("comet_project"),
            workspace=run_config.get("comet_workspace", None)
        )
        self.experiment.log_parameters({**train_config, **run_config})
        self.device = torch.device(
            run_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = LCNN(num_classes=2).to(self.device)

        checkpoint_dir = "/kaggle/working/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(checkpoint_dir, "best_lcnn.pt")
        print(f"Checkpoint directory created at: {checkpoint_dir}")
        print(f"Model will be saved to: {self.checkpoint_path}")

        sampler = self._create_weighted_sampler(train_config)
        bs = train_config['batch_size']
        num_workers = run_config.get('num_workers', 4)
        self.train_loader = DataLoader(
            ASVSpoofDataset(train_config['train_dir'], train_config['train_protocol'], 'train'),
            batch_size=bs,
            sampler=sampler,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=num_workers
        )
        self.dev_loader = DataLoader(
            ASVSpoofDataset(train_config['dev_dir'], train_config['dev_protocol'], 'dev'),
            batch_size=bs,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=num_workers
        )
        self.eval_loader = DataLoader(
            ASVSpoofDataset(train_config['eval_dir'], train_config['eval_protocol'], 'eval'),
            batch_size=bs,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=num_workers
        )

        self.eval_label_map = {}
        with open(train_config['eval_protocol'], 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                utt, lbl = parts[1], parts[-1]
                self.eval_label_map[utt] = 1 if lbl == 'bonafide' else 0

        class_weights = self._compute_class_weights(train_config)
        print(f"Class weights: spoof={class_weights[0]:.8f}, bonafide={class_weights[1]:.8f}")
        self.loss = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=run_config['lr'], weight_decay=1e-1)  # Увеличен weight_decay
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=run_config.get('T_max', 10))  # Включен с T_max=10
        self.epochs = run_config['epochs']

    def _compute_class_weights(self, cfg):
        ds = ASVSpoofDataset(cfg['train_dir'], cfg['train_protocol'], 'train')
        labels = [label for _, label in ds.items]
        counts = np.bincount(labels)
        weights = 1.0 / counts
        weights[1] *= 3.0
        print(f"Raw counts: spoof={counts[0]}, bonafide={counts[1]}")
        return torch.tensor(weights, dtype=torch.float32)

    def _create_weighted_sampler(self, cfg):
        ds = ASVSpoofDataset(cfg['train_dir'], cfg['train_protocol'], 'train')
        labels = [label for _, label in ds.items]
        counts = np.bincount(labels)
        w = 1.0 / counts
        sample_weights = [w[label] for label in labels]
        return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    def train_one_epoch(self, epoch: int) -> float:
        start_time = time.time()
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        print(f"\n[Epoch {epoch}] training on {num_batches} batches…")
        for batch_idx, batch in enumerate(self.train_loader, 1):
            feats = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(feats)
            loss = self.loss(logits, labels)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            total_loss += loss.item() * feats.size(0)
            if batch_idx % 10 == 0 or batch_idx == num_batches:
                print(f"  [Epoch {epoch}] batch {batch_idx}/{num_batches}  loss = {loss.item():.4f}")
        avg_loss = total_loss / len(self.train_loader.dataset)
        epoch_time = time.time() - start_time
        print(f"[Epoch {epoch}] finished — avg train loss = {avg_loss:.4f}, epoch_time = {epoch_time:.2f}s\n")
        self.experiment.log_metric("train_loss", avg_loss, step=epoch)
        self.experiment.log_metric("grad_norm", grad_norm, step=epoch)
        self.experiment.log_metric("epoch_time", epoch_time, step=epoch)
        return avg_loss

    def validate(self, loader, split: str, epoch: int) -> float:
        self.model.eval()
        all_labels, all_scores = [], []
        with torch.no_grad():
            for batch in loader:
                feats = batch['features'].to(self.device)
                if 'labels' in batch:
                    labels = batch['labels'].to(self.device)
                else:
                    utts = batch['utt_id']
                    labels = torch.tensor([
                        self.eval_label_map.get(u, 0) for u in utts
                    ], dtype=torch.long, device=self.device)
                logits = self.model(feats)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_scores.append(probs)
                all_labels.append(labels.cpu().numpy())
        y_true = np.concatenate(all_labels)
        y_scores = np.concatenate(all_scores)
        bona = y_scores[y_true == 1]
        spoof = y_scores[y_true == 0]
        eer, _ = compute_eer(bona, spoof)
        accuracy = accuracy_score(y_true, (y_scores > 0.5).astype(int))
        self.experiment.log_metric(f'{split}_EER', eer * 100, step=epoch)
        self.experiment.log_metric(f'{split}_accuracy', accuracy * 100, step=epoch)
        return eer

    def run(self):
        best_eer, no_improve = float('inf'), 0
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            dev_eer = self.validate(self.dev_loader, 'dev', epoch)
            eval_eer = self.validate(self.eval_loader, 'eval', epoch)
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                f"dev_EER={dev_eer*100:.2f}%, eval_EER={eval_eer*100:.2f}%"
            )
            if eval_eer < best_eer:
                best_eer, no_improve = eval_eer, 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"Saved model to {self.checkpoint_path} with eval_EER={eval_eer*100:.2f}%")
            else:
                no_improve += 1
            if self.scheduler:
                self.scheduler.step()
            # if no_improve >= 5:
            #     print("Early stopping")
            #     break
        final_eer = self.validate(self.eval_loader, 'eval', self.epochs)
        print(f'Final eval_EER={final_eer*100:.2f}%')

if __name__ == '__main__':
    train_cfg = {
        "train_dir":      "/kaggle/input/processed/processed/train",
        "train_protocol": "/kaggle/input/raw-data/ASVspoof2019_LA_cm_protocols/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
        "dev_dir":        "/kaggle/input/processed/processed/dev",
        "dev_protocol":   "/kaggle/input/raw-data/ASVspoof2019_LA_cm_protocols/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
        "eval_dir":       "/kaggle/input/processed/processed/eval",
        "eval_protocol":  "/kaggle/input/raw-data/ASVspoof2019_LA_cm_protocols/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
        "batch_size": 64
    }

    run_cfg = {
        "lr": 5e-5,
        "epochs": 10,
        "checkpoint_path": "/kaggle/working/checkpoints/best_lcnn.pt",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "comet_api_key": os.environ["COMET_API_KEY"],
        "comet_project": "spoofing-detection",
        "comet_workspace": None,
        "num_workers": 4,
        "T_max": 10
    }
    trainer = Trainer(train_cfg, run_config=run_cfg)
    trainer.run()