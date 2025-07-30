from comet_ml import Experiment
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.datasets.asvspoof import ASVSpoofDataset
from src.datasets.collate import collate_fn
from src.model.lcnn import LCNN
from src.metrics.calculate_eer import compute_eer, calculate_tDCF_EER

class Trainer:
    def __init__(self, train_config: dict, run_config: dict):
        self.experiment = Experiment(
            api_key=run_config["comet_api_key"],
            project_name=run_config.get("comet_project"),
            workspace=run_config.get("comet_workspace", None)
        )
        self.experiment.log_parameters({**train_config, **run_config})
        # self.device = torch.device(run_config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.device = torch.device(run_config.get("device"))
        self.model = LCNN(num_classes=2).to(self.device)
        class_weights = self._compute_class_weights(train_config)
        self.loss = torch.nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=run_config['lr'], weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10)
        bs = train_config['batch_size']
        train_ds = ASVSpoofDataset(
            train_config['train_dir'],
            train_config['train_protocol'],
            'train'
        )
        sampler = self._create_weighted_sampler(train_ds)
        self.train_loader = DataLoader(
            train_ds,
            batch_size=bs,
            sampler=sampler,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=4
        )
        self.dev_loader = DataLoader(
            ASVSpoofDataset(
                train_config['dev_dir'],
                train_config['dev_protocol'],
                'dev'
            ),
            batch_size=bs,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=4
        )
        self.eval_loader = DataLoader(
            ASVSpoofDataset(
                train_config['eval_dir'],
                train_config['eval_protocol'],
                'eval'
            ),
            batch_size=bs,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=4
        )
        self.epochs = run_config['epochs']
        self.checkpoint_path = run_config['checkpoint_path']
        self.asv_score_file = train_config.get('asv_score_file', 'data/raw/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt')

    def _compute_class_weights(self, train_config):
        ds = ASVSpoofDataset(
            train_config['train_dir'],
            train_config['train_protocol'],
            'train'
        )
        labels = [label for _, label in ds.items]
        counts = np.bincount(labels)
        weights = 1.0 / counts
        weights[0] *= 2.0
        class_weights = torch.tensor(weights, dtype=torch.float32)
        return class_weights

    def _create_weighted_sampler(self, train_ds):
        labels = [label for _, label in train_ds.items]
        counts = np.bincount(labels)
        weights = 1.0 / counts
        sample_weights = [weights[label] for label in labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        return sampler

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        print(f"\n[Epoch {epoch}] training on {num_batches} batches…")
        for batch_idx, batch in enumerate(self.train_loader, 1):
            feats = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)
            label_counts = np.bincount(labels.cpu().numpy(), minlength=2)
            print(f"Batch {batch_idx}: Class distribution = {label_counts}")
            self.optimizer.zero_grad()
            logits = self.model(feats)
            loss = self.loss(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            total_loss += loss.item() * feats.size(0)
            if batch_idx % 10 == 0 or batch_idx == num_batches:
                print(f"  [Epoch {epoch}] batch {batch_idx}/{num_batches}  loss = {loss.item():.4f}")
        avg_loss = total_loss / len(self.train_loader.dataset)
        print(f"[Epoch {epoch}] finished — avg train loss = {avg_loss:.4f}\n")
        self.experiment.log_metric("train_loss", avg_loss, step=epoch)
        return avg_loss

    def validate(self, loader, split: str, epoch: int) -> tuple:
        self.model.eval()
        all_labels, all_scores, all_utts = [], [], []
        total_loss = 0.0
        n_batches = len(loader)
        with torch.no_grad():
            for batch in loader:
                feats = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                utts = batch['utt_id'] if 'utt_id' in batch else [f"utt_{i}" for i in range(feats.size(0))]
                logits = self.model(feats)
                loss = self.loss(logits, labels)
                total_loss += loss.item()
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_scores.append(probs)
                all_labels.append(labels.cpu().numpy())
                all_utts.extend(utts)
        val_loss = total_loss / n_batches
        self.experiment.log_metric(f"{split}_loss", val_loss, step=epoch)
        y_true = np.concatenate(all_labels)
        y_scores = np.concatenate(all_scores)
        bona = y_scores[y_true == 0]
        spoof = y_scores[y_true == 1]
        eer, threshold = compute_eer(bona, spoof)
        self.experiment.log_metric(f"{split}_EER", eer, step=epoch)
        with open("temp_cm_scores.txt", "w") as f:
            for utt, score in zip(all_utts, y_scores):
                key = "bonafide" if y_true[list(all_utts).index(utt)] == 0 else "spoof"
                f.write(f"{utt} - {key} {score}\n")
        eer_cm, min_tDCF = calculate_tDCF_EER(
            "temp_cm_scores.txt",
            self.asv_score_file,
            "temp_output.txt",
            printout=False
        )
        self.experiment.log_metric(f"{split}_min_tDCF", min_tDCF, step=epoch)
        return eer, min_tDCF

    def run(self):
        best_eer = float('inf')
        no_improve = 0
        target_eer = 0.1
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            dev_eer, dev_tDCF = self.validate(self.dev_loader, "dev", epoch)
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, dev_EER={dev_eer*100:.2f}%, dev_tDCF={dev_tDCF:.4f}")
            if dev_eer < best_eer:
                best_eer = dev_eer
                no_improve = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"New best model saved (EER={best_eer*100:.2f}%)")
            else:
                no_improve += 1
            self.scheduler.step()
            if best_eer < target_eer:
                print(f"Target EER<{target_eer*100:.1f}% reached ({best_eer*100:.2f}%).")
                break
            if no_improve >= 5:
                print("Early stopping: no improvement for 5 epochs.")
                break
        eval_eer, eval_tDCF = self.validate(self.eval_loader, "eval", epoch)
        print(f"Final eval_EER={eval_eer*100:.2f}%, eval_tDCF={eval_tDCF:.4f}")
        self.experiment.end()

if __name__ == '__main__':
    train_cfg = {
        'train_dir': 'data/processed/train',
        'train_protocol': 'data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt',
        'dev_dir': 'data/processed/dev',
        'dev_protocol': 'data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt',
        'eval_dir': 'data/processed/eval',
        'eval_protocol': 'data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt',
        'batch_size': 16,
        'asv_score_file': 'data/raw/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt'
    }
    run_cfg = {
        'lr': 1e-3,
        'epochs': 3,
        'checkpoint_path': 'best_lcnn.pt',
        'device': 'cpu',
        'comet_api_key': "YOUR_API_KEY",
        'comet_project': "spoof-recognition-public",
        'comet_workspace': None,
    }
    trainer = Trainer(train_cfg, run_cfg)
    trainer.run()