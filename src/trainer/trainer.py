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
            api_key=run_config['comet_api_key'],
            project_name=run_config.get('comet_project'),
            workspace=run_config.get('comet_workspace')
        )
        self.experiment.log_parameters({**train_config, **run_config})
        self.device = torch.device(run_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = LCNN(num_classes=2).to(self.device)

        train_ds = ASVSpoofDataset(train_config['train_dir'], train_config['train_protocol'], mode='train')
        dev_ds   = ASVSpoofDataset(train_config['dev_dir'],   train_config['dev_protocol'],   mode='dev')
        eval_ds  = ASVSpoofDataset(train_config['eval_dir'],  train_config['eval_protocol'],  mode='eval')

        class_weights = self._compute_class_weights(train_ds)
        sampler       = self._create_weighted_sampler(train_ds)

        bs = train_config['batch_size']
        self.train_loader = DataLoader(train_ds,   batch_size=bs, sampler=sampler,  collate_fn=collate_fn, pin_memory=True, num_workers=4)
        self.dev_loader   = DataLoader(dev_ds,     batch_size=bs, shuffle=False,     collate_fn=collate_fn, pin_memory=True, num_workers=4)
        self.eval_loader  = DataLoader(eval_ds,    batch_size=bs, shuffle=False,     collate_fn=collate_fn, pin_memory=True, num_workers=4)

        self.eval_label_map = {}
        with open(train_config['eval_protocol'], 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                utt = parts[1]
                label = 1 if parts[-1] == 'bonafide' else 0
                self.eval_label_map[utt] = label

        self.loss      = torch.nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=run_config['lr'], weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=run_config.get('T_max', 10))
        self.epochs         = run_config['epochs']
        self.checkpoint_path= run_config['checkpoint_path']
        self.asv_score_file = train_config.get('asv_score_file')

    def _compute_class_weights(self, dataset):
        counts = np.bincount([label for _, label in dataset.items])
        weights = 1.0 / counts
        weights[0] *= 2.0
        return torch.tensor(weights, dtype=torch.float32)

    def _create_weighted_sampler(self, dataset):
        counts = np.bincount([label for _, label in dataset.items])
        w = 1.0 / counts
        sample_weights = [w[label] for _, label in dataset.items]
        return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            feats  = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(feats)
            loss   = self.loss(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            total_loss += loss.item() * feats.size(0)
        avg_loss = total_loss / len(self.train_loader.dataset)
        self.experiment.log_metric('train_loss', avg_loss, step=epoch)
        return avg_loss

    def validate(self, loader, split: str, epoch: int) -> tuple:
        self.model.eval()
        all_labels, all_scores, all_utts = [], [], []
        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                feats = batch['features'].to(self.device)
                if 'labels' in batch:
                    labels = batch['labels'].to(self.device)
                    utts   = batch.get('utt_id', [f'{split}_{i}' for i in range(labels.size(0))])
                else:
                    utts = batch['utt_id']
                    labels = torch.tensor([self.eval_label_map.get(u, 0) for u in utts], device=self.device)
                logits = self.model(feats)
                loss   = self.loss(logits, labels)
                total_loss += loss.item()
                scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_scores.append(scores)
                all_labels.append(labels.cpu().numpy())
                all_utts.extend(utts)
        val_loss = total_loss / len(loader)
        y_true   = np.concatenate(all_labels)
        y_scores = np.concatenate(all_scores)
        bona  = y_scores[y_true == 0]
        spoof = y_scores[y_true == 1]
        eer, _        = compute_eer(bona, spoof)
        self.experiment.log_metric(f'{split}_loss',    val_loss, step=epoch)
        self.experiment.log_metric(f'{split}_EER',     eer,      step=epoch)
        with open('temp_cm_scores.txt', 'w') as f:
            for u, l, s in zip(all_utts, y_true, y_scores):
                key = 'bonafide' if l == 0 else 'spoof'
                f.write(f"{u} - {key} {s}\n")
        _, min_tDCF = calculate_tDCF_EER('temp_cm_scores.txt', self.asv_score_file, 'temp_output.txt', printout=False)
        self.experiment.log_metric(f'{split}_min_tDCF', min_tDCF, step=epoch)
        return eer, min_tDCF

    def run(self):
        best_eer, patience, target = float('inf'), 0, 0.1
        for epoch in range(1, self.epochs + 1):
            self.train_one_epoch(epoch)
            dev_eer, _  = self.validate(self.dev_loader,  'dev',  epoch)
            eval_eer, _ = self.validate(self.eval_loader, 'eval', epoch)
            print(f'Epoch {epoch}: dev_EER={dev_eer*100:.2f}%, eval_EER={eval_eer*100:.2f}%')
            if dev_eer < best_eer:
                best_eer, patience = dev_eer, 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                patience += 1
            self.scheduler.step()
            if best_eer < target or patience >= 5:
                break
        final_eer, _ = self.validate(self.eval_loader, 'eval', epoch)
        print(f'Final Eval EER: {final_eer*100:.2f}%')
        if final_eer > target:
            raise RuntimeError(f"Eval EER {final_eer*100:.2f}% exceeds target")
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