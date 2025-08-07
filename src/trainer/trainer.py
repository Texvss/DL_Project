import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.datasets.asvspoof import ASVSpoofDataset
from src.datasets.collate import collate_fn
from src.model.lcnn import LCNN
from src.metrics.calculate_eer import compute_eer
from sklearn.metrics import accuracy_score
from src.logger.cometml import CometMLWriter
import hydra
from omegaconf import DictConfig, OmegaConf

torch._dynamo.config.suppress_errors = True

class Trainer:
    def __init__(self, cfg: DictConfig):
        self.logger = CometMLWriter(
            project_config=OmegaConf.to_container(cfg),
            project_name=cfg.get("comet_project", "spoofing-detection"),
            workspace=cfg.get("comet_workspace", None),
            run_id=None,
            run_name=f"run_{time.strftime('%Y%m%d_%H%M%S')}",
            mode="online"
        )
        self.logger.log_parameters(OmegaConf.to_container(cfg))
        self.device = torch.device(cfg.device if cfg.device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = hydra.utils.instantiate(cfg.model, _recursive_=False).to(self.device)

        self.checkpoint_dir = "/kaggle/working/checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.checkpoint_dir, cfg.checkpoint_path.split("/")[-1])
        print(f"Checkpoint directory created at: {self.checkpoint_dir}")
        print(f"Model will be saved to: {self.checkpoint_path}")

        # Инициализация DataLoader'ов через hydra
        self.train_loader = hydra.utils.instantiate(
            cfg.dataloader,
            dataset=hydra.utils.instantiate(cfg.dataset.train, _recursive_=False),
            batch_size=cfg.dataloader.batch_size,
            sampler=self._create_weighted_sampler(cfg.dataset.train),
            collate_fn=collate_fn,
            pin_memory=cfg.dataloader.pin_memory,
            num_workers=cfg.dataloader.num_workers,
            shuffle=cfg.shuffle
        )
        self.dev_loader = hydra.utils.instantiate(
            cfg.dataloader,
            dataset=hydra.utils.instantiate(cfg.dataset.dev, _recursive_=False),
            batch_size=cfg.dataloader.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=cfg.dataloader.pin_memory,
            num_workers=cfg.dataloader.num_workers
        )
        self.eval_loader = hydra.utils.instantiate(
            cfg.dataloader,
            dataset=hydra.utils.instantiate(cfg.dataset.eval, _recursive_=False),
            batch_size=cfg.dataloader.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=cfg.dataloader.pin_memory,
            num_workers=cfg.dataloader.num_workers
        )

        self.eval_label_map = {}
        with open(cfg.dataset.eval.protocol, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                utt, lbl = parts[1], parts[-1]
                self.eval_label_map[utt] = 1 if lbl == 'bonafide' else 0

        class_weights = self._compute_class_weights(cfg.dataset.train)
        print(f"Class weights: spoof={class_weights[0]:.8f}, bonafide={class_weights[1]:.8f}")
        self.loss = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.T_max)
        self.epochs = cfg.epochs
        self.grad_clip_norm = cfg.grad_clip_norm

    def _compute_class_weights(self, cfg):
        ds = ASVSpoofDataset(cfg.dir, cfg.protocol, 'train')
        labels = [label for _, label in ds.items]
        counts = np.bincount(labels)
        weights = 1.0 / counts
        weights[1] *= 3.0
        print(f"Raw counts: spoof={counts[0]}, bonafide={counts[1]}")
        return torch.tensor(weights, dtype=torch.float32)

    def _create_weighted_sampler(self, cfg):
        ds = ASVSpoofDataset(cfg.dir, cfg.protocol, 'train')
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
        self.logger.set_step(epoch, mode="train")
        for batch_idx, batch in enumerate(self.train_loader, 1):
            feats = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(feats)
            loss = self.loss(logits, labels)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
            self.optimizer.step()
            total_loss += loss.item() * feats.size(0)
            if batch_idx % 10 == 0 or batch_idx == num_batches:
                print(f"  [Epoch {epoch}] batch {batch_idx}/{num_batches}  loss = {loss.item():.4f}")
        avg_loss = total_loss / len(self.train_loader.dataset)
        epoch_time = time.time() - start_time
        print(f"[Epoch {epoch}] finished — avg train loss = {avg_loss:.4f}, epoch_time = {epoch_time:.2f}s\n")
        self.logger.add_scalar("train_loss", avg_loss)
        self.logger.add_scalar("grad_norm", grad_norm.item())
        self.logger.add_scalar("epoch_time", epoch_time)
        return avg_loss

    def validate(self, loader, split: str, epoch: int) -> float:
        self.model.eval()
        all_labels, all_scores = [], []
        self.logger.set_step(epoch, mode=split)
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
        self.logger.add_scalar(f"{split}_EER", eer * 100)
        self.logger.add_scalar(f"{split}_accuracy", accuracy * 100)
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
                self.logger.add_checkpoint(self.checkpoint_path, self.checkpoint_dir)
            else:
                no_improve += 1
            if self.scheduler:
                self.scheduler.step()
            # if no_improve >= 5:
            #     print("Early stopping")
            #     break
        final_eer = self.validate(self.eval_loader, 'eval', self.epochs)
        print(f'Final eval_EER={final_eer*100:.2f}%')

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    trainer = Trainer(cfg)
    trainer.run()

if __name__ == '__main__':
    main()
