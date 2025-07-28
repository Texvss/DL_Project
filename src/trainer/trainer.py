import os
import torch
import numpy as np
import wandb

from torch.utils.data import DataLoader
from src.datasets.asvspoof       import ASVSpoofDataset
from src.datasets.collate        import collate_fn
from src.model.lcnn              import LCNNModel4
from src.metrics.calculate_eer   import compute_eer

class Trainer:

    def __init__(
        self,
        train_config: dict,
        model_config: dict,
        run_config: dict
    ):
        wandb.init(
            project=run_config.get("wandb_project", "asvspoof-lcnn4"),
            config={**train_config, **model_config, **run_config}
        )
        self.config = wandb.config

        self.device = torch.device(self.config.device)

        self.model = LCNNModel4(
            input_channels=self.config.input_channels,
            channels_list=self.config.channels_list,
            kernel_sizes=self.config.kernel_sizes,
            steps=self.config.steps,
            kernel_pool=self.config.kernel_pool,
            step_pool=self.config.step_pool,
            dropout=self.config.dropout,
            FLayer_size=self.config.FLayer_size,
            n_classes=self.config.n_classes
        ).to(self.device)

        self.loss      = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr
        )

        bs = self.config.batch_size
        self.train_loader = DataLoader(
            ASVSpoofDataset(
                self.config.train_dir,
                self.config.train_protocol,
                'train'
            ),
            batch_size=bs, shuffle=True,
            collate_fn=collate_fn, pin_memory=True
        )
        self.dev_loader = DataLoader(
            ASVSpoofDataset(
                self.config.dev_dir,
                self.config.dev_protocol,
                'dev'
            ),
            batch_size=bs, shuffle=False,
            collate_fn=collate_fn, pin_memory=True
        )
        self.eval_loader = DataLoader(
            ASVSpoofDataset(
                self.config.eval_dir,
                self.config.eval_protocol,
                'eval'
            ),
            batch_size=bs, shuffle=False,
            collate_fn=collate_fn, pin_memory=True
        )

        self.epochs          = self.config.epochs
        self.checkpoint_path = self.config.checkpoint_path

    def train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            feats  = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(feats)
            loss   = self.loss(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * feats.size(0)

        avg_loss = total_loss / len(self.train_loader.dataset)
        wandb.log({"train_loss": avg_loss})
        return avg_loss

    def validate(self, loader: DataLoader, split: str) -> tuple[float, float]:
        self.model.eval()
        all_labels, all_scores = [], []
        with torch.no_grad():
            for batch in loader:
                feats  = batch['features'].to(self.device)
                labels = batch['labels'].cpu().numpy()

                logits = self.model(feats).cpu().numpy()
                scores = logits[:, 1]

                all_labels.append(labels)
                all_scores.append(scores)

        y_true   = np.concatenate(all_labels)
        y_scores = np.concatenate(all_scores)
        bona   = y_scores[y_true == 0]
        spoof  = y_scores[y_true == 1]
        eer, thr = compute_eer(bona, spoof)

        wandb.log({f"{split}_EER": eer})
        return eer, thr

    def run(self):
        best_eer = 1.0
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_one_epoch()

            dev_eer, _ = self.validate(self.dev_loader, split="dev")
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Dev EER={dev_eer*100:.2f}%")

            if dev_eer < best_eer:
                best_eer = dev_eer
                torch.save(self.model.state_dict(), self.checkpoint_path)
                wandb.log({"best_dev_EER": best_eer})
                print(f"New best model saved (EER={best_eer*100:.2f}%)")

        print("Evaluating on eval set…")
        eval_eer, _ = self.validate(self.eval_loader, split="eval")
        print(f"Eval EER={eval_eer*100:.2f}%")

        wandb.log({"eval_EER": eval_eer})
        wandb.finish()


if __name__ == '__main__':
    train_cfg = {
        'train_dir':      'data/processed/train',
        'train_protocol':'data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt',
        'dev_dir':       'data/processed/dev',
        'dev_protocol': 'data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt',
        'eval_dir':      'data/processed/eval',
        'eval_protocol':'data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt',
        'batch_size':     16
    }
    model_cfg = {
        'input_channels':1, 'channels_list':[96,192,384,256],
        'kernel_sizes':[9,5,5,4],    'steps':[1,1,1,1],
        'kernel_pool':2, 'step_pool':2,
        'dropout':0.3,   'FLayer_size':512,  'n_classes':2
    }
    run_cfg = {
        'lr':1e-3, 'epochs':20,
        'checkpoint_path':'best_lcnn4.pt',
        'device':'cuda',
        'wandb_project':"asvspoof-lcnn4"
    }

    trainer = Trainer(train_cfg, model_cfg, run_cfg)
    trainer.run()
