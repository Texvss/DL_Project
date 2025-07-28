import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from comet_ml import Experiment

from src.datasets.asvspoof     import ASVSpoofDataset
from src.datasets.collate      import collate_fn
from src.model.lcnn            import LCNNModel4
from src.metrics.calculate_eer import compute_eer


class Trainer:
    def __init__(self, train_config: dict, model_config: dict, run_config: dict):
        self.experiment = Experiment(
            api_key      = run_config["comet_api_key"],
            project_name = run_config.get("comet_project"),
            workspace    = run_config.get("comet_workspace", None)
        )
        self.experiment.log_parameters({**train_config, **model_config, **run_config})

        self.device = torch.device(
            run_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = LCNNModel4(
            input_channels=model_config['input_channels'],
            channels_list = model_config['channels_list'],
            kernel_sizes  = model_config['kernel_sizes'],
            steps         = model_config['steps'],
            kernel_pool   = model_config['kernel_pool'],
            step_pool     = model_config['step_pool'],
            dropout       = model_config['dropout'],
            FLayer_size   = model_config['FLayer_size'],
            n_classes     = model_config['n_classes']
        ).to(self.device)

        self.loss      = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=run_config['lr'])

        bs = train_config['batch_size']
        self.train_loader = DataLoader(
            ASVSpoofDataset(
                train_config['train_dir'],
                train_config['train_protocol'],
                'train'
            ),
            batch_size=bs, shuffle=True,  collate_fn=collate_fn,
            pin_memory=True, num_workers=4
        )
        self.dev_loader = DataLoader(
            ASVSpoofDataset(
                train_config['dev_dir'],
                train_config['dev_protocol'],
                'dev'
            ),
            batch_size=bs, shuffle=False, collate_fn=collate_fn,
            pin_memory=True, num_workers=4
        )
        self.eval_loader = DataLoader(
            ASVSpoofDataset(
                train_config['eval_dir'],
                train_config['eval_protocol'],
                'eval'
            ),
            batch_size=bs, shuffle=False, collate_fn=collate_fn,
            pin_memory=True, num_workers=4
        )

        self.epochs          = run_config['epochs']
        self.checkpoint_path = run_config['checkpoint_path']


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
        self.experiment.log_metric("train_loss", avg_loss, step=epoch)
        return avg_loss


    def validate(self, loader: DataLoader, split: str, epoch: int) -> float:
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
        eer, _ = compute_eer(bona, spoof)

        self.experiment.log_metric(f"{split}_EER", eer, step=epoch)
        return eer


    def run(self):
        best_eer = float('inf')

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            dev_eer    = self.validate(self.dev_loader, "dev", epoch)

            print(f"Epoch {epoch:2d} → train_loss={train_loss:.4f}, dev_EER={dev_eer*100:.2f}%")

            if dev_eer < best_eer:
                best_eer = dev_eer
                torch.save(self.model.state_dict(), self.checkpoint_path)
                self.experiment.log_model("best_lcnn4", self.checkpoint_path)
                print(f"  New best model saved (EER={best_eer*100:.2f}%)")

        eval_eer = self.validate(self.eval_loader, "eval", self.epochs)
        print(f"Final eval_EER={eval_eer*100:.2f}%")

        self.experiment.end()


if __name__ == '__main__':
    train_cfg = {
        'train_dir':      'data/processed/train',
        'train_protocol':'data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt',
        'dev_dir':        'data/processed/dev',
        'dev_protocol':  'data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt',
        'eval_dir':       'data/processed/eval',
        'eval_protocol': 'data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt',
        'batch_size':     64,
    }
    model_cfg = {
        'input_channels':1,
        'channels_list':[96,192,384,256],
        'kernel_sizes':[9,5,5,4],
        'steps':[1,1,1,1],
        'kernel_pool':2,
        'step_pool':2,
        'dropout':0.3,
        'FLayer_size':512,
        'n_classes':2
    }
    run_cfg = {
        'lr':               1e-4,
        'epochs':           20,
        'checkpoint_path': 'best_lcnn4.pt',
        'device':          'cuda',
        'comet_api_key':   "YTEBlOIr52k3Tuyoh3G18TYVX",
        'comet_project':   "spoof-recognition-public",
        'comet_workspace': None,
    }

    trainer = Trainer(train_cfg, model_cfg, run_cfg)
    trainer.run()
