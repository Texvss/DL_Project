import hydra
from omegaconf import DictConfig
from src.trainer.trainer import Trainer

@hydra.main(version_base=None, config_path="src/configs", config_name="default")
def main(cfg: DictConfig):
    trainer = Trainer(cfg)
    trainer.run()

if __name__ == "__main__":
    main()