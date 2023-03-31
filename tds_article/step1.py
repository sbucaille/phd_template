import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule

@hydra.main(version_base=None, config_path="../conf", config_name="train.yaml")
def my_app(cfg: DictConfig) -> None:
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    print(datamodule)

if __name__ == "__main__":
    my_app()
