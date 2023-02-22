from omegaconf import DictConfig, OmegaConf
import hydra

from torch.utils.data import Dataset, DataLoader

@hydra.main(version_base=None, config_path="../conf", config_name="train.yaml")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    dataset : Dataset = hydra.utils.instantiate(cfg.data)
    print(dataset)
    dataloader : DataLoader = hydra.utils.instantiate(dataset, cfg.dataloader)
    print(dataloader)

if __name__ == "__main__":
    my_app()