import os

import pytorch_lightning
from omegaconf import DictConfig, OmegaConf
import hydra
from pytorch_lightning import Trainer
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose

from src.tasks.base import BaseTask

os.environ['HYDRA_FULL_ERROR'] = '1'


@hydra.main(version_base=None, config_path="../conf", config_name="train.yaml")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    dataset: Dataset = hydra.utils.instantiate(cfg.data, transform=Compose([ToTensor()]))
    dataloader: DataLoader = hydra.utils.instantiate(
        cfg.dataloader,
        dataset=dataset
    )
    model: nn.Module = hydra.utils.instantiate(cfg.model)

    task: BaseTask = hydra.utils.instantiate(
        cfg.task,
        model=model
    )

    trainer = Trainer(**cfg.trainer)
    trainer.fit(task, dataloader)


if __name__ == "__main__":
    my_app()
