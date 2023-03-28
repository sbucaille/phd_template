import os
import time
from pathlib import Path
from typing import List

import pytorch_lightning
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
from pytorch_lightning import Trainer, Callback
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose
import timm

from src.data.base import BaseDataModule
from src.models import ModelWrapper
from src.tasks.base import BaseTask

os.environ['HYDRA_FULL_ERROR'] = '1'


@hydra.main(version_base=None, config_path="../conf", config_name="train.yaml")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    torch.set_float32_matmul_precision('high')

    datamodule: BaseDataModule = hydra.utils.instantiate(cfg.data)

    model_wrapper : ModelWrapper = hydra.utils.instantiate(cfg.model_wrapper)
    model = model_wrapper.model
    # model = torch.compile(model)

    task: BaseTask = hydra.utils.instantiate(
        cfg.task,
        model=model
    )

    callbacks: List[Callback] = [hydra.utils.instantiate(callback) for callback in cfg.callbacks.values()]
    print(callbacks)
    trainer = Trainer(**cfg.trainer, callbacks=callbacks)
    # if cfg.resume:
    #     trainer.fit(task, datamodule, ckpt_path=Path(cfg.callbacks.model_checkpoint.dirpath) / "last.ckpt")
    # else :
    #     trainer.fit(task, datamodule)

    task.save_model(cfg.model_paths.weights)


if __name__ == "__main__":
    my_app()
