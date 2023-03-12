from typing import Tuple

from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor

from src.data.base import BaseDataModule


class FakeDataModule(BaseDataModule):

    def __init__(
            self,
            size: int,
            image_size: Tuple[int, int, int],
            num_classes: int,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.size = size
        self.num_classes = num_classes
        self.image_size = image_size

    def setup(self, stage: str) -> None:
        self.train_set = FakeData(self.size, self.image_size, self.num_classes, transform=ToTensor())
        self.validation_set = FakeData(self.size, self.image_size, self.num_classes, transform=ToTensor())
        self.testing_set = FakeData(self.size, self.image_size, self.num_classes, transform=ToTensor())

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=self.shuffle)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.validation_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.testing_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)
