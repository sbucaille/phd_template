from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn


class BaseTask(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            learning_rate: float,
            *args: Any,
            **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.learning_rate = learning_rate
        self.model = model
        print(self.model)
        self.loss = nn.CrossEntropyLoss()
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        return self.model(x)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(
            self,
            batch,
            *args: Any,
            **kwargs: Any
    ) -> STEP_OUTPUT:
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)
        self.log('train_loss', loss)
        return loss

