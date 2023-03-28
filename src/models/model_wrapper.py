from typing import Tuple, List

import torch
from omegaconf import DictConfig
from torch import Tensor


class ModelWrapper:
    def __init__(
            self,
            model,
            input: DictConfig,
            output: DictConfig,
            device: torch.device
    ):
        self.model = model
        self.input = input
        self.output = output
        self.device = device
        self.model.to(self.device)

    def generate_inputs(self) -> Tuple[List[str], List[Tensor]]:
        if self.input.require_custom_generation:
            return self.model.generate_inputs(self.input, self.device)
        else:
            input_names = [input.name for input in self.input.tensors]
            inputs = [torch.rand(*input.size, device=self.device) for input in self.input.tensors]
            return input_names, inputs

    def generate_outputs(self) -> List[str]:
        output_names = [output.name for output in self.output.tensors]
        return output_names
