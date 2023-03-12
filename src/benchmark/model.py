from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.benchmark import Timer

from src.benchmark.base import Benchmark


class ModelBenchmark(Benchmark):

    def __init__(
            self,
            model: nn.Module,
            weights_path: Path,
            input_size: Tuple[int, int, int],
            device: torch.device
    ):
        self.model = model
        self.weights_path = weights_path
        # self.model.load_state_dict(torch.load(self.weights_path))
        self.input_size = input_size
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def benchmark(self, n: int = 100):
        input = torch.rand(*self.input_size).unsqueeze(dim=0).to(self.device)
        timer = Timer(
            "self.model(input)",
            globals={'self': self, 'input': input}
        )
        print(timer.timeit(n))
