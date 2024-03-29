from pathlib import Path
from typing import List, Tuple

import numpy as np
from omegaconf import OmegaConf
from torch.utils.benchmark import Timer

from src.benchmark.base import Benchmark
import torch
import onnxruntime as rt

from src.models import ModelWrapper


class OnnxBenchmark(Benchmark):
    def __init__(
            self,
            model_wrapper: ModelWrapper,
            onnx_path: Path,
            providers: List[Tuple],
            device: str,
            device_index: int
    ):
        self.model_wrapper = model_wrapper
        self.onnx_path = onnx_path
        self.providers = providers
        self.device = device
        self.device_index = device_index
        self.load_onnx()

    def load_onnx(self):
        providers = [
            (provider.name, OmegaConf.to_container(provider.parameters, resolve=True)) for provider in self.providers
        ]
        self.session = rt.InferenceSession(self.onnx_path, providers=providers)

    def allocate_io_bindings(self):
        io_binding = self.session.io_binding()
        for input_name, input in zip(*self.model_wrapper.generate_inputs()):
            io_binding.bind_input(
                name=input_name,
                device_type=self.device,
                device_id=self.device_index,
                element_type=np.float32,
                shape=input.shape,
                buffer_ptr=input.data_ptr()
            )

        # for input in self.session.get_inputs():
        #     shape = [1 if isinstance(dimension, str) else dimension for dimension in input.shape]
        #     input_tensor = torch.rand(
        #         shape,
        #         dtype=torch.float32,
        #         device=self.device
        #     )
        #     io_binding.bind_input(
        #         name=input.name,
        #         device_type=self.device,
        #         device_id=self.device_index,
        #         element_type=np.float32,
        #         shape=shape,
        #         buffer_ptr=input_tensor.data_ptr()
        #     )

        for output in self.session.get_outputs():
            shape = [1 if isinstance(dimension, str) else dimension for dimension in output.shape]
            output_tensor = torch.empty(
                shape,
                dtype=torch.float32,
                device=self.device
            )
            io_binding.bind_output(
                name=output.name,
                device_type=self.device,
                device_id=self.device_index,
                element_type=np.float32,
                shape=shape,
                buffer_ptr=output_tensor.data_ptr()
            )

        return io_binding

    def benchmark_inference_time(self, n: int = 100):
        io_binding = self.allocate_io_bindings()
        self.session.run_with_iobinding(io_binding)
        timer = Timer(
            "self.session.run_with_iobinding(io_binding)",
            globals={'self': self, 'io_binding': io_binding}
        )
        print(timer.timeit(n))
