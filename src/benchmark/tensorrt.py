from pathlib import Path
from time import sleep

import torch
from torch.utils.benchmark import Timer

from src.benchmark.base import Benchmark

import tensorrt as trt


class TensorRTBenchmark(Benchmark):
    def __init__(
            self,
            tensorrt_path: Path

    ):
        self.tensorrt_path = tensorrt_path
        self.load_model()

    def load_model(self):
        runtime = trt.Runtime(trt.Logger())
        with open(self.tensorrt_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def allocate_io_bindings(self):
        in_bindings = list(filter(lambda binding: self.engine.binding_is_input(binding),
                                  [binding for binding in range(self.engine.num_bindings)]))
        out_bindings = list(filter(lambda binding: not self.engine.binding_is_input(binding),
                                   [binding for binding in range(self.engine.num_bindings)]))

        tensors = []
        pointers = []
        for in_binding in in_bindings:
            in_tensor = torch.rand(tuple(self.engine.get_binding_shape(in_binding)), dtype=torch.float32, device="cuda")
            tensors.append(in_tensor)
            pointers.append(in_tensor.data_ptr())

        for out_binding in out_bindings:
            out_tensor = torch.empty(tuple(self.engine.get_binding_shape(out_binding)), dtype=torch.float32,
                                     device="cuda")
            tensors.append(out_tensor)
            pointers.append(out_tensor.data_ptr())

        return tensors, pointers, in_bindings, out_bindings

    def benchmark_inference_time(self, n: int = 100, monitor_gpu_memory: bool = False):
        tensors, pointers, in_bindings, out_bindings = self.allocate_io_bindings()
        timer = Timer(
            "self.context.execute_async_v2(pointers, torch.cuda.current_stream().cuda_stream)",
            globals={'self': self, 'pointers': pointers}
        )
        measurement = timer.timeit(n)
        print(measurement)