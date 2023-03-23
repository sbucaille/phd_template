from pathlib import Path
from time import sleep

import torch
from torch.utils.benchmark import Timer

from src.benchmark.base import Benchmark

from cuda import cuda
import tensorrt as trt


class TensorRTBenchmark(Benchmark):
    def __init__(
            self,
            tensorrt_path: Path,
            warmup: float = 0.25
    ):
        self.cuda_context = None
        self.tensorrt_path = tensorrt_path
        self.load_model()
        self.warmup = warmup

    def load_model(self):
        free_memory_before = self.get_free_gpu_memory()
        runtime = trt.Runtime(trt.Logger())
        with open(self.tensorrt_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        result, self.stream = cuda.cuStreamCreate(0)
        free_memory_after = self.get_free_gpu_memory()
        print(f"Model size :", (free_memory_before - free_memory_after) / 1024 ** 2, "MB")

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
        for _ in range(int(n * self.warmup)):
            self.context.execute_async_v2(pointers, self.stream)
        timer = Timer(
            "self.context.execute_async_v2(pointers, self.stream)",
            globals={'self': self, 'pointers': pointers}
        )
        measurement = timer.timeit(n)
        print(measurement)

    def get_free_gpu_memory(self):
        _, device = cuda.cuDeviceGet(0)
        if self.cuda_context is None:
            _, self.cuda_context = cuda.cuCtxCreate(0, device)
        result, free_mem, total_mem = cuda.cuMemGetInfo()
        return free_mem