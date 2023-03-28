from torch.utils.benchmark import Timer

from src.benchmark.base import Benchmark
from src.models import ModelWrapper


class ModelBenchmark(Benchmark):

    def __init__(
            self,
            model_wrapper: ModelWrapper,
    ):
        self.model_wrapper = model_wrapper
        self.model = model_wrapper.model
        self.model.eval()

    def benchmark_inference_time(self, n: int = 100):
        _, inputs = self.model_wrapper.generate_inputs()
        timer = Timer(
            "self.model(*input)",
            globals={'self': self, 'input': inputs}
        )
        print(timer.timeit(n))
