import os
from typing import List, Any

import hydra
from omegaconf import DictConfig, OmegaConf

from src.benchmark.base import Benchmark


os.environ['HYDRA_FULL_ERROR'] = '1'

@hydra.main(version_base=None, config_path="../conf", config_name="model_benchmark.yaml")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # model: nn.Module = hydra.utils.instantiate(cfg.model)
    # if not OmegaConf.is_missing(cfg, "model_path"):
    #     model.load_state_dict(torch.load(cfg.model_path))

    benchmarks : List[Benchmark] = [hydra.utils.instantiate(benchmark) for benchmark in cfg.benchmarks.values()]
    for benchmark in benchmarks:
        benchmark.benchmark_inference_time(cfg.number_run)




if __name__ == "__main__":
    my_app()