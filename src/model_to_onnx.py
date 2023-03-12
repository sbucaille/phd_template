import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from onnxruntime import OrtValue
from torch import nn, Tensor
from torch.utils.benchmark import Timer
import onnx
import onnxruntime as rt

os.environ['HYDRA_FULL_ERROR'] = '1'


@hydra.main(version_base=None, config_path="../conf", config_name="model_to_onnx.yaml")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    model: nn.Module = hydra.utils.instantiate(cfg.model)
    if "model_path" in cfg.keys():
        model.load_state_dict(torch.load(cfg.model_path))
    model.eval()

    input = torch.rand(cfg.batch_size, *cfg.input_size)

    torch.onnx.export(
        model=model,
        args=input,
        f=cfg.onnx_path,
        export_params=cfg.export_params,
        # dynamic_axes=cfg.dynamic_shapes,
        input_names=cfg.input_names,
        output_names=cfg.output_names,
        opset_version=cfg.opset_version
    )


if __name__ == "__main__":
    my_app()
