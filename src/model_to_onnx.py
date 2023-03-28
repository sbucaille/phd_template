import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from src.models import ModelWrapper

os.environ['HYDRA_FULL_ERROR'] = '1'


@hydra.main(version_base=None, config_path="../conf", config_name="model_to_onnx.yaml")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    model_wrapper : ModelWrapper = hydra.utils.instantiate(cfg.model_wrapper)
    model = model_wrapper.model
    model.load_state_dict(torch.load(cfg.model_paths.weights))
    model.eval()

    input_names, inputs = model_wrapper.generate_inputs()
    output_names = model_wrapper.generate_outputs()

    # if cfg.inp.require_custom_generation :
    #     input_names, inputs = model.generate_inputs(cfg.model.input_info, torch.device("cpu"))
    #     # inputs = {input_names[i] : inputs[i] for i in range(len(input_names))}
    #     print(type(inputs))
    # else :
    #     input_names = [input.name for input in cfg.model.input_info.inputs]
    #     inputs = [torch.rand(*input).to(torch.device("cpu")) for input in cfg.model.input_info.inputs]
    # # input = torch.rand(cfg.batch_size, *cfg.input_size)


    # output_names = [output.name for output in cfg.model.input_info.outputs]


    torch.onnx.export(
        model=model,
        args=inputs,
        f=cfg.onnx_path,
        export_params=cfg.export_params,
        # dynamic_axes=cfg.dynamic_shapes,
        input_names=input_names,
        output_names=output_names,
        opset_version=cfg.opset_version,
        verbose=True
    )


if __name__ == "__main__":
    my_app()
