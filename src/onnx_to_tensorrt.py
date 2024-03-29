import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

# from cuda import cuda
# import pycuda.autoinit
import tensorrt as trt


@hydra.main(version_base=None, config_path="../conf", config_name="onnx_to_tensorrt.yaml")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()

    if cfg.engine_precision == 'FP16':
        config.set_flag(trt.BuilderFlag.FP16)

    parser = trt.OnnxParser(network, logger)

    with open(cfg.onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed parsing .onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing .onnx file!")

    for i in range(network.num_inputs):
        input = network.get_input(i)
        print(input.name)
        print(input.shape)

        profile.set_shape(input.name,
                          input.shape,
                          input.shape,
                          input.shape)
    config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)
    with open(cfg.tensorrt_path, 'wb') as f:
        f.write(engine)
    print("TensorRT engine created")


if __name__ == "__main__":
    my_app()
