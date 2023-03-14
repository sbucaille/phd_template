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

    input = network.get_input(0)
    print(input.name)

    if cfg.batch_size is not None:
        profile_shape_min = (cfg.batch_size.min, cfg.input_size[0], cfg.input_size[1], cfg.input_size[2])
        profile_shape_opt = (cfg.batch_size.opt, cfg.input_size[0], cfg.input_size[1], cfg.input_size[2])
        profile_shape_max = (cfg.batch_size.max, cfg.input_size[0], cfg.input_size[1], cfg.input_size[2])
    else:
        profile_shape_min = (cfg.input_size[0], cfg.input_size[1], cfg.input_size[2])
        profile_shape_opt = (cfg.input_size[0], cfg.input_size[1], cfg.input_size[2])
        profile_shape_max = (cfg.input_size[0], cfg.input_size[1], cfg.input_size[2])

    print(profile_shape_opt)
    profile.set_shape(input.name,
                      profile_shape_min,
                      profile_shape_opt,
                      profile_shape_max)
    config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)
    with open(cfg.tensorrt_path, 'wb') as f:
        f.write(engine)
    print("TensorRT engine created")


if __name__ == "__main__":
    my_app()
