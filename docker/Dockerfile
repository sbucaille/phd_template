FROM nvcr.io/nvidia/tensorrt:23.02-py3

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update -y
RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install python3.10 -y

RUN apt install python3-pip -y
RUN pip install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu116

RUN pip install onnx onnxruntime-gpu==1.13.1
RUN pip install cuda-python pycuda tensorrt
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.7/compat/libcuda.so.1

RUN pip install pytorch-lightning hydra-core
RUN pip install git+https://github.com/rwightman/pytorch-image-models.git