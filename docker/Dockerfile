#docker run -it -w /app -e PYTHONPATH=/app -v ~/phd_template/:/app --gpus all phd bash

FROM nvcr.io/nvidia/tensorrt:22.12-py3

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update -y
RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install python3.10 -y

RUN apt install python3-pip -y
RUN pip install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu116
RUN pip install transformers[onnx]
RUN pip install git+https://github.com/rwightman/pytorch-image-models.git

RUN pip install pytorch-lightning hydra-core

RUN pip install onnx onnxruntime-gpu==1.14

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

ENV PYTHONPATH=/app