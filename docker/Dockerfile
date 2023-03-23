#docker run -it -w /app -e PYTHONPATH=/app -v ~/phd_template/:/app --gpus all phd bash

FROM nvcr.io/nvidia/tensorrt:22.12-py3

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update -y
RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install python3.10 -y
RUN apt install python3-pip -y

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

ENV PYTHONPATH=/app