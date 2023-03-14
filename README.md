# phd_template
Template to create training pipeline using PytorchLighning, Hydra and DVC tools

Medium article : soon

Instructions :
```bash
cd docker
docker build -t phd .
cd ..
docker run -it -shm-size=1g -v .:/app phd python <train|model_to_onnx|...>.py 
```
