FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime

WORKDIR /workspace
ENV PYTHONPATH /workspace

ADD requirements.txt /workspace/
RUN pip install -r requirements.txt

ADD configs /workspace/configs/
ADD homographies /workspace/homographies/
ADD inference /workspace/inference/
ADD runfiles /workspace/runfiles/
ADD superres /workspace/superres/
ADD *.py /workspace/

ADD results/checkpoints/256/ /workspace/results/checkpoints/256/
ADD results/encoders/256_mlp/models/ /workspace/results/encoders/256_mlp/models/
ADD results/encoders/256_resnet/models/ /workspace/results/encoders/256_resnet/models/
ADD results/pretrained_models/ /workspace/results/pretrained_models/
ADD https://download.pytorch.org/models/vgg19-dcbb9e9d.pth /root/.cache/torch/checkpoints/vgg19-dcbb9e9d.pth
