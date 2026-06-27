FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip \
    git wget curl libgl1 libglib2.0-0 libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# PyTorch 2.6 + CUDA 12.4
RUN pip3 install --no-cache-dir \
    "torch==2.6.0+cu124" "torchvision==0.21.0+cu124" "torchaudio==2.6.0+cu124" \
    --index-url https://download.pytorch.org/whl/cu124

# ai-toolkit main branch — has dedicated Flux2Model (arch: flux2) with proper
# meta-device load + assign-weights handling for FLUX.2-dev. Do NOT pin to 75781fb
# (predates flux2 support). Upgrade optimum-quanto to 0.2.7 (--no-deps to keep torch 2.6)
# since requirements pins 0.2.4 which breaks fake-impl registration on torch 2.6.
RUN git clone https://github.com/ostris/ai-toolkit /ai-toolkit && \
    cd /ai-toolkit && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir --force-reinstall --no-deps "optimum-quanto==0.2.7"

# Runtime packages
RUN pip3 install --no-cache-dir boto3 wandb

# scipy 1.12.0 (pinned by ai-toolkit) requires numpy<2; force-downgrade if needed
RUN pip3 install --no-cache-dir "numpy<2"

ENV PYTHONPATH=/ai-toolkit
WORKDIR /ai-toolkit

COPY train_entry.py /opt/ml/code/train_entry.py
RUN chmod +x /opt/ml/code/train_entry.py

ENTRYPOINT ["python3", "/opt/ml/code/train_entry.py"]
