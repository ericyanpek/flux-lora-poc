FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip \
    git wget curl libgl1 libglib2.0-0 libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# PyTorch 2.4 + CUDA 12.1 — compatible with optimum-quanto 0.2.4 (PyTorch 2.6 breaks fake impl registration)
RUN pip3 install --no-cache-dir \
    "torch==2.4.1+cu121" "torchvision==0.19.1+cu121" "torchaudio==2.4.1+cu121" \
    --index-url https://download.pytorch.org/whl/cu121

# ai-toolkit: pinned to 75781fb — last known-good commit for FLUX.2-dev (issue #872)
RUN git clone https://github.com/ostris/ai-toolkit /ai-toolkit && \
    cd /ai-toolkit && git checkout 75781fb && \
    pip3 install --no-cache-dir -r requirements.txt

# Force torch back to 2.4 — torchao==0.10.0 in requirements.txt upgrades torch to 2.6
# which breaks optimum-quanto 0.2.4's fake impl registration
RUN pip3 install --no-cache-dir --force-reinstall \
    "torch==2.4.1+cu121" "torchvision==0.19.1+cu121" "torchaudio==2.4.1+cu121" \
    --index-url https://download.pytorch.org/whl/cu121

# Runtime packages
RUN pip3 install --no-cache-dir boto3 wandb

# scipy 1.12.0 (pinned by ai-toolkit) requires numpy<2; force-downgrade if needed
RUN pip3 install --no-cache-dir "numpy<2"

ENV PYTHONPATH=/ai-toolkit
WORKDIR /ai-toolkit

COPY train_entry.py /opt/ml/code/train_entry.py
RUN chmod +x /opt/ml/code/train_entry.py

ENTRYPOINT ["python3", "/opt/ml/code/train_entry.py"]
