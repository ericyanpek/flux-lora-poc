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

# ai-toolkit pinned to 75781fb; install requirements first then force optimum-quanto to 0.2.7
# (requirements.txt pins optimum-quanto==0.2.4, so we must upgrade AFTER)
RUN git clone https://github.com/ostris/ai-toolkit /ai-toolkit && \
    cd /ai-toolkit && git checkout 75781fb && \
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
