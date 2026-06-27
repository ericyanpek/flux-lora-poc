FROM 984072314535.dkr.ecr.us-east-1.amazonaws.com/flux-poc-base:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update -o Acquire::Retries=3 && \
    apt-get install -y -o Acquire::Retries=3 \
    python3.11 python3.11-dev python3-pip \
    git wget curl libgl1 libglib2.0-0 libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# PyTorch 2.6 + CUDA 12.4
RUN pip3 install --no-cache-dir \
    "torch==2.6.0+cu124" "torchvision==0.21.0+cu124" "torchaudio==2.6.0+cu124" \
    --index-url https://download.pytorch.org/whl/cu124

# ai-toolkit: pinned to 75781fb — last known-good commit for FLUX.2-dev (issue #872)
# Pin torchao to 0.7.0 to avoid pulling torch 2.6 deps that break optimum-quanto 0.2.4
# Pin optimum-quanto to 0.2.6 — last version before mandatory torch 2.6 requirement
RUN git clone https://github.com/ostris/ai-toolkit /ai-toolkit && \
    cd /ai-toolkit && git checkout 75781fb && \
    pip3 install --no-cache-dir \
        "torchao==0.7.0" \
        "optimum-quanto==0.2.6" && \
    pip3 install --no-cache-dir \
        --ignore-installed torch torchvision torchaudio \
        -r requirements.txt

# Runtime packages
RUN pip3 install --no-cache-dir boto3 wandb

# scipy 1.12.0 (pinned by ai-toolkit) requires numpy<2; force-downgrade if needed
RUN pip3 install --no-cache-dir "numpy<2"

ENV PYTHONPATH=/ai-toolkit
WORKDIR /ai-toolkit

COPY train_entry.py /opt/ml/code/train_entry.py
RUN chmod +x /opt/ml/code/train_entry.py

ENTRYPOINT ["python3", "/opt/ml/code/train_entry.py"]
