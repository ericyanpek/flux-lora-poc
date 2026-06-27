"""Shared config loaded from poc/.env"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

ACCOUNT        = os.environ["AWS_ACCOUNT"]
REGION         = os.environ["AWS_REGION"]
TRAINING_REGION = os.environ["TRAINING_REGION"]
BUCKET         = f"flux-poc-{ACCOUNT}-{REGION}"
ECR_REPO       = "flux-poc-training"
ECR_URI        = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/{ECR_REPO}:latest"
AMI_ID         = os.environ["AMI_ID"]
SUBNET_CANDIDATES = [
    os.environ["SUBNET_AZ1"],
    os.environ["SUBNET_AZ2"],
    os.environ["SUBNET_AZ3"],
    os.environ["SUBNET_AZ4"],
]
DATASET_PREFIX  = os.environ["DATASET_PREFIX"]
TRIGGER_WORD    = os.environ["TRIGGER_WORD"]
LOCAL_DATASET_PATH = os.environ.get("LOCAL_DATASET_PATH", "")

INSTANCE_TYPE   = "g6e.4xlarge"  # 128GB RAM: transformer (CPU) + Mistral-24B during quantization. g7e (96GB VRAM) preferred but capacity-constrained
PROFILE_NAME    = "flux-poc-ec2-instance-profile"
SG_NAME         = "flux-poc-training-sg"
ROLE_NAME       = "flux-poc-ec2-role"
SM_ROLE_NAME    = "AmazonSageMaker-ExecutionRole-20250207T115166"
CODEBUILD_PROJECT = "flux-poc-build"
CODEBUILD_ROLE    = "flux-poc-codebuild-role"
CW_LOG_GROUP      = "/codebuild/flux-poc-build"
