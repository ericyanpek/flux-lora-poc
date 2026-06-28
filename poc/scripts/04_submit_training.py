"""
Launches a g6e.2xlarge EC2 instance to run FLUX.2-dev LoRA training.
UserData handles: ECR pull, S3 sync, docker run, result upload, shutdown.
Run: python3 04_submit_training.py --hf-token hf_xxx --wandb-key wbk_xxx
"""
import argparse
import base64
import boto3
import datetime
import os
import time
from config import (
    ACCOUNT, REGION as S3_REGION, TRAINING_REGION, BUCKET, ECR_URI,
    AMI_ID, INSTANCE_TYPE, PROFILE_NAME, SG_NAME, SUBNET_CANDIDATES, DATASET_PREFIX,
    TRIGGER_WORD,
)

REGION = TRAINING_REGION
ECR_REGION = S3_REGION


def get_or_create_security_group(ec2, vpc_id: str) -> str:
    resp = ec2.describe_security_groups(
        Filters=[
            {"Name": "group-name", "Values": [SG_NAME]},
            {"Name": "vpc-id", "Values": [vpc_id]},
        ]
    )
    if resp["SecurityGroups"]:
        sg_id = resp["SecurityGroups"][0]["GroupId"]
        print(f"SG: using existing {sg_id} in {vpc_id}")
        return sg_id

    sg = ec2.create_security_group(
        GroupName=SG_NAME,
        Description="FLUX POC training - egress only, no inbound (SSM)",
        VpcId=vpc_id,
    )
    sg_id = sg["GroupId"]
    print(f"SG: created {sg_id} in {vpc_id} (egress-only)")
    return sg_id


def build_userdata(job_id: str, has_wandb: bool) -> str:
    wandb_fetch = f'WANDB_API_KEY=$(aws ssm get-parameter --region {ECR_REGION} --name /flux-poc/wandb-key --with-decryption --query Parameter.Value --output text 2>/dev/null || echo "")' if has_wandb else 'WANDB_API_KEY=""'
    script = f"""#!/bin/bash
set -euo pipefail
exec > /var/log/flux-training.log 2>&1

JOB_ID="{job_id}"
BUCKET="{BUCKET}"
ECR_URI="{ECR_URI}"
ECR_REGION="{ECR_REGION}"
SSM_REGION="{ECR_REGION}"

echo "=== FLUX.2-dev LoRA Training: $JOB_ID ==="

# Docker setup (DLAMI has docker pre-installed but may need start)
systemctl start docker || true
sleep 5

# Fetch secrets from SSM
HF_TOKEN=$(aws ssm get-parameter --region $SSM_REGION --name /flux-poc/hf-token --with-decryption --query Parameter.Value --output text)
{wandb_fetch}

# ECR login
aws ecr get-login-password --region $ECR_REGION | docker login --username AWS --password-stdin {ACCOUNT}.dkr.ecr.$ECR_REGION.amazonaws.com

# Pull training image
docker pull $ECR_URI

# Model cache on EBS (persistent across stop/start) — ~106GB FLUX.2+Mistral pre-downloaded.
# EBS expanded to 350GB for this. Survives instance stop, so no 90GB re-download.
HF_CACHE=/opt/flux-cache/hf
mkdir -p /tmp/training-data /tmp/output $HF_CACHE

# Sync dataset from S3
aws s3 sync s3://$BUCKET/{DATASET_PREFIX} /tmp/training-data/
echo "Dataset synced: $(ls /tmp/training-data | wc -l) files"

# Run training.
# --shm-size=24g: DataLoader workers need shared memory (default 64MB causes Bus error)
# expandable_segments: reduce VRAM fragmentation (FLUX.2 prepare peak is near 46GB)
# TRIGGER_WORD: passed through to ai-toolkit config
# HF cache on NVMe so the ~90GB FLUX.2 + Mistral download survives container --rm
set +e
docker run --gpus all --rm --shm-size=24g \\
  -e HF_TOKEN="$HF_TOKEN" \\
  -e WANDB_API_KEY="$WANDB_API_KEY" \\
  -e TRIGGER_WORD="{TRIGGER_WORD}" \\
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
  -e HF_HOME=/root/.cache/huggingface \\
  -v $HF_CACHE:/root/.cache/huggingface \\
  -v /tmp/training-data:/opt/ml/input/data/training \\
  -v /tmp/output:/opt/ml/model \\
  $ECR_URI
EXIT_CODE=$?
set -e

# Write status marker to S3
if [ $EXIT_CODE -eq 0 ]; then
  echo "SUCCESS" | aws s3 cp - s3://$BUCKET/outputs/$JOB_ID/status.txt
  echo "Training succeeded"
else
  echo "FAILED:$EXIT_CODE" | aws s3 cp - s3://$BUCKET/outputs/$JOB_ID/status.txt
  echo "Training FAILED with exit code $EXIT_CODE"
fi

# Upload results. ai-toolkit writes LoRA weights + samples into the training_folder
# subdir (flux-lora-poc/), NOT /opt/ml/model. Sync that, excluding bulky caches.
aws s3 sync /tmp/training-data/flux-lora-poc/ s3://$BUCKET/outputs/$JOB_ID/ \\
  --exclude "*_latent_cache/*" --exclude "*_t_e_cache/*"
aws s3 sync /tmp/output/ s3://$BUCKET/outputs/$JOB_ID/ 2>/dev/null || true
echo "Results uploaded to s3://$BUCKET/outputs/$JOB_ID/"

if [ $EXIT_CODE -eq 0 ]; then
  echo "Shutting down..."
  shutdown -h now
else
  echo "Instance kept running for debugging. Check /var/log/flux-training.log"
fi
"""
    return base64.b64encode(script.encode()).decode()


def launch_instance(hf_token: str, wandb_key: str) -> tuple:
    ssm = boto3.client("ssm", region_name=ECR_REGION)
    ssm.put_parameter(Name="/flux-poc/hf-token", Value=hf_token, Type="SecureString", Overwrite=True)
    if wandb_key:
        ssm.put_parameter(Name="/flux-poc/wandb-key", Value=wandb_key, Type="SecureString", Overwrite=True)
    print(f"Secrets stored in SSM Parameter Store ({ECR_REGION})")

    ec2 = boto3.client("ec2", region_name=REGION)

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    job_id = f"flux2-lora-ec2-{ts}"
    userdata = build_userdata(job_id, bool(wandb_key))

    resp = None
    for subnet_id in SUBNET_CANDIDATES:
        subnet_info = ec2.describe_subnets(SubnetIds=[subnet_id])["Subnets"][0]
        vpc_id = subnet_info["VpcId"]
        az = subnet_info["AvailabilityZone"]
        sg_id = get_or_create_security_group(ec2, vpc_id)

        for attempt in range(5):
            try:
                resp = ec2.run_instances(
                    ImageId=AMI_ID,
                    InstanceType=INSTANCE_TYPE,
                    MinCount=1,
                    MaxCount=1,
                    SubnetId=subnet_id,
                    SecurityGroupIds=[sg_id],
                    IamInstanceProfile={"Name": PROFILE_NAME},
                    UserData=userdata,
                    BlockDeviceMappings=[{
                        "DeviceName": "/dev/sda1",
                        "Ebs": {"VolumeSize": 200, "VolumeType": "gp3", "DeleteOnTermination": True},
                    }],
                    TagSpecifications=[{
                        "ResourceType": "instance",
                        "Tags": [
                            {"Key": "Name", "Value": job_id},
                            {"Key": "flux:job-id", "Value": job_id},
                            {"Key": "flux:purpose", "Value": "training"},
                        ],
                    }],
                )
                print(f"Launched in {az} ({subnet_id})")
                break
            except ec2.exceptions.ClientError as e:
                err = str(e)
                if "Invalid IAM Instance Profile" in err or "iamInstanceProfile" in err:
                    print(f"IAM profile not yet propagated, retrying in 10s... (attempt {attempt+1}/5)")
                    time.sleep(10)
                elif "InsufficientInstanceCapacity" in err:
                    print(f"No {INSTANCE_TYPE} capacity in {az}, trying next AZ...")
                    break
                else:
                    raise
        else:
            raise RuntimeError("IAM instance profile failed to propagate after 50s")

        if resp:
            break
    else:
        raise RuntimeError(f"No {INSTANCE_TYPE} capacity available in any of the tried AZs: " + ", ".join(SUBNET_CANDIDATES))

    instance_id = resp["Instances"][0]["InstanceId"]
    print(f"✅ EC2 instance launched: {instance_id}")
    print(f"   Job ID:  {job_id}")
    print(f"   Type:    {INSTANCE_TYPE} (L40S 48GB)")
    print(f"   Monitor: https://console.aws.amazon.com/ec2/home?region={REGION}#Instances:instanceId={instance_id}")
    if wandb_key:
        print(f"   W&B:     https://wandb.ai (project: flux2-lora-poc)")

    state_file = "/tmp/last_flux_ec2_job.txt"
    with open(state_file, "w") as f:
        f.write(f"{instance_id}\n{job_id}")
    print(f"   State:   {state_file}")

    return instance_id, job_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--wandb-key", default=None)
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    wandb_key = args.wandb_key or os.environ.get("WANDB_API_KEY", "")

    if not hf_token:
        raise SystemExit("HF_TOKEN required: pass --hf-token or set HF_TOKEN env var")

    launch_instance(hf_token, wandb_key)
