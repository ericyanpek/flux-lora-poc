# EC2 迁移实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 FLUX.2-dev LoRA 训练从 SageMaker 迁移到 EC2 g6e.2xlarge，清理 SageMaker 遗留资源，接入 W&B 监控。

**Architecture:** UserData 自动化方案——EC2 启动后自动 pull 镜像、sync 数据集、运行训练容器、推结果到 S3、关机。本地脚本只负责触发启动和轮询状态。

**Tech Stack:** boto3, Python 3, Docker, ai-toolkit, W&B, AWS EC2/IAM/S3/ECR

---

## 常量（所有脚本共用）

```
ACCOUNT    = "984072314535"
REGION     = "us-east-1"
BUCKET     = "flux-poc-984072314535-us-east-1"
ECR_URI    = "984072314535.dkr.ecr.us-east-1.amazonaws.com/flux-poc-training:latest"
AMI_ID     = "ami-012ba162b9cd2729c"   # Deep Learning OSS Nvidia Driver AMI (Ubuntu 22.04), supports G6e
SUBNET_ID  = "subnet-0aa203a61adbcd8be"
INSTANCE_TYPE = "g6e.2xlarge"
VOLUME_GB  = 200
ROLE_NAME  = "flux-poc-ec2-role"
PROFILE_NAME = "flux-poc-ec2-instance-profile"
SM_ROLE_NAME = "AmazonSageMaker-ExecutionRole-20250207T115166"
```

---

### Task 1: 更新 Dockerfile — 加 wandb

**Files:**
- Modify: `poc/docker/Dockerfile`

- [ ] **Step 1: 替换 SageMaker 依赖行，加入 wandb**

将 Dockerfile 中：
```dockerfile
# SageMaker runtime packages (not in ai-toolkit requirements)
RUN pip3 install --no-cache-dir boto3 sagemaker-training
```
替换为：
```dockerfile
# Runtime packages
RUN pip3 install --no-cache-dir boto3 wandb
```

- [ ] **Step 2: 验证文件内容正确**

```bash
grep -n "wandb\|sagemaker-training" poc/docker/Dockerfile
```
预期输出：
```
行号: RUN pip3 install --no-cache-dir boto3 wandb
```
不应出现 `sagemaker-training`。

- [ ] **Step 3: Commit**

```bash
git -C /Users/yabolin/claude-code/flux add poc/docker/Dockerfile
git -C /Users/yabolin/claude-code/flux commit -m "chore: replace sagemaker-training with wandb in Dockerfile"
```

---

### Task 2: 更新 train_entry.py — FLUX.2-dev + rank 32 + W&B

**Files:**
- Modify: `poc/docker/train_entry.py`

- [ ] **Step 1: 完整替换 train_entry.py**

```python
"""
Bridge script: reads env vars, builds ai-toolkit YAML config, runs training,
saves output to /opt/ml/model/.
On EC2: env vars injected via docker run -e. On SageMaker: reads hyperparameters.json.
"""
import json
import os
import sys
import yaml

sys.path.insert(0, "/ai-toolkit")

HYPERPARAM_PATH = "/opt/ml/input/config/hyperparameters.json"
TRAINING_DATA_PATH = "/opt/ml/input/data/training"
OUTPUT_PATH = "/opt/ml/model"
CHECKPOINT_PATH = "/opt/ml/checkpoints"


def load_hyperparameters():
    # EC2 path: env vars take precedence; SageMaker path: read JSON file
    hp = {}
    if os.path.exists(HYPERPARAM_PATH):
        with open(HYPERPARAM_PATH) as f:
            raw = json.load(f)
        hp = {k: v.strip('"') if isinstance(v, str) else v for k, v in raw.items()}
    # env vars override file (EC2 mode)
    for key in ["trigger_word", "model_name", "steps", "lr", "rank", "sample_every"]:
        env_val = os.environ.get(key.upper())
        if env_val:
            hp[key] = env_val
    return hp


def build_config(hp: dict) -> dict:
    trigger_word = hp.get("trigger_word", "GAMECATV1")
    steps = int(hp.get("steps", "1500"))
    lr = float(hp.get("lr", "1e-4"))
    rank = int(hp.get("rank", "32"))
    sample_every = int(hp.get("sample_every", "250"))
    model_name = hp.get("model_name", "black-forest-labs/FLUX.2-dev")
    wandb_key = os.environ.get("WANDB_API_KEY", "")

    sample_prompts = [
        f"a {trigger_word} character sitting on a beach, casual game style, vibrant colors",
        f"a {trigger_word} character in a fantasy forest, detailed character art",
        f"a {trigger_word} character portrait, close up, high quality",
        "a character sitting on a beach, casual game style, vibrant colors",
    ]

    process = {
        "type": "sd_trainer",
        "training_folder": TRAINING_DATA_PATH,
        "output_folder": OUTPUT_PATH,
        "device": "cuda:0",
        "model": {
            "name_or_path": model_name,
            "is_flux": True,
            "quantize": True,
            "low_vram": False,  # L40S 48GB — no need for CPU-offload quantize
        },
        "train": {
            "batch_size": 1,
            "steps": steps,
            "gradient_accumulation_steps": 4,
            "train_unet": True,
            "train_text_encoder": False,
            "lr": lr,
            "optimizer": "adamw8bit",
            "lr_scheduler": "cosine",
            "gradient_checkpointing": True,
            "noise_scheduler": "flowmatch",
            "dtype": "bf16",
        },
        "network": {
            "type": "lora",
            "linear": rank,
            "linear_alpha": rank,
        },
        "save": {
            "save_every": sample_every,
            "save_format": "safetensors",
            "max_step_saves_to_keep": 4,
        },
        "sample": {
            "sample_every": sample_every,
            "width": 1024,
            "height": 1024,
            "prompts": sample_prompts,
            "neg": "",
            "seed": 42,
            "guidance_scale": 3.5,
            "sample_steps": 20,
            "walk_seed": False,
        },
        "datasets": [{
            "folder_path": TRAINING_DATA_PATH,
            "caption_ext": "txt",
            "resolution": [1024, 1024],
            "default_caption": f"a character in {trigger_word} style",
            "flip_aug": True,
        }],
    }

    if wandb_key:
        process["logging"] = {
            "use_wandb": True,
            "project": "flux2-lora-poc",
            "run_name": trigger_word,
        }

    return {
        "job": "extension",
        "config": {
            "name": "flux-lora-poc",
            "process": [process],
        },
    }


def main():
    hp = load_hyperparameters()
    print(f"Hyperparameters: {hp}")

    hf_token = hp.get("hf_token", os.environ.get("HF_TOKEN", ""))
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        print("HF token set")
    else:
        print("WARNING: No HF_TOKEN — FLUX.2-dev download will fail if license-gated")

    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if wandb_key:
        print("W&B logging enabled")
    else:
        print("W&B key not set — logging disabled")

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    config = build_config(hp)
    config_path = "/tmp/train_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print("Generated ai-toolkit config:")
    with open(config_path) as f:
        print(f.read())

    from toolkit.job import get_job
    job = get_job(config_path)
    job.run()
    job.cleanup()
    print("Training complete")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证关键字段**

```bash
grep -n "FLUX.2-dev\|rank.*32\|low_vram.*False\|use_wandb\|WANDB_API_KEY" poc/docker/train_entry.py
```
预期：每行都能找到对应内容，不出现 `FLUX.1-dev` 或 `low_vram.*True`。

- [ ] **Step 3: Commit**

```bash
git -C /Users/yabolin/claude-code/flux add poc/docker/train_entry.py
git -C /Users/yabolin/claude-code/flux commit -m "feat: switch to FLUX.2-dev, rank 32, add W&B logging"
```

---

### Task 3: 新增 00_cleanup_sagemaker.py

**Files:**
- Create: `poc/scripts/00_cleanup_sagemaker.py`

- [ ] **Step 1: 创建清理脚本**

```python
"""
One-time cleanup of SageMaker POC leftovers.
- Removes non-latest ECR images (saves ~$0.10/GB/month storage)
- Removes flux-poc-s3-access inline policy from SageMaker role
Run: python3 00_cleanup_sagemaker.py
"""
import boto3

ACCOUNT = "984072314535"
REGION = "us-east-1"
ECR_REPO = "flux-poc-training"
SM_ROLE_NAME = "AmazonSageMaker-ExecutionRole-20250207T115166"


def cleanup_ecr_old_images():
    ecr = boto3.client("ecr", region_name=REGION)
    resp = ecr.describe_images(repositoryName=ECR_REPO)
    images = resp["imageDetails"]

    # find the digest tagged as 'latest'
    latest_digest = None
    for img in images:
        if "latest" in img.get("imageTags", []):
            latest_digest = img["imageDigest"]
            break

    to_delete = [
        {"imageDigest": img["imageDigest"]}
        for img in images
        if img["imageDigest"] != latest_digest
    ]

    if not to_delete:
        print("ECR: no old images to delete")
        return

    ecr.batch_delete_image(repositoryName=ECR_REPO, imageIds=to_delete)
    print(f"ECR: deleted {len(to_delete)} old image(s), kept latest ({latest_digest[:19]}...)")


def cleanup_sm_iam_policy():
    iam = boto3.client("iam", region_name=REGION)
    policy_name = "flux-poc-s3-access"
    try:
        iam.delete_role_policy(RoleName=SM_ROLE_NAME, PolicyName=policy_name)
        print(f"IAM: removed inline policy '{policy_name}' from {SM_ROLE_NAME}")
    except iam.exceptions.NoSuchEntityException:
        print(f"IAM: policy '{policy_name}' not found (already removed or never applied)")


if __name__ == "__main__":
    cleanup_ecr_old_images()
    cleanup_sm_iam_policy()
    print("\n✅ SageMaker cleanup complete")
```

- [ ] **Step 2: 验证脚本语法**

```bash
cd /Users/yabolin/claude-code/flux && python3 -c "import ast; ast.parse(open('poc/scripts/00_cleanup_sagemaker.py').read()); print('syntax OK')"
```
预期：`syntax OK`

- [ ] **Step 3: Commit**

```bash
git -C /Users/yabolin/claude-code/flux add poc/scripts/00_cleanup_sagemaker.py
git -C /Users/yabolin/claude-code/flux commit -m "feat: add SageMaker cleanup script"
```

---

### Task 4: 更新 01_setup_infra.py — 创建 EC2 IAM instance profile

**Files:**
- Modify: `poc/scripts/01_setup_infra.py`

- [ ] **Step 1: 完整替换 01_setup_infra.py**

```python
"""
One-time infrastructure setup for EC2-based FLUX.2-dev training.
Creates: S3 bucket, ECR repo, EC2 IAM role + instance profile.
Run: python3 01_setup_infra.py
"""
import boto3
import json

ACCOUNT = "984072314535"
REGION = "us-east-1"
BUCKET = f"flux-poc-{ACCOUNT}-{REGION}"
ECR_REPO = "flux-poc-training"
ROLE_NAME = "flux-poc-ec2-role"
PROFILE_NAME = "flux-poc-ec2-instance-profile"


def create_bucket():
    s3 = boto3.client("s3", region_name=REGION)
    try:
        s3.create_bucket(Bucket=BUCKET)
        print(f"S3: created bucket {BUCKET}")
    except Exception as e:
        if "BucketAlreadyOwnedByYou" in str(e) or "BucketAlreadyExists" in str(e):
            print(f"S3: bucket already exists")
        else:
            raise
    s3.put_public_access_block(
        Bucket=BUCKET,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True,
            "BlockPublicPolicy": True, "RestrictPublicBuckets": True,
        },
    )
    for prefix in ["datasets/", "outputs/", "checkpoints/"]:
        s3.put_object(Bucket=BUCKET, Key=prefix)
    print("S3: folder structure OK")


def create_ecr_repo():
    ecr = boto3.client("ecr", region_name=REGION)
    try:
        ecr.create_repository(
            repositoryName=ECR_REPO,
            imageScanningConfiguration={"scanOnPush": True},
        )
        print(f"ECR: created repo {ECR_REPO}")
    except ecr.exceptions.RepositoryAlreadyExistsException:
        print(f"ECR: repo already exists")


def create_ec2_iam_profile():
    iam = boto3.client("iam", region_name=REGION)

    # 1. Create role with EC2 trust policy
    trust = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    })
    try:
        iam.create_role(RoleName=ROLE_NAME, AssumeRolePolicyDocument=trust)
        print(f"IAM: created role {ROLE_NAME}")
    except iam.exceptions.EntityAlreadyExistsException:
        print(f"IAM: role already exists")

    # 2. Attach SSM managed policy
    iam.attach_role_policy(
        RoleName=ROLE_NAME,
        PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
    )

    # 3. Inline policy: S3 + ECR
    inline = json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
                "Resource": [
                    f"arn:aws:s3:::{BUCKET}",
                    f"arn:aws:s3:::{BUCKET}/*",
                ],
            },
            {
                "Effect": "Allow",
                "Action": [
                    "ecr:GetAuthorizationToken",
                    "ecr:BatchGetImage",
                    "ecr:GetDownloadUrlForLayer",
                ],
                "Resource": "*",
            },
        ],
    })
    iam.put_role_policy(RoleName=ROLE_NAME, PolicyName="flux-poc-ec2-policy", PolicyDocument=inline)
    print("IAM: S3 + ECR inline policy applied")

    # 4. Create instance profile and add role
    try:
        iam.create_instance_profile(InstanceProfileName=PROFILE_NAME)
        print(f"IAM: created instance profile {PROFILE_NAME}")
    except iam.exceptions.EntityAlreadyExistsException:
        print(f"IAM: instance profile already exists")

    try:
        iam.add_role_to_instance_profile(InstanceProfileName=PROFILE_NAME, RoleName=ROLE_NAME)
        print(f"IAM: role attached to instance profile")
    except iam.exceptions.LimitExceededException:
        print(f"IAM: role already attached to instance profile")


if __name__ == "__main__":
    create_bucket()
    create_ecr_repo()
    create_ec2_iam_profile()
    print(f"\n✅ Infrastructure ready")
    print(f"  S3:      s3://{BUCKET}/")
    print(f"  ECR:     {ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/{ECR_REPO}")
    print(f"  Profile: {PROFILE_NAME}")
```

- [ ] **Step 2: 验证语法**

```bash
cd /Users/yabolin/claude-code/flux && python3 -c "import ast; ast.parse(open('poc/scripts/01_setup_infra.py').read()); print('syntax OK')"
```
预期：`syntax OK`

- [ ] **Step 3: Commit**

```bash
git -C /Users/yabolin/claude-code/flux add poc/scripts/01_setup_infra.py
git -C /Users/yabolin/claude-code/flux commit -m "feat: replace SM IAM patch with EC2 instance profile setup"
```

---

### Task 5: 替换 04_submit_training.py — 启动 EC2 g6e.2xlarge

**Files:**
- Modify: `poc/scripts/04_submit_training.py`

- [ ] **Step 1: 完整替换 04_submit_training.py**

```python
"""
Launches a g6e.2xlarge EC2 instance to run FLUX.2-dev LoRA training.
UserData handles: ECR pull, S3 sync, docker run, result upload, shutdown.
Run: python3 04_submit_training.py --hf-token hf_xxx --wandb-key wbk_xxx
"""
import argparse
import base64
import boto3
import datetime

ACCOUNT = "984072314535"
REGION = "us-east-1"
BUCKET = f"flux-poc-{ACCOUNT}-{REGION}"
ECR_URI = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/flux-poc-training:latest"
AMI_ID = "ami-012ba162b9cd2729c"
SUBNET_ID = "subnet-0aa203a61adbcd8be"
INSTANCE_TYPE = "g6e.2xlarge"
PROFILE_NAME = "flux-poc-ec2-instance-profile"
SG_NAME = "flux-poc-training-sg"


def get_or_create_security_group(ec2) -> str:
    resp = ec2.describe_security_groups(
        Filters=[{"Name": "group-name", "Values": [SG_NAME]}]
    )
    if resp["SecurityGroups"]:
        sg_id = resp["SecurityGroups"][0]["GroupId"]
        print(f"SG: using existing {sg_id}")
        return sg_id

    # Get VPC from subnet
    subnet = ec2.describe_subnets(SubnetIds=[SUBNET_ID])["Subnets"][0]
    vpc_id = subnet["VpcId"]

    sg = ec2.create_security_group(
        GroupName=SG_NAME,
        Description="FLUX POC training — egress only, no inbound (SSM)",
        VpcId=vpc_id,
    )
    sg_id = sg["GroupId"]
    # Remove default all-egress rule is not needed — we want full egress for S3/ECR/W&B
    print(f"SG: created {sg_id} (egress-only)")
    return sg_id


def build_userdata(job_id: str, hf_token: str, wandb_key: str) -> str:
    script = f"""#!/bin/bash
set -euo pipefail
exec > /var/log/flux-training.log 2>&1

JOB_ID="{job_id}"
BUCKET="{BUCKET}"
ECR_URI="{ECR_URI}"
REGION="{REGION}"
HF_TOKEN="{hf_token}"
WANDB_API_KEY="{wandb_key}"

echo "=== FLUX.2-dev LoRA Training: $JOB_ID ==="

# Docker setup (DLAMI has docker pre-installed but may need start)
systemctl start docker || true
sleep 5

# ECR login
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin {ACCOUNT}.dkr.ecr.$REGION.amazonaws.com

# Pull training image
docker pull $ECR_URI

# Sync dataset from S3
mkdir -p /tmp/training-data /tmp/output
aws s3 sync s3://$BUCKET/datasets/poc-character-v1/ /tmp/training-data/
echo "Dataset synced: $(ls /tmp/training-data | wc -l) files"

# Run training
set +e
docker run --gpus all --rm \\
  -e HF_TOKEN="$HF_TOKEN" \\
  -e WANDB_API_KEY="$WANDB_API_KEY" \\
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

# Upload results regardless of exit code (partial results useful for debugging)
aws s3 sync /tmp/output/ s3://$BUCKET/outputs/$JOB_ID/

echo "Results uploaded to s3://$BUCKET/outputs/$JOB_ID/"
echo "Shutting down..."
shutdown -h now
"""
    return base64.b64encode(script.encode()).decode()


def launch_instance(hf_token: str, wandb_key: str) -> tuple:
    ec2 = boto3.client("ec2", region_name=REGION)

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    job_id = f"flux2-lora-ec2-{ts}"

    sg_id = get_or_create_security_group(ec2)
    userdata = build_userdata(job_id, hf_token, wandb_key)

    resp = ec2.run_instances(
        ImageId=AMI_ID,
        InstanceType=INSTANCE_TYPE,
        MinCount=1,
        MaxCount=1,
        SubnetId=SUBNET_ID,
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
    parser.add_argument("--hf-token", required=True)
    parser.add_argument("--wandb-key", default="")
    args = parser.parse_args()
    launch_instance(args.hf_token, args.wandb_key)
```

- [ ] **Step 2: 验证语法**

```bash
cd /Users/yabolin/claude-code/flux && python3 -c "import ast; ast.parse(open('poc/scripts/04_submit_training.py').read()); print('syntax OK')"
```
预期：`syntax OK`

- [ ] **Step 3: Commit**

```bash
git -C /Users/yabolin/claude-code/flux add poc/scripts/04_submit_training.py
git -C /Users/yabolin/claude-code/flux commit -m "feat: replace SageMaker submission with EC2 g6e.2xlarge launcher"
```

---

### Task 6: 替换 05_monitor.py — 轮询 EC2 状态

**Files:**
- Modify: `poc/scripts/05_monitor.py`

- [ ] **Step 1: 完整替换 05_monitor.py**

```python
"""
Polls EC2 instance state until stopped, then downloads results from S3.
Usage:
  python3 05_monitor.py                        # reads /tmp/last_flux_ec2_job.txt
  python3 05_monitor.py --instance-id i-xxx --job-id flux2-lora-ec2-xxx
"""
import argparse
import boto3
import time
from pathlib import Path

ACCOUNT = "984072314535"
REGION = "us-east-1"
BUCKET = f"flux-poc-{ACCOUNT}-{REGION}"
RESULTS_DIR = Path("/Users/yabolin/claude-code/flux/poc/results")
STATE_FILE = Path("/tmp/last_flux_ec2_job.txt")


def read_state_file() -> tuple:
    if not STATE_FILE.exists():
        raise ValueError(f"No state file at {STATE_FILE}. Pass --instance-id and --job-id explicitly.")
    lines = STATE_FILE.read_text().strip().splitlines()
    if len(lines) < 2:
        raise ValueError(f"State file malformed. Expected 2 lines (instance-id, job-id).")
    return lines[0].strip(), lines[1].strip()


def poll_until_stopped(instance_id: str) -> str:
    ec2 = boto3.client("ec2", region_name=REGION)
    print(f"Monitoring EC2 instance: {instance_id}")
    print("Polling every 60s. Ctrl+C stops polling (instance continues running).\n")

    last_state = ""
    while True:
        resp = ec2.describe_instances(InstanceIds=[instance_id])
        state = resp["Reservations"][0]["Instances"][0]["State"]["Name"]

        if state != last_state:
            print(f"  State: {state}")
            last_state = state

        if state == "stopped":
            print("\nInstance stopped — training complete or failed.")
            return state
        if state == "terminated":
            print("\nInstance terminated unexpectedly.")
            return state

        time.sleep(60)


def check_status(job_id: str) -> str:
    s3 = boto3.client("s3", region_name=REGION)
    status_key = f"outputs/{job_id}/status.txt"
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=status_key)
        return obj["Body"].read().decode().strip()
    except s3.exceptions.NoSuchKey:
        return "UNKNOWN (status.txt not found)"


def download_results(job_id: str) -> Path:
    s3 = boto3.client("s3", region_name=REGION)
    output_prefix = f"outputs/{job_id}/"
    local_dir = RESULTS_DIR / job_id
    local_dir.mkdir(parents=True, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    downloaded = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=output_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel_path = key[len(output_prefix):]
            if not rel_path:
                continue
            local_path = local_dir / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(BUCKET, key, str(local_path))
            downloaded.append(str(local_path))
            print(f"  Downloaded: {rel_path}")

    lora_files = [f for f in downloaded if f.endswith(".safetensors")]
    sample_files = [f for f in downloaded if any(f.endswith(e) for e in [".png", ".jpg", ".jpeg"])]
    print(f"\n✅ Results saved to: {local_dir}")
    print(f"   LoRA weights:  {len(lora_files)}")
    print(f"   Sample images: {len(sample_files)}")
    return local_dir


def print_summary(local_dir: Path):
    print("\n=== POC RESULTS SUMMARY ===")
    for f in sorted(local_dir.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.relative_to(local_dir)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-id", default=None)
    parser.add_argument("--job-id", default=None)
    args = parser.parse_args()

    if args.instance_id and args.job_id:
        instance_id, job_id = args.instance_id, args.job_id
    else:
        instance_id, job_id = read_state_file()

    print(f"Job ID: {job_id}")
    poll_until_stopped(instance_id)

    status = check_status(job_id)
    print(f"Training status: {status}")

    if status.startswith("FAILED"):
        code = status.split(":", 1)[-1]
        print(f"\nTraining failed with exit code {code}.")
        print(f"Check logs: https://console.aws.amazon.com/cloudwatch/home?region={REGION}#logsV2:log-groups")
        print(f"Or SSH/SSM into instance {instance_id} and read /var/log/flux-training.log")
    else:
        local_dir = download_results(job_id)
        print_summary(local_dir)
        print(f"\nOpen sample images: open {local_dir}")
```

- [ ] **Step 2: 验证语法**

```bash
cd /Users/yabolin/claude-code/flux && python3 -c "import ast; ast.parse(open('poc/scripts/05_monitor.py').read()); print('syntax OK')"
```
预期：`syntax OK`

- [ ] **Step 3: Commit**

```bash
git -C /Users/yabolin/claude-code/flux add poc/scripts/05_monitor.py
git -C /Users/yabolin/claude-code/flux commit -m "feat: replace SageMaker monitor with EC2 state poller"
```

---

### Task 7: 执行清理和基础设施初始化

- [ ] **Step 1: 运行 SageMaker 清理**

```bash
cd /Users/yabolin/claude-code/flux/poc/scripts && python3 00_cleanup_sagemaker.py
```
预期输出包含：
```
ECR: deleted N old image(s), kept latest...
IAM: removed inline policy 'flux-poc-s3-access'...
✅ SageMaker cleanup complete
```

- [ ] **Step 2: 运行基础设施初始化**

```bash
cd /Users/yabolin/claude-code/flux/poc/scripts && python3 01_setup_infra.py
```
预期输出包含：
```
IAM: created role flux-poc-ec2-role  (或 already exists)
IAM: S3 + ECR inline policy applied
IAM: created instance profile flux-poc-ec2-instance-profile  (或 already exists)
✅ Infrastructure ready
```

- [ ] **Step 3: 验证 instance profile 可见**

```bash
aws iam get-instance-profile --instance-profile-name flux-poc-ec2-instance-profile \
  --query "InstanceProfile.{Profile:InstanceProfileName,Role:Roles[0].RoleName}" \
  --output table
```
预期：显示 `flux-poc-ec2-instance-profile` 和 `flux-poc-ec2-role`。

- [ ] **Step 4: Commit（如果有未提交的变动）**

```bash
git -C /Users/yabolin/claude-code/flux status
```
此时应无未提交文件。

---

### Task 8: 重建并推送 Docker 镜像

- [ ] **Step 1: 构建并推送新镜像（含 wandb，无 sagemaker-training）**

```bash
bash /Users/yabolin/claude-code/flux/poc/scripts/02_build_push.sh
```
预期最后输出：
```
Image pushed: 984072314535.dkr.ecr.us-east-1.amazonaws.com/flux-poc-training:latest
```
注意：构建时间约 10-20 分钟。

- [ ] **Step 2: 验证镜像包含 wandb、不含 sagemaker-training**

```bash
# 检查本地镜像的 pip list（如果本地有构建缓存）
docker run --rm --entrypoint pip3 \
  984072314535.dkr.ecr.us-east-1.amazonaws.com/flux-poc-training:latest \
  list 2>/dev/null | grep -E "wandb|sagemaker"
```
预期：出现 `wandb`，不出现 `sagemaker`。

---

## 运行顺序（首次）

```bash
# 一次性初始化（已做过可跳过）
python3 01_setup_infra.py

# 一次性清理 SageMaker 遗留
python3 00_cleanup_sagemaker.py

# 镜像有变更时重建
bash 02_build_push.sh

# 每次训练
python3 04_submit_training.py --hf-token hf_xxx --wandb-key wbk_xxx

# 监控 + 下载结果
python3 05_monitor.py
```
