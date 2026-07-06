"""
Launches a g6e.4xlarge EC2 inference instance running ComfyUI with FLUX.2-dev fp8 models.
UserData handles: ComfyUI install, fp8 model fetch (S3-first, HF fallback), LoRA sync, startup.
Instance is persistent (no auto-shutdown) — access via SSM port forwarding on port 8188.
Run: python3 07_deploy_comfyui.py
"""
import base64
import boto3
import datetime
import time
from config import (
    ACCOUNT, REGION as S3_REGION, TRAINING_REGION, BUCKET,
    AMI_ID, PROFILE_NAME, SG_NAME, SUBNET_CANDIDATES,
)

REGION = TRAINING_REGION
INFER_INSTANCE_TYPE = "g6e.4xlarge"


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


def build_userdata() -> str:
    script = f"""#!/bin/bash
set -euo pipefail
exec > /var/log/comfyui-setup.log 2>&1
echo "=== ComfyUI setup $(date) ==="

BUCKET="{BUCKET}"
S3_REGION="{S3_REGION}"

HF_TOKEN=$(aws ssm get-parameter --region $S3_REGION --name /flux-poc/hf-token --with-decryption --query Parameter.Value --output text)
export HF_TOKEN

# 1. ComfyUI on persistent EBS
cd /opt
git clone https://github.com/comfyanonymous/ComfyUI.git 2>/dev/null || (cd /opt/ComfyUI && git pull)
cd /opt/ComfyUI
python3 -m venv venv || true
source venv/bin/activate
pip install --upgrade pip
# PyTorch cu124 (L40S/Ada compatible; DLAMI may have it but reinstalling in venv is safe)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 2. Create model directories
mkdir -p models/diffusion_models models/text_encoders models/vae models/loras

# 3. fetch_model: S3-first, fallback to HF download then cache back to S3
# Usage: fetch_model <s3_key> <local_path> <hf_repo> <hf_file>
fetch_model() {{
    local S3_KEY="$1"
    local LOCAL_PATH="$2"
    local HF_REPO="$3"
    local HF_FILE="$4"

    if aws s3 ls "s3://$BUCKET/$S3_KEY" > /dev/null 2>&1; then
        echo "S3 hit: $S3_KEY -> $LOCAL_PATH"
        aws s3 cp "s3://$BUCKET/$S3_KEY" "$LOCAL_PATH"
    else
        echo "S3 miss: $S3_KEY — downloading from HF $HF_REPO/$HF_FILE"
        pip install huggingface_hub 2>/dev/null || true
        huggingface-cli download "$HF_REPO" "$HF_FILE" --local-dir "$(dirname $LOCAL_PATH)" --token "$HF_TOKEN"
        # Rename if huggingface-cli placed file by its basename
        HF_BASENAME=$(basename "$HF_FILE")
        if [ "$HF_BASENAME" != "$(basename $LOCAL_PATH)" ] && [ -f "$(dirname $LOCAL_PATH)/$HF_BASENAME" ]; then
            mv "$(dirname $LOCAL_PATH)/$HF_BASENAME" "$LOCAL_PATH"
        fi
        echo "Caching to S3: s3://$BUCKET/$S3_KEY"
        aws s3 cp "$LOCAL_PATH" "s3://$BUCKET/$S3_KEY"
    fi
}}

# fp8 base models from Comfy-Org/flux2-dev split_files
fetch_model \
    "comfyui-models/diffusion_models/flux2_dev_fp8mixed.safetensors" \
    "models/diffusion_models/flux2_dev_fp8mixed.safetensors" \
    "Comfy-Org/flux2-dev" \
    "split_files/diffusion_models/flux2_dev_fp8mixed.safetensors"

fetch_model \
    "comfyui-models/text_encoders/mistral_3_small_flux2_fp8.safetensors" \
    "models/text_encoders/mistral_3_small_flux2_fp8.safetensors" \
    "Comfy-Org/flux2-dev" \
    "split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors"

fetch_model \
    "comfyui-models/vae/flux2-vae.safetensors" \
    "models/vae/flux2-vae.safetensors" \
    "Comfy-Org/flux2-dev" \
    "split_files/vae/flux2-vae.safetensors"

# LoRAs from training outputs (S3 only — no HF fallback)
echo "Fetching LoRA weights from S3..."
aws s3 cp "s3://$BUCKET/outputs/lora-style-20260706-112005/flux-lora-poc.safetensors" \
    "models/loras/slotstyle.safetensors" || echo "WARN: slotstyle LoRA not found, skipping"
aws s3 cp "s3://$BUCKET/outputs/lora-char-20260706-134937/flux-lora-poc.safetensors" \
    "models/loras/slotchar.safetensors" || echo "WARN: slotchar LoRA not found, skipping"

# 4. Start ComfyUI (SSM port forwarding; --lowvram offloads layers to save VRAM)
nohup venv/bin/python main.py --listen 127.0.0.1 --port 8188 --lowvram > /var/log/comfyui.log 2>&1 &
echo "=== ComfyUI started on 127.0.0.1:8188 ==="
echo "=== Setup complete $(date) ==="
"""
    return base64.b64encode(script.encode()).decode()


def launch_instance() -> str:
    ec2 = boto3.client("ec2", region_name=REGION)

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    instance_name = f"comfyui-infer-{ts}"
    userdata = build_userdata()

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
                    InstanceType=INFER_INSTANCE_TYPE,
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
                            {"Key": "Name", "Value": instance_name},
                            {"Key": "flux:purpose", "Value": "inference"},
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
                    print(f"No {INFER_INSTANCE_TYPE} capacity in {az}, trying next AZ...")
                    break
                else:
                    raise
        else:
            raise RuntimeError("IAM instance profile failed to propagate after 50s")

        if resp:
            break
    else:
        raise RuntimeError(
            f"No {INFER_INSTANCE_TYPE} capacity available in any of the tried AZs: "
            + ", ".join(SUBNET_CANDIDATES)
        )

    instance_id = resp["Instances"][0]["InstanceId"]
    print(f"EC2 instance launched: {instance_id}")
    print(f"   Name:    {instance_name}")
    print(f"   Type:    {INFER_INSTANCE_TYPE}")
    print(f"   Region:  {REGION}")
    print(f"   Console: https://console.aws.amazon.com/ec2/home?region={REGION}#Instances:instanceId={instance_id}")

    state_file = "/tmp/last_comfyui_instance.txt"
    with open(state_file, "w") as f:
        f.write(instance_id)
    print(f"   State:   {state_file}")

    print()
    print("=== SSM Port Forwarding (run after instance is ready ~15-20 min) ===")
    print(
        f"aws ssm start-session --target {instance_id}"
        f" --document-name AWS-StartPortForwardingSession"
        f' --parameters \'{{"portNumber":["8188"],"localPortNumber":["8188"]}}\''
        f" --region {REGION}"
    )
    print()
    print("Setup log:  aws ssm start-session --target", instance_id, "# then: tail -f /var/log/comfyui-setup.log")
    print("ComfyUI log: /var/log/comfyui.log")
    print("Browser:    http://localhost:8188  (after SSM tunnel is open)")
    print("Note:       First run ~15-20 min (model downloads + pip installs)")

    return instance_id


if __name__ == "__main__":
    launch_instance()
