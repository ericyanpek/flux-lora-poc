"""
Launches a g6e.4xlarge EC2 inference instance running ComfyUI with FLUX.2-dev fp8 models.
UserData handles: ComfyUI install, fp8 model fetch (S3-first, HF fallback), LoRA sync, startup.
Instance is persistent (no auto-shutdown) — access via SSM port forwarding on port 8188.

Run:
  python3 07_deploy_comfyui.py                       # default: style + char (分层组合)
  python3 07_deploy_comfyui.py --layers style        # 单 LoRA(只拉 style)
  python3 07_deploy_comfyui.py --layers slotip        # 单 LoRA(base 路径产物 lora-slotip-*)
  python3 07_deploy_comfyui.py --layers style char   # 显式指定多层

Each layer resolves the latest SUCCESS run at outputs/lora-<layer>-*/ and lands at
models/loras/<layer>.safetensors. A missing layer is soft-skipped; hard error only if
zero layers loaded (unless ALLOW_BASE=1). Use the same <layer> name as --config in
comfy_gen.py (e.g. deploy --layers style → comfy_gen --config style).
"""
import argparse
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


def build_userdata(lora_specs) -> str:
    # lora_specs: list of (prefix, dest_basename). Each layer is fetched independently;
    # a missing layer is SOFT-skipped (warn + continue) so single-LoRA deploys work.
    # Hard error only if ZERO LoRAs loaded (unless ALLOW_BASE=1) — a ComfyUI that
    # silently came up with no adapter at all is worse than a loud failure.
    fetch_lines = "\n".join(
        f'fetch_latest_lora "{prefix}" "models/loras/{dest}" '
        f'&& LOADED=$((LOADED+1)) || echo "  layer {prefix} not found — skipped"'
        for prefix, dest in lora_specs
    )
    script = f"""#!/bin/bash
set -euo pipefail
exec > /var/log/comfyui-setup.log 2>&1
echo "=== ComfyUI setup $(date) ==="

BUCKET="{BUCKET}"
S3_REGION="{S3_REGION}"

HF_TOKEN=$(aws ssm get-parameter --region $S3_REGION --name /flux-poc/hf-token --with-decryption --query Parameter.Value --output text)
export HF_TOKEN

# 0. DLAMI's system python lacks ensurepip → `python3 -m venv` fails silently.
#    Install python3-venv BEFORE creating the venv.
apt-get update -y >/dev/null 2>&1 || true
apt-get install -y python3-venv python3-pip >/dev/null 2>&1 || true

# 1. ComfyUI on persistent EBS
cd /opt
git clone https://github.com/comfyanonymous/ComfyUI.git 2>/dev/null || (cd /opt/ComfyUI && git pull)
cd /opt/ComfyUI
python3 -m venv venv
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
        # NOTE: ComfyUI's requirements pull huggingface_hub >=1.x, where the old
        # `huggingface-cli download` command is REMOVED (prints a deprecation
        # notice and exits non-zero → would kill `set -e`). Use `hf download`.
        hf download "$HF_REPO" "$HF_FILE" --local-dir /tmp/hfdl --token "$HF_TOKEN"
        # hf download preserves the repo-relative path ($HF_FILE) under --local-dir
        mv "/tmp/hfdl/$HF_FILE" "$LOCAL_PATH"
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

# LoRAs from training outputs (S3 only — no HF fallback).
# Resolve the latest SUCCESSFUL run per layer, so a new training run is picked up
# with no code change. ai-toolkit writes each run to
# outputs/lora-<layer>-<timestamp>/{{flux-lora-poc.safetensors, status.txt}}.
#
# CURRENT CONTRACT (POC): latest-SUCCESS-by-S3-LastModified. This is the actual
# training→inference coordination mechanism today. The W&B Artifacts + S3 manifest
# + SSM pointer registry described in docs/architecture/dual-stack-plan.md is the
# ASPIRATIONAL production design, not yet implemented.
#
# Safety over the naive "sort | tail -1":
#   1. Skip runs whose status.txt != SUCCESS (a failed run's partial output must
#      never be auto-promoted).
#   2. Order by S3 LastModified (monotonic, API-provided) rather than parsing the
#      timestamp out of the prefix name (which assumes single-writer/same-timezone).
#   3. Per-layer miss is SOFT (warn + skip) so single-LoRA deploys work; a HARD
#      error (exit 1) fires only if ZERO layers loaded, unless ALLOW_BASE=1.
echo "Resolving latest SUCCESSFUL LoRA runs from S3..."
LOADED=0
fetch_latest_lora() {{
    local LAYER="$1"       # e.g. style | char | slotip
    local DEST="$2"
    # List run prefixes for this layer, newest-first by LastModified.
    local PREFIXES
    PREFIXES=$(aws s3api list-objects-v2 --bucket "$BUCKET" --prefix "outputs/lora-$LAYER-" \
        --query 'reverse(sort_by(Contents,&LastModified))[].Key' --output text 2>/dev/null \
        | tr '\t' '\n' | grep '/status.txt$' | sed 's#/status.txt$##')
    local RUN
    for RUN in $PREFIXES; do
        local ST
        ST=$(aws s3 cp "s3://$BUCKET/$RUN/status.txt" - 2>/dev/null)
        if [ "$ST" = "SUCCESS" ]; then
            echo "Latest SUCCESS $LAYER LoRA: $RUN"
            aws s3 cp "s3://$BUCKET/$RUN/flux-lora-poc.safetensors" "$DEST"
            return 0
        fi
        echo "  skip $RUN (status=$ST)"
    done
    return 1   # not found — caller decides (soft-skip)
}}
{fetch_lines}
if [ "$LOADED" -eq 0 ] && [ "${{ALLOW_BASE:-0}}" != "1" ]; then
    echo "ERROR: no successful LoRA run found for any requested layer (set ALLOW_BASE=1 to run base-only)"
    exit 1
fi
echo "LoRAs loaded: $LOADED"

# 4. Start ComfyUI (SSM port forwarding; --lowvram offloads layers to save VRAM)
nohup venv/bin/python main.py --listen 127.0.0.1 --port 8188 --lowvram > /var/log/comfyui.log 2>&1 &
echo "=== ComfyUI started on 127.0.0.1:8188 ==="
echo "=== Setup complete $(date) ==="
"""
    return base64.b64encode(script.encode()).decode()


def launch_instance(lora_specs) -> str:
    ec2 = boto3.client("ec2", region_name=REGION)

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    instance_name = f"comfyui-infer-{ts}"
    userdata = build_userdata(lora_specs)

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", default=["style", "char"],
                    help="要拉取的 LoRA 层名(= outputs/lora-<层>-* 前缀,也 = comfy_gen --config)。"
                         "默认 style char;单 LoRA 传单个,如 --layers style 或 --layers slotip")
    args = ap.parse_args()
    # 落地文件名需与 comfy_gen.py 的 LoRA 引用一致:
    #   style -> slotstyle.safetensors, char -> slotchar.safetensors(comfy_gen 内置默认)
    #   其它层 -> <层>.safetensors,配合 comfy_gen.py --lora <文件名> 使用
    DEST_NAME = {"style": "slotstyle.safetensors", "char": "slotchar.safetensors"}
    lora_specs = [(layer, DEST_NAME.get(layer, f"{layer}.safetensors")) for layer in args.layers]
    print(f"Deploying ComfyUI with LoRA layers: {', '.join(args.layers)}")
    for layer, dest in lora_specs:
        print(f"   {layer:12s} -> models/loras/{dest}")
    launch_instance(lora_specs)
