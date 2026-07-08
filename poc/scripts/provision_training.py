#!/usr/bin/env python3
"""
一次性建一台长驻 FLUX.2 训练机(g6e.4xlarge),供 ctl.py start/stop/train 复用。

按项目已知规格建机:
  - 实例类型 g6e.4xlarge(config.INSTANCE_TYPE,L40S 48GB)
  - DLAMI(.env 的 AMI_ID)、TRAINING_REGION(us-west-2)
  - 01_setup_infra.py 建好的 instance profile(flux-poc-ec2-instance-profile,
    已挂 AmazonSSMManagedInstanceCore → 全程走 SSM,无需入站规则)
  - 350GB gp3 EBS(缓存 ~106GB FLUX.2+Mistral,跨 stop/start 存活,免重下)
  - egress-only 安全组(复用 SG_NAME,无 inbound)
  - 跨 SUBNET_CANDIDATES 多 AZ 试容量,规避 InsufficientInstanceCapacity

建成后等 running + SSM Agent 上线,并把 instance-id 写回 poc/.env 的
PERSISTENT_INSTANCE_ID,之后一律用 ctl.py 管理生命周期。

Run: python3 provision_training.py [--force]
  --force: PERSISTENT_INSTANCE_ID 已存在时仍新建(默认拒绝,避免建出孤儿实例)
"""
import argparse
import time
from pathlib import Path

import boto3
from config import (
    ACCOUNT, TRAINING_REGION, AMI_ID, INSTANCE_TYPE, PROFILE_NAME, SG_NAME,
    SUBNET_CANDIDATES, PERSISTENT_INSTANCE_ID,
)

REGION = TRAINING_REGION
ENV_PATH = Path(__file__).parent.parent / ".env"
EBS_SIZE_GB = 350  # 缓存 ~106GB FLUX.2+Mistral;04 的 200GB 不足


def get_or_create_security_group(ec2, vpc_id: str) -> str:
    """egress-only 安全组:无入站(全走 SSM),复用同名 SG。"""
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


def wait_ssm_ready(instance_id: str, timeout: int = 300) -> bool:
    """等实例 SSM Agent 上线(与 ctl.py._wait_ssm_ready 同逻辑)。"""
    ssm = boto3.client("ssm", region_name=REGION)
    print("  waiting for SSM agent...", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = ssm.describe_instance_information(
            Filters=[{"Key": "InstanceIds", "Values": [instance_id]}]
        )["InstanceInformationList"]
        if info and info[0]["PingStatus"] == "Online":
            print(" ready")
            return True
        print(".", end="", flush=True)
        time.sleep(5)
    print(" timeout")
    return False


def write_instance_id_to_env(instance_id: str) -> None:
    """把 PERSISTENT_INSTANCE_ID 写回 poc/.env(替换已有行或追加)。"""
    if not ENV_PATH.exists():
        print(f"⚠ {ENV_PATH} 不存在,请手动设置 PERSISTENT_INSTANCE_ID={instance_id}")
        return
    lines = ENV_PATH.read_text().splitlines()
    new_line = f"PERSISTENT_INSTANCE_ID={instance_id}"
    replaced = False
    for i, line in enumerate(lines):
        if line.strip().startswith("PERSISTENT_INSTANCE_ID="):
            lines[i] = new_line
            replaced = True
            break
    if not replaced:
        lines.append(new_line)
    ENV_PATH.write_text("\n".join(lines) + "\n")
    print(f"  .env updated: {new_line}")


def provision() -> str:
    ec2 = boto3.client("ec2", region_name=REGION)

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
                    BlockDeviceMappings=[{
                        "DeviceName": "/dev/sda1",
                        "Ebs": {
                            "VolumeSize": EBS_SIZE_GB,
                            "VolumeType": "gp3",
                            "DeleteOnTermination": True,
                        },
                    }],
                    TagSpecifications=[{
                        "ResourceType": "instance",
                        "Tags": [
                            {"Key": "Name", "Value": "flux-poc-training"},
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
        raise RuntimeError(
            f"No {INSTANCE_TYPE} capacity available in any of the tried AZs: "
            + ", ".join(SUBNET_CANDIDATES)
        )

    instance_id = resp["Instances"][0]["InstanceId"]
    print(f"✅ EC2 training instance launched: {instance_id}")
    print(f"   Type:    {INSTANCE_TYPE} (L40S 48GB)")
    print(f"   EBS:     {EBS_SIZE_GB}GB gp3")
    print(f"   Region:  {REGION}")
    print(f"   Console: https://console.aws.amazon.com/ec2/home?region={REGION}#Instances:instanceId={instance_id}")

    print("Waiting for instance to reach running...")
    ec2.get_waiter("instance_running").wait(InstanceIds=[instance_id])
    print("  instance running")
    wait_ssm_ready(instance_id)

    write_instance_id_to_env(instance_id)
    print("\n下一步:python3 ctl.py status  /  python3 ctl.py train")
    return instance_id


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="PERSISTENT_INSTANCE_ID 已存在时仍新建(默认拒绝)")
    args = ap.parse_args()

    if PERSISTENT_INSTANCE_ID and not args.force:
        raise SystemExit(
            f"PERSISTENT_INSTANCE_ID 已设为 {PERSISTENT_INSTANCE_ID}。\n"
            "已有长驻训练机就用 `python3 ctl.py start` 唤醒;\n"
            "确需另建一台请加 --force(注意旧实例会变成孤儿,需自行清理)。"
        )
    provision()
