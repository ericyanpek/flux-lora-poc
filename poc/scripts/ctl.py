#!/usr/bin/env python3
"""
flux ctl — 一条命令管理 FLUX.2 训练/推理实例的生命周期。

封装所有 EC2 start/stop + SSM 远程执行细节,免去手工拼命令。

用法:
  python3 ctl.py status                 # 实例状态 + GPU/磁盘/容器
  python3 ctl.py start                  # 启动实例(等到 SSM 就绪)
  python3 ctl.py stop                   # 停止实例(省钱;EBS+模型缓存保留)
  python3 ctl.py train [--steps N]      # 在实例上跑训练(用 SLOTIP 数据集)
  python3 ctl.py infer ["<prompt>"]     # 推理不在训练机做 → 指引到 ComfyUI 推理机
  python3 ctl.py logs [train|infer]     # 看最近日志
  python3 ctl.py run "<shell command>"  # 在实例上执行任意命令

约定:实例 ID、region 从 poc/.env 读取(PERSISTENT_INSTANCE_ID / TRAINING_REGION)。
"""
import argparse
import json
import sys
import time
import boto3
from config import (
    PERSISTENT_INSTANCE_ID as IID, TRAINING_REGION as REGION, REGION as S3_REGION,
    ECR_URI, BUCKET, DATASET_PREFIX, TRIGGER_WORD,
)

ec2 = boto3.client("ec2", region_name=REGION)
ssm = boto3.client("ssm", region_name=REGION)


def _state():
    r = ec2.describe_instances(InstanceIds=[IID])
    return r["Reservations"][0]["Instances"][0]["State"]["Name"]


def _ssm_run(commands, timeout=3600, wait=True):
    """在实例上跑一组 shell 命令,返回 (status, stdout)。"""
    resp = ssm.send_command(
        InstanceIds=[IID],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": commands},
        TimeoutSeconds=min(timeout, 172800),
    )
    cid = resp["Command"]["CommandId"]
    if not wait:
        return cid, ""
    # 轮询直到结束
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(3)
        try:
            inv = ssm.get_command_invocation(CommandId=cid, InstanceId=IID)
        except ssm.exceptions.InvocationDoesNotExist:
            continue
        if inv["Status"] in ("Success", "Failed", "Cancelled", "TimedOut"):
            return inv["Status"], inv["StandardOutputContent"] + inv["StandardErrorContent"]
    return "Timeout", ""


def _wait_ssm_ready(timeout=300):
    """等实例 SSM Agent 上线。"""
    print("  waiting for SSM agent...", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = ssm.describe_instance_information(
            Filters=[{"Key": "InstanceIds", "Values": [IID]}]
        )["InstanceInformationList"]
        if info and info[0]["PingStatus"] == "Online":
            print(" ready")
            return True
        print(".", end="", flush=True)
        time.sleep(5)
    print(" timeout")
    return False


# ---- commands ----

def cmd_status(args):
    st = _state()
    print(f"Instance {IID} ({REGION}): {st}")
    if st != "running":
        print("  (start it with: python3 ctl.py start)")
        return
    status, out = _ssm_run([
        "echo GPU:; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo n/a",
        "echo DISK:; df -h / | tail -1",
        "echo CACHE:; du -sh /opt/flux-cache/hf 2>/dev/null || echo none",
        "echo CONTAINERS:; docker ps --format '{{.Status}} {{.Image}}' 2>/dev/null || echo none",
    ], timeout=60)
    print(out)


def cmd_start(args):
    st = _state()
    if st == "running":
        print(f"Already running: {IID}")
    else:
        print(f"Starting {IID}...")
        ec2.start_instances(InstanceIds=[IID])
        ec2.get_waiter("instance_running").wait(InstanceIds=[IID])
        print("  instance running")
    _wait_ssm_ready()


def cmd_stop(args):
    st = _state()
    if st in ("stopped", "stopping"):
        print(f"Already {st}: {IID}")
        return
    print(f"Stopping {IID} (EBS + model cache preserved)...")
    ec2.stop_instances(InstanceIds=[IID])
    print("  stop requested (not waiting). Check: python3 ctl.py status")


def _ensure_running():
    if _state() != "running":
        print("Instance not running. Run: python3 ctl.py start")
        sys.exit(1)


def cmd_train(args):
    _ensure_running()
    ts = time.strftime("%Y%m%d-%H%M%S")
    layer = getattr(args, "layer", None)
    if layer == "style":
        prefix, trigger, job = "datasets/slot-ip-v1-style/", "slotstyle", f"style-{ts}"
    elif layer == "char":
        prefix, trigger, job = "datasets/slot-ip-v1-char/", "slotchar", f"char-{ts}"
    else:
        prefix, trigger, job = DATASET_PREFIX, TRIGGER_WORD, f"slotip-{ts}"
    steps_env = f"-e STEPS={args.steps}" if args.steps else ""
    layer_env = f"-e LAYER={layer}" if layer else ""
    print(f"Launching {job} (layer={layer or 'base'}, trigger={trigger}, steps={args.steps or 'default'})...")
    cmd = (
        f"bash -c 'exec >> /var/log/flux-train-{job}.log 2>&1; "
        f"mkdir -p /tmp/td-{job} /tmp/out-{job} /opt/flux-cache/hf; "
        f"aws s3 sync s3://{BUCKET}/{prefix} /tmp/td-{job}/ >/dev/null 2>&1; "
        f"aws ecr get-login-password --region {S3_REGION} | docker login --username AWS --password-stdin {ECR_URI.split('/')[0]} >/dev/null 2>&1; "
        f"docker pull {ECR_URI} >/dev/null 2>&1; "
        f"HF=$(aws ssm get-parameter --region {S3_REGION} --name /flux-poc/hf-token --with-decryption --query Parameter.Value --output text); "
        f"WB=$(aws ssm get-parameter --region {S3_REGION} --name /flux-poc/wandb-key --with-decryption --query Parameter.Value --output text 2>/dev/null || echo \"\"); "
        f"docker run --gpus all --rm --shm-size=24g -e HF_TOKEN=\"$HF\" -e WANDB_API_KEY=\"$WB\" "
        f"-e TRIGGER_WORD={trigger} {layer_env} {steps_env} -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        f"-e HF_HOME=/root/.cache/huggingface -v /opt/flux-cache/hf:/root/.cache/huggingface "
        f"-v /tmp/td-{job}:/opt/ml/input/data/training -v /tmp/out-{job}:/opt/ml/model {ECR_URI} 2>&1; "
        f"EXIT=$?; "
        f"aws s3 sync /tmp/td-{job}/flux-lora-poc/ s3://{BUCKET}/outputs/lora-{job}/ --exclude \"*_cache/*\" >/dev/null 2>&1; "
        f"echo $([ $EXIT -eq 0 ] && echo SUCCESS || echo FAILED:$EXIT) | aws s3 cp - s3://{BUCKET}/outputs/lora-{job}/status.txt' &"
    )
    cid, _ = _ssm_run([cmd], wait=False)
    print(f"  job started (background). SSM cmd: {cid}")
    print(f"  watch:   python3 ctl.py logs train")
    print(f"  results: s3://{BUCKET}/outputs/lora-{job}/")


def cmd_infer(args):
    # 推理不在训练机上做。实测:ai-toolkit 的 loader 在 46GB 单卡上跑 FLUX.2 推理
    # 会 segfault/OOM(见 poc/scripts/inference/ 下几个失败尝试)。可靠路径是独立的
    # ComfyUI 推理机(官方 fp8 预量化底模 + 分层 LoRA),已验证跑通。
    print("训练机不做推理。请用独立 ComfyUI 推理机(已验证可用):")
    print("  1) 部署:  python3 poc/scripts/07_deploy_comfyui.py")
    print("  2) SSM 端口转发 8188(部署脚本会打印命令)")
    print("  3) 出图:  在推理机上跑 poc/scripts/inference/comfy_gen.py")
    print("            --config base|style|char|combo --out /exp/<cfg>")
    print("背景与设计:docs/superpowers/specs/2026-06-28-comfyui-inference-design.md")
    if args.prompt:
        print(f"\n(你给的 prompt {args.prompt!r} 请填进 comfy_gen.py 的 THEMES 或作为自定义主题)")


def cmd_logs(args):
    _ensure_running()
    kind = args.kind or "train"
    pat = "flux-train-" if kind == "train" else "pirate-infer"
    status, out = _ssm_run([
        f"ls -t /var/log/{pat}*.log 2>/dev/null | head -1 | xargs -r tail -40 | grep -viE 'Downloading|[0-9]+%\\|' || echo 'no logs'",
    ], timeout=60)
    print(out)


def cmd_run(args):
    _ensure_running()
    status, out = _ssm_run([args.command], timeout=600)
    print(f"[{status}]")
    print(out)


if __name__ == "__main__":
    if not IID:
        sys.exit("PERSISTENT_INSTANCE_ID not set in poc/.env")
    p = argparse.ArgumentParser(prog="flux-ctl")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("start")
    sub.add_parser("stop")
    pt = sub.add_parser("train"); pt.add_argument("--steps", type=int, default=None)
    pt.add_argument("--layer", choices=["style", "char"], default=None)
    pi = sub.add_parser("infer"); pi.add_argument("prompt", nargs="?", default=None)
    pl = sub.add_parser("logs"); pl.add_argument("kind", nargs="?", choices=["train", "infer"])
    pr = sub.add_parser("run"); pr.add_argument("command")
    args = p.parse_args()
    {
        "status": cmd_status, "start": cmd_start, "stop": cmd_stop,
        "train": cmd_train, "infer": cmd_infer, "logs": cmd_logs, "run": cmd_run,
    }[args.cmd](args)
