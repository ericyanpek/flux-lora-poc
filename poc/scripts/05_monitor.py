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
from config import REGION, TRAINING_REGION, BUCKET

RESULTS_DIR = Path(__file__).parent.parent / "results"
STATE_FILE = Path("/tmp/last_flux_ec2_job.txt")


def read_state_file() -> tuple:
    if not STATE_FILE.exists():
        raise ValueError(f"No state file at {STATE_FILE}. Pass --instance-id and --job-id explicitly.")
    lines = STATE_FILE.read_text().strip().splitlines()
    if len(lines) < 2:
        raise ValueError(f"State file malformed. Expected 2 lines (instance-id, job-id).")
    return lines[0].strip(), lines[1].strip()


def poll_until_stopped(instance_id: str) -> str:
    ec2 = boto3.client("ec2", region_name=TRAINING_REGION)
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
    final_state = poll_until_stopped(instance_id)

    if final_state == "terminated":
        print(f"\nInstance was terminated unexpectedly (spot interruption or manual action).")
        print(f"Check CloudTrail for termination reason.")
        raise SystemExit(1)

    status = check_status(job_id)
    print(f"Training status: {status}")

    if status.startswith("FAILED"):
        code = status.split(":", 1)[-1]
        print(f"\nTraining failed with exit code {code}.")
        print(f"Or SSM into instance {instance_id} and read /var/log/flux-training.log")
        raise SystemExit(1)   # 非零退出,便于 CI/自动化据此判定失败
    else:
        local_dir = download_results(job_id)
        print_summary(local_dir)
        print(f"\nOpen sample images: open {local_dir}")
