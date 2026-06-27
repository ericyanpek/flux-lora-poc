"""
Triggers a CodeBuild build and streams logs to the terminal in real time.
Replaces the local docker build + push workflow.
Run: python3 02_trigger_build.py
"""
import boto3
import time
import sys
from config import ACCOUNT, REGION, CODEBUILD_PROJECT, CW_LOG_GROUP


def start_build() -> str:
    cb = boto3.client("codebuild", region_name=REGION)
    resp = cb.start_build(projectName=CODEBUILD_PROJECT)
    build_id = resp["build"]["id"]
    print(f"✅ CodeBuild started: {build_id}")
    print(f"   Console: https://console.aws.amazon.com/codesuite/codebuild/{ACCOUNT}/projects/{CODEBUILD_PROJECT}/build/{build_id.split(':')[-1]}/log?region={REGION}")
    return build_id


def get_log_stream(build_id: str) -> str:
    cb = boto3.client("codebuild", region_name=REGION)
    for _ in range(30):
        resp = cb.batch_get_builds(ids=[build_id])
        build = resp["builds"][0]
        stream = build.get("logs", {}).get("streamName")
        if stream:
            return stream
        time.sleep(3)
    raise RuntimeError("CloudWatch log stream did not appear within 90s")


def stream_logs(build_id: str, log_stream: str):
    logs = boto3.client("logs", region_name=REGION)
    cb = boto3.client("codebuild", region_name=REGION)
    next_token = None
    terminal_statuses = {"SUCCEEDED", "FAILED", "FAULT", "STOPPED", "TIMED_OUT"}

    print(f"\n--- Build logs ---")
    while True:
        kwargs = {"logGroupName": CW_LOG_GROUP, "logStreamName": log_stream, "startFromHead": True}
        if next_token:
            kwargs["nextToken"] = next_token
        try:
            resp = logs.get_log_events(**kwargs)
            for event in resp["events"]:
                print(event["message"], end="" if event["message"].endswith("\n") else "\n")
            next_token = resp.get("nextForwardToken")
        except logs.exceptions.ResourceNotFoundException:
            pass

        build_resp = cb.batch_get_builds(ids=[build_id])
        status = build_resp["builds"][0]["buildStatus"]
        if status in terminal_statuses:
            # flush remaining logs after build ends
            for _ in range(6):
                time.sleep(3)
                flush_kwargs = {"logGroupName": CW_LOG_GROUP, "logStreamName": log_stream, "startFromHead": True}
                if next_token:
                    flush_kwargs["nextToken"] = next_token
                try:
                    flush_resp = logs.get_log_events(**flush_kwargs)
                    for event in flush_resp["events"]:
                        print(event["message"], end="" if event["message"].endswith("\n") else "\n")
                    new_token = flush_resp.get("nextForwardToken")
                    if new_token == next_token:
                        break
                    next_token = new_token
                except logs.exceptions.ResourceNotFoundException:
                    break
            return status

        time.sleep(5)


if __name__ == "__main__":
    build_id = start_build()
    print("Waiting for log stream...")
    log_stream = get_log_stream(build_id)
    final_status = stream_logs(build_id, log_stream)
    print(f"\n--- Build {final_status} ---")
    if final_status != "SUCCEEDED":
        sys.exit(1)
