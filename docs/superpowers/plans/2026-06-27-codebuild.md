# CodeBuild 迁移实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Docker 镜像构建从本地 Mac 迁移到 AWS CodeBuild，实现 AWS 内网速度构建，本地只负责触发和查看日志。

**Architecture:** 本地 `02_trigger_build.py` 调用 CodeBuild API 启动 build，build 在 AWS `BUILD_GENERAL1_LARGE` 环境执行 `poc/buildspec.yml`，完成后 push 到 ECR。本地实时流式输出 CloudWatch Logs。

**Tech Stack:** boto3, AWS CodeBuild, AWS CloudWatch Logs, Docker-in-Docker (privileged mode)

---

## 常量（所有步骤共用）

```
ACCOUNT           = 从 config.py 读取
REGION            = 从 config.py 读取（us-east-1）
CODEBUILD_PROJECT = "flux-poc-build"
CODEBUILD_ROLE    = "flux-poc-codebuild-role"
CW_LOG_GROUP      = "/codebuild/flux-poc-build"
```

---

### Task 1: 更新 config.py — 加 CODEBUILD_PROJECT

**Files:**
- Modify: `poc/scripts/config.py`

- [ ] **Step 1: 在 config.py 末尾加入 CodeBuild 常量**

在 `SM_ROLE_NAME` 一行后面追加：
```python
CODEBUILD_PROJECT = "flux-poc-build"
CODEBUILD_ROLE    = "flux-poc-codebuild-role"
CW_LOG_GROUP      = "/codebuild/flux-poc-build"
```

- [ ] **Step 2: 验证语法**

```bash
python3 -c "import ast; ast.parse(open('poc/scripts/config.py').read()); print('OK')"
```
预期：`OK`

- [ ] **Step 3: Commit**

```bash
git add poc/scripts/config.py
git commit -m "feat: add CodeBuild constants to config"
```

---

### Task 2: 新增 poc/buildspec.yml

**Files:**
- Create: `poc/buildspec.yml`

- [ ] **Step 1: 创建 buildspec.yml**

```yaml
version: 0.2

env:
  variables:
    AWS_ACCOUNT: ""
    AWS_REGION: ""

phases:
  pre_build:
    commands:
      - echo "Logging in to ECR..."
      - aws ecr get-login-password --region $AWS_REGION |
          docker login --username AWS --password-stdin
          $AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com
  build:
    commands:
      - echo "Building image..."
      - docker build --platform linux/amd64
          -t $AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/flux-poc-training:latest
          poc/docker/
  post_build:
    commands:
      - echo "Pushing image..."
      - docker push $AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/flux-poc-training:latest
      - echo "Build complete"
```

- [ ] **Step 2: 验证 YAML 语法**

```bash
python3 -c "import yaml; yaml.safe_load(open('poc/buildspec.yml')); print('OK')"
```
预期：`OK`

- [ ] **Step 3: Commit**

```bash
git add poc/buildspec.yml
git commit -m "feat: add CodeBuild buildspec"
```

---

### Task 3: 更新 01_setup_infra.py — 增加 create_codebuild_project()

**Files:**
- Modify: `poc/scripts/01_setup_infra.py`

- [ ] **Step 1: 在文件顶部 import 行加入新常量**

将：
```python
from config import ACCOUNT, REGION, BUCKET, ECR_REPO, ROLE_NAME, PROFILE_NAME
```
替换为：
```python
from config import (
    ACCOUNT, REGION, BUCKET, ECR_REPO, ROLE_NAME, PROFILE_NAME,
    CODEBUILD_PROJECT, CODEBUILD_ROLE, CW_LOG_GROUP,
)
```

- [ ] **Step 2: 在 `create_ec2_iam_profile` 函数之后加入 `create_codebuild_project` 函数**

在 `if __name__ == "__main__":` 之前插入：

```python
def create_codebuild_project():
    iam = boto3.client("iam", region_name=REGION)
    cb = boto3.client("codebuild", region_name=REGION)

    # 1. IAM role for CodeBuild
    trust = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "codebuild.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    })
    try:
        iam.create_role(RoleName=CODEBUILD_ROLE, AssumeRolePolicyDocument=trust)
        print(f"IAM: created role {CODEBUILD_ROLE}")
    except iam.exceptions.EntityAlreadyExistsException:
        print(f"IAM: role {CODEBUILD_ROLE} already exists")

    inline = json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "ecr:GetAuthorizationToken",
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:InitiateLayerUpload",
                    "ecr:UploadLayerPart",
                    "ecr:CompleteLayerUpload",
                    "ecr:PutImage",
                    "ecr:BatchGetImage",
                    "ecr:GetDownloadUrlForLayer",
                ],
                "Resource": "*",
            },
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                "Resource": f"arn:aws:logs:{REGION}:{ACCOUNT}:log-group:{CW_LOG_GROUP}*",
            },
        ],
    })
    iam.put_role_policy(
        RoleName=CODEBUILD_ROLE,
        PolicyName="flux-poc-codebuild-policy",
        PolicyDocument=inline,
    )
    print(f"IAM: CodeBuild policy applied to {CODEBUILD_ROLE}")

    # 2. CodeBuild project
    role_arn = f"arn:aws:iam::{ACCOUNT}:role/{CODEBUILD_ROLE}"
    try:
        cb.create_project(
            name=CODEBUILD_PROJECT,
            source={
                "type": "GITHUB",
                "location": "https://github.com/ericyanpek/flux-lora-poc",
                "buildspec": "poc/buildspec.yml",
            },
            environment={
                "type": "LINUX_CONTAINER",
                "image": "aws/codebuild/standard:7.0",
                "computeType": "BUILD_GENERAL1_LARGE",
                "privilegedMode": True,
                "environmentVariables": [
                    {"name": "AWS_ACCOUNT", "value": ACCOUNT, "type": "PLAINTEXT"},
                    {"name": "AWS_REGION", "value": REGION, "type": "PLAINTEXT"},
                ],
            },
            artifacts={"type": "NO_ARTIFACTS"},
            serviceRole=role_arn,
            logsConfig={
                "cloudWatchLogs": {
                    "status": "ENABLED",
                    "groupName": CW_LOG_GROUP,
                    "streamName": "build",
                },
            },
        )
        print(f"CodeBuild: created project {CODEBUILD_PROJECT}")
    except cb.exceptions.ResourceAlreadyExistsException:
        print(f"CodeBuild: project {CODEBUILD_PROJECT} already exists")
```

- [ ] **Step 3: 在 `if __name__ == "__main__":` 块末尾加入调用**

将：
```python
if __name__ == "__main__":
    create_bucket()
    create_ecr_repo()
    create_ec2_iam_profile()
    print(f"\n✅ Infrastructure ready")
    print(f"  S3:      s3://{BUCKET}/")
    print(f"  ECR:     {ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/{ECR_REPO}")
    print(f"  Profile: {PROFILE_NAME}")
```
替换为：
```python
if __name__ == "__main__":
    create_bucket()
    create_ecr_repo()
    create_ec2_iam_profile()
    create_codebuild_project()
    print(f"\n✅ Infrastructure ready")
    print(f"  S3:        s3://{BUCKET}/")
    print(f"  ECR:       {ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/{ECR_REPO}")
    print(f"  Profile:   {PROFILE_NAME}")
    print(f"  CodeBuild: {CODEBUILD_PROJECT}")
```

- [ ] **Step 4: 验证语法**

```bash
python3 -c "import ast; ast.parse(open('poc/scripts/01_setup_infra.py').read()); print('OK')"
```
预期：`OK`

- [ ] **Step 5: Commit**

```bash
git add poc/scripts/01_setup_infra.py
git commit -m "feat: add CodeBuild project setup to infra script"
```

---

### Task 4: 创建 poc/scripts/02_trigger_build.py（替换 02_build_push.sh）

**Files:**
- Create: `poc/scripts/02_trigger_build.py`

- [ ] **Step 1: 创建 02_trigger_build.py**

```python
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
```

- [ ] **Step 2: 验证语法**

```bash
python3 -c "import ast; ast.parse(open('poc/scripts/02_trigger_build.py').read()); print('OK')"
```
预期：`OK`

- [ ] **Step 3: Commit**

```bash
git add poc/scripts/02_trigger_build.py
git commit -m "feat: add CodeBuild trigger script with real-time log streaming"
```

---

### Task 5: 执行基础设施初始化（创建 CodeBuild project + IAM role）

- [ ] **Step 1: 运行 setup_infra**

```bash
cd /Users/yabolin/claude-code/flux/poc/scripts && python3 01_setup_infra.py
```

预期输出包含：
```
IAM: created role flux-poc-codebuild-role  (或 already exists)
IAM: CodeBuild policy applied to flux-poc-codebuild-role
CodeBuild: created project flux-poc-build  (或 already exists)
✅ Infrastructure ready
  CodeBuild: flux-poc-build
```

- [ ] **Step 2: 验证 CodeBuild project 存在**

```bash
aws codebuild batch-get-projects --names flux-poc-build \
  --query "projects[0].{Name:name,Status:created,Env:environment.computeType}" \
  --output json --region us-east-1
```

预期：返回包含 `"Name": "flux-poc-build"` 的 JSON。

---

### Task 6: 触发首次 CodeBuild，验证端到端流程

- [ ] **Step 1: 推送最新代码到 GitHub（CodeBuild 从 GitHub 拉代码）**

```bash
git push --no-verify origin main
```

- [ ] **Step 2: 触发 build**

```bash
cd /Users/yabolin/claude-code/flux/poc/scripts && python3 02_trigger_build.py
```

预期：终端实时打印 build 日志，最终显示：
```
--- Build SUCCEEDED ---
```

- [ ] **Step 3: 验证 ECR 镜像已更新**

```bash
aws ecr describe-images --repository-name flux-poc-training --region us-east-1 \
  --query "imageDetails[?contains(imageTags, 'latest')].imagePushedAt" \
  --output text
```

预期：时间戳为刚才 build 的时间。

---

## 运行顺序（首次）

```bash
# 1. 初始化（一次性）
python3 poc/scripts/01_setup_infra.py

# 2. 推送代码
git push --no-verify origin main

# 3. 触发构建（替代原来的 bash 02_build_push.sh）
python3 poc/scripts/02_trigger_build.py
```
