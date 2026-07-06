# CodeBuild 迁移设计：本地 docker build → AWS CodeBuild

## 目标

将 Docker 镜像构建从本地 Mac（25-30 MB/s）迁移到 AWS CodeBuild（AWS 内网带宽），缩短 build 时间，保持与现有工作流一致的操作体验。

## 文件变更

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `poc/buildspec.yml` | 新增 | CodeBuild 执行脚本：docker build + push |
| `poc/scripts/01_setup_infra.py` | 更新 | 增加 `create_codebuild_project()` |
| `poc/scripts/02_build_push.sh` | 替换为 `02_trigger_build.py` | 触发 CodeBuild，实时流日志，等待完成 |
| `poc/scripts/config.py` | 更新 | 加 `CODEBUILD_PROJECT = "flux-poc-build"` |

## 流程

```
本地
  python3 02_trigger_build.py
    ├── codebuild:start-build
    └── 轮询 CloudWatch Logs，实时打印到终端
                    ↓
              CodeBuild (us-east-1)
                ├── git clone ericyanpek/flux-lora-poc
                ├── docker build --platform linux/amd64 poc/docker/
                └── docker push → ECR flux-poc-training:latest
```

## IAM

新增 `flux-poc-codebuild-role`，trust policy 指向 `codebuild.amazonaws.com`：
- `ecr:GetAuthorizationToken`、`ecr:BatchGetImage`、`ecr:InitiateLayerUpload`、`ecr:UploadLayerPart`、`ecr:CompleteLayerUpload`、`ecr:PutImage`（ECR push）
- `logs:CreateLogGroup`、`logs:CreateLogStream`、`logs:PutLogEvents`（CloudWatch Logs write）
- `s3:GetObject`、`s3:PutObject` on `codepipeline-*`（CodeBuild artifact cache，可选）

## CodeBuild Project

```
name:            flux-poc-build
source:          GITHUB — ericyanpek/flux-lora-poc，buildspec: poc/buildspec.yml
environment:     BUILD_GENERAL1_LARGE (4 vCPU / 7GB RAM)
                 image: aws/codebuild/standard:7.0
                 privileged: true  (docker-in-docker)
artifacts:       NO_ARTIFACTS
logs:            CloudWatch Logs — group: /codebuild/flux-poc-build
```

## buildspec.yml

```yaml
version: 0.2
env:
  variables:
    AWS_ACCOUNT: ""    # 由 CodeBuild 环境变量注入（在 project 里配置）
    AWS_REGION: ""
phases:
  pre_build:
    commands:
      - aws ecr get-login-password --region $AWS_REGION |
          docker login --username AWS --password-stdin
          $AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com
  build:
    commands:
      - docker build --platform linux/amd64
          -t $AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/flux-poc-training:latest
          poc/docker/
  post_build:
    commands:
      - docker push $AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/flux-poc-training:latest
      - echo "Image pushed"
```

## 02_trigger_build.py 行为

1. 调用 `codebuild:start-build`，传入 project name
2. 获取 build ID，找到对应的 CloudWatch Log stream
3. 每 5 秒轮询一次，将新日志行打印到终端（实时追尾效果）
4. build 完成（SUCCEEDED / FAILED / STOPPED）后打印结果并退出

调用方式与现在一致，直接替换：
```bash
python3 poc/scripts/02_trigger_build.py
```

## 不在本次范围内

- GitHub webhook 自动触发
- 多架构 build（arm64）
- build cache（S3）
- CodePipeline 集成
