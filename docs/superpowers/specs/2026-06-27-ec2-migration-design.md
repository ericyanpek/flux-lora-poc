# POC 迁移：SageMaker → EC2 g6e.2xlarge

## 目标

将 FLUX.2-dev LoRA 训练从 SageMaker（无 g6e 配额）迁移到 EC2 g6e.2xlarge（L40S 48GB），同时清理 SageMaker 遗留资源，并接入 W&B 可视化监控。

## 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `scripts/00_cleanup_sagemaker.py` | 新增 | 删除 ECR 镜像（省存储费）、撤销 SageMaker IAM inline policy |
| `scripts/01_setup_infra.py` | 更新 | 移除 SM role patch，改为创建 EC2 instance profile |
| `scripts/02_build_push.sh` | 不变 | 复用 |
| `scripts/03_upload_dataset.py` | 不变 | 复用 |
| `scripts/04_submit_training.py` | 替换 | 启动 g6e.2xlarge，注入 UserData |
| `scripts/05_monitor.py` | 替换 | 轮询 EC2 state，成功后下载 S3 结果 |
| `docker/train_entry.py` | 更新 | 换 FLUX.2-dev、rank 32、加 W&B |
| `docker/Dockerfile` | 更新 | 加 `pip install wandb` |

## 基础设施

**保留**
- S3 bucket `flux-poc-984072314535-us-east-1`（数据集 + 输出路径不变）
- ECR repo `flux-poc-training`（镜像继续推这里）

**新增**
- IAM role `flux-poc-ec2-role` + instance profile `flux-poc-ec2-instance-profile`
  - `AmazonSSMManagedInstanceCore`（SSM Agent，免 SSH 急救入口）
  - inline policy：S3 read/write on `flux-poc-*`，ECR pull

**清理**
- ECR 中的旧镜像（保留 latest，删除其余 digest）
- SageMaker role 上的 `flux-poc-s3-access` inline policy

## EC2 启动配置

```
InstanceType: ml.g6e.2xlarge → ec2: g6e.2xlarge
AMI: 复用 emotion-companion-dev 的 ami-012ba162b9cd2729c（Amazon Linux 2 with GPU driver）
Subnet: subnet-0aa203a61adbcd8be（us-east-1a，与现有 g5 相同）
SecurityGroup: 新建 flux-poc-training-sg（仅出站，无入站，SSM 不需要开口）
VolumeSize: 200GB gp3（模型 ~50GB + 工具链 ~12GB + 余量）
```

## UserData 流程

```
1. yum install docker awscli -y && systemctl start docker
2. ECR docker login
3. aws s3 sync s3://.../datasets/poc-character-v1/ /tmp/training-data/
4. docker pull <ECR_URI>:latest
5. docker run --gpus all -e HF_TOKEN -e WANDB_API_KEY \
       -v /tmp/training-data:/opt/ml/input/data/training \
       -v /tmp/output:/opt/ml/model \
       <ECR_URI>:latest
6. 获取 docker 退出码；写 S3 marker：outputs/<job-id>/status.txt（"SUCCESS" 或 "FAILED:<code>"）
7. aws s3 sync /tmp/output/ s3://.../outputs/<job-id>/
8. shutdown -h now
```

job-id 格式：`flux2-lora-ec2-YYYYMMDD-HHMMSS`，启动时生成，写入实例 Tag `flux:job-id`。

## train_entry.py 变更

```python
# 模型
model_name = "black-forest-labs/FLUX.2-dev"

# LoRA rank（L40S 48GB 有足够余量）
rank = int(hp.get("rank", "32"))

# W&B（可选，无 key 时降级为 EmptyLogger）
config["config"]["process"][0]["logging"] = {
    "use_wandb": bool(wandb_key),
    "project": "flux2-lora-poc",
    "run_name": trigger_word,
}
```

`WANDB_API_KEY` 通过环境变量注入，`train_entry.py` 读 `os.environ.get("WANDB_API_KEY", "")`，空字符串时 `use_wandb: False`，不报错。

## 监控流程（05_monitor.py）

1. 读 `/tmp/last_flux_ec2_job.txt` 获取 instance-id + job-id
2. 每 60 秒 `describe-instances` 查 state
3. state = `stopped` 时，读 `s3://.../outputs/<job-id>/status.txt`
4. SUCCESS → 下载结果到 `poc/results/<job-id>/`；FAILED → 打印错误码并提示去 CloudWatch 查日志

## 调用方式（与现在保持一致）

```bash
# 1. 一次性：创建 IAM profile（已有则跳过）
python3 01_setup_infra.py

# 2. 清理 SageMaker 遗留（一次性）
python3 00_cleanup_sagemaker.py

# 3. 重建并推送镜像（模型/代码有变化时）
bash 02_build_push.sh

# 4. 启动训练
python3 04_submit_training.py --hf-token hf_xxx --wandb-key wbk_xxx

# 5. 监控 + 下载结果
python3 05_monitor.py
```

## 不在本次范围内

- 多 GPU / 分布式训练
- Spot 实例（g6e spot 配额也是 0）
- 自动重试失败任务
