# FLUX.2-dev LoRA 训练 POC

在 AWS EC2 g6e.2xlarge（L40S 48GB）上对 FLUX.2-dev 进行 LoRA 微调的端到端 POC。

## 目标

验证 FLUX.2-dev（Gemma 3 12B 文本编码器）的 LoRA 微调是否能在单卡 L40S 上完成，并与 FLUX.1-dev 的生成质量做对比。

## 架构

```
本地脚本                    AWS
04_submit_training.py
  ├── 写 token 到 SSM ──→  SSM Parameter Store (us-east-1)
  └── 启动 EC2 实例 ──────→ g6e.2xlarge (us-west-2)
                               │
                               ├── docker pull ←── ECR (us-east-1)
                               ├── s3 sync ←─────── S3 数据集 (us-east-1)
                               ├── docker run
                               │     └── ai-toolkit (pinned: 75781fb)
                               │           ├── FLUX.2-dev 模型下载 (~64GB)
                               │           ├── fp8 量化 + LoRA 训练
                               │           └── W&B 实时上报
                               └── s3 sync ──────→ S3 结果 (us-east-1)
```

**训练框架：** [ai-toolkit](https://github.com/ostris/ai-toolkit) `75781fb`（最后已知支持 FLUX.2-dev 的版本）

**监控：** W&B project `flux2-lora-poc`，实时 loss 曲线 + 每 250 步生成样图

## 快速开始

```bash
# 1. 配置环境
cp poc/.env.example poc/.env
# 编辑 .env，填入 AWS 账号、region、subnet、AMI 等

# 2. 一次性基础设施初始化
pip install boto3 python-dotenv
python3 poc/scripts/01_setup_infra.py

# 3. 构建并推送训练镜像（约 15 分钟）
bash poc/scripts/02_build_push.sh

# 4. 上传训练数据集
python3 poc/scripts/03_upload_dataset.py

# 5. 启动训练（HF_TOKEN 需申请 FLUX.2-dev 访问权限）
python3 poc/scripts/04_submit_training.py \
  --hf-token hf_xxx \
  --wandb-key wbk_xxx
```

训练完成后实例自动关机，结果上传到 S3：

```bash
python3 poc/scripts/05_monitor.py   # 轮询状态，完成后下载 LoRA 权重和样图
```

## 训练配置

| 参数 | 值 |
|------|-----|
| 基础模型 | black-forest-labs/FLUX.2-dev |
| LoRA rank | 32 |
| Steps | 1500 |
| Batch size | 1（gradient accumulation 4） |
| 优化器 | adamw8bit |
| LR | 1e-4，cosine decay |
| 量化 | fp8，low_vram 模式 |

## POC 结果

> **TODO**

- [ ] 完成首次完整训练（1500 steps）
- [ ] 对比 trigger word 有/无的生成差异（A/B 样图）
- [ ] 与 FLUX.1-dev 同参数训练结果质量对比
- [ ] 记录实际训练耗时和显存峰值
- [ ] 评估 LoRA 权重在 ComfyUI 中的加载兼容性

## 踩坑记录

| 问题 | 解法 |
|------|------|
| SageMaker g6e 配额为 0 | 改用 EC2，G/VT 配额独立且充足 |
| us-east-1 全区域 g6e 无库存 | 切换到 us-west-2 |
| FLUX.2-dev 是 gated repo | HF 网页单独申请访问权限 |
| ai-toolkit `low_vram=False` meta tensor crash | FLUX.2-dev 的 from_pretrained 使用 meta device，必须 `low_vram=True` |
| ai-toolkit main 分支 optimum-quanto 兼容性回归 | 锁定到 commit `75781fb` |

## 基础设施

```
S3:     flux-poc-<account>-us-east-1/
        ├── datasets/          训练图片 + caption
        ├── outputs/           训练结果（LoRA 权重 + 样图）
        └── checkpoints/       中间检查点
ECR:    flux-poc-training:latest
IAM:    flux-poc-ec2-role + flux-poc-ec2-instance-profile
SSM:    /flux-poc/hf-token, /flux-poc/wandb-key（SecureString）
```
