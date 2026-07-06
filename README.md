# FLUX.2-dev LoRA 训练 / 推理平台

面向游戏美术图片制作的 FLUX.2-dev LoRA 微调与推理工程,部署于 AWS。训练与推理均已在单卡 L40S(46GB)上端到端跑通,并产出可复现的对照 Demo。

> **状态**:训练 ✅ · 分层 LoRA(Style + Character)✅ · 多层组合 ✅ · ComfyUI 独立推理 ✅ · 原生多参考图 🟡(设计+骨架,待 GPU 验证)
>
> ✅ 均有产物背书:分层/组合见 [`docs/experiments/layered-lora-results.md`](docs/experiments/layered-lora-results.md);推理见 [`poc/scripts/inference/comfy_gen.py`](poc/scripts/inference/comfy_gen.py)。🟡 表示代码就绪但未经 GPU 验证。

---

## 能力与结论

- **单 LoRA 风格迁移**:rank-32(390MB),1500 步训练,实测 1000 步收敛。目标美术风格迁移成功。
- **分层 LoRA**:同一数据集经两套 caption 策略,训出解耦程度可控的 Style 层(详细 caption → 学画风)与 Character 层(稀疏 caption → 学主体)。原理:*未写入 caption 的共有特征被编码进 LoRA*。两层统一 rank,便于加权组合。
- **多层组合**:推理侧串接 `LoraLoader` 加权叠加(Style 0.9 + Char 0.8),兼顾画风与主体。提供 base / style / char / combo × 多主题的对照矩阵,详见实验报告。
- **风格泛化**:训练集外的新主体同样继承目标美术风格,单一风格 LoRA 可复用于任意主体。
- **推理**:独立 ComfyUI 实例加载官方预量化 fp8 底模,单图约 35s(模型常驻)。ai-toolkit 训练的 bf16 LoRA 与 fp8 底模直接兼容,无 key mismatch 或 Float8 报错。

关键工程约束:FLUX.2-dev = 32B rectified-flow transformer + Mistral-Small-3.x-24B 文本编码器[^mistral](双模型,合计约 90GB)。L40S 标称 48GB,PyTorch 进程实际可用约 44.4GB,训练显存管理是本项目的核心工程挑战。

[^mistral]: 上游小版本命名不一致:diffusers 博客记为 3.1,BFL 官方 flux2 inference repo 实为 `Mistral-Small-3.2-24B-Instruct-2506`,diffusers 类名 `Mistral3ForConditionalGeneration` 不含小版本号。本文统一记为 3.x。

---

## 架构

```
本地 ctl.py  ──SSM──►  EC2 g6e.4xlarge (us-west-2, 长驻实例,stop/start 复用)
                          │
                          ├── docker pull ◄── ECR (us-east-1, CodeBuild 构建)
                          ├── 模型缓存 ◄────── EBS /opt/flux-cache/hf (持久,免重下 90GB)
                          ├── 数据集 ◄──────── S3 (us-east-1)
                          ├── docker run (ai-toolkit, arch:flux2)
                          │     ├── transformer fp8 CPU 量化 → 上 GPU (patch1)
                          │     ├── Mistral CPU 量化 → 缓存 text embedding
                          │     ├── prepare 前卸载 Mistral 到 CPU (patch2, 省 24GB)
                          │     ├── LoRA 训练 (GPU 仅存 transformer, ~36GB)
                          │     └── W&B 实时上报
                          └── s3 sync ──────► S3 outputs (LoRA + 样图)
```

训练与推理为独立实例,互不干扰。

| 项 | 选型 |
|----|------|
| 训练框架 | [ai-toolkit](https://github.com/ostris/ai-toolkit) + `arch: flux2`(专用 Flux2Model,commit pin) |
| 推理框架 | ComfyUI(官方预量化 fp8 文件,独立实例) |
| 镜像构建 | AWS CodeBuild(规避本地带宽与 arm64/amd64 差异) |
| 监控 | W&B project `flux2-lora-poc` |
| 密钥 | SSM Parameter Store(`/flux-poc/{hf-token, wandb-key, dockerhub}`) |

---

## 快速开始

```bash
# 0. 配置
cp poc/.env.example poc/.env               # AWS 账号 / region / subnet / AMI / 数据集 / 实例 ID

# 1. 基础设施(S3 + ECR + IAM + CodeBuild,一次性)
pip install boto3 python-dotenv
python3 poc/scripts/01_setup_infra.py

# 2. 构建训练镜像(CodeBuild,约 8–10 min)
python3 poc/scripts/02_trigger_build.py

# 3. 上传数据集
python3 poc/scripts/03_upload_dataset.py

# 4. 训练(ctl.py + 长驻实例)
python3 poc/scripts/ctl.py start           # 启动并等待 SSM 就绪
python3 poc/scripts/ctl.py train --layer style   # 分层训练:style / char
python3 poc/scripts/ctl.py logs train
python3 poc/scripts/ctl.py stop            # 停机(EBS + 模型缓存保留)

# 5. 推理(独立 ComfyUI 实例)
python3 poc/scripts/07_deploy_comfyui.py                                  # 部署,自动拉取最新 LoRA
python3 poc/scripts/inference/comfy_gen.py --config combo --out /exp/combo # 出图
```

`04_submit_training.py` / `05_monitor.py` 为早期单次运行脚本;推荐 `ctl.py` + 长驻实例(模型缓存在 EBS,start 后秒级可用)。

---

## 训练配置(已验证)

| 参数 | 值 | 说明 |
|------|-----|------|
| 架构入口 | `arch: flux2` | 专用 Flux2Model;`is_flux:true` 与 FLUX.2 不兼容 |
| 基础模型 | black-forest-labs/FLUX.2-dev | gated;Mistral 编码器亦 gated |
| LoRA rank | 32 | 各层统一,便于加权组合 |
| Steps | 1500(1000 收敛) | Character 层 1200 |
| 分辨率 | 640 | 降分辨率仅减激活显存,对权重占满型 OOM 无效 |
| 量化 | fp8 + low_vram | transformer / Mistral 均 CPU 量化后上 GPU(patch1) |
| 文本编码 | cache + unload | 缓存 embedding 后于 prepare 前卸载 Mistral(patch2) |
| 优化器 / LR | adamw8bit / 1e-4 cosine | |

**显存管理**:峰值出现在 `prepare_accelerator` 阶段——transformer 与 Mistral 同时驻留 GPU 达约 44GB,超 46GB 上限。两个 patch(见 [`patch_flux2_te.py`](poc/docker/patch_flux2_te.py))解决:① Mistral 在 CPU 量化后再上 GPU;② prepare 前将 Mistral 卸载至 CPU(已缓存 embedding,训练无需其常驻)。卸载后训练显存稳定于约 36GB。配套:`--shm-size=24g`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`、HF 缓存挂载 EBS。

---

## 推理

单卡 L40S 上,ai-toolkit loader 直接推理 FLUX.2 会 segfault / OOM。可靠路径为独立 ComfyUI 实例:

- 加载官方预量化 fp8 文件(transformer 34G + Mistral TE 17G + VAE),`--lowvram` 分时 offload,GPU 峰值 <42GB,25 步单图约 35s(模型常驻)。
- **LoRA 兼容性已验证**:ai-toolkit bf16 rank-32 LoRA 在 fp8 底模上直接加载生效,无 key mismatch / Float8 报错。
- 设计见 [`docs/superpowers/specs/2026-06-28-comfyui-inference-design.md`](docs/superpowers/specs/2026-06-28-comfyui-inference-design.md)。

**第二条互补主线(🟡)**:FLUX.2 底模原生支持多参考图(最多 10 张,无需训练),面向主体一致性(同一主体跨场景)。骨架 [`09_multiref_infer.py`](poc/scripts/inference/09_multiref_infer.py),设计见 [`2026-06-30-multiref-inference-design.md`](docs/superpowers/specs/2026-06-30-multiref-inference-design.md)。未经 GPU 验证。

**生产在线 API**:建议 g7e RTX PRO 6000 96GB 使编码器 + transformer + VAE 全常驻;缺货期以 L40S + SageMaker Async 队列过渡。详见双栈规划。

---

## 双栈架构规划

📐 [`docs/architecture/dual-stack-plan.md`](docs/architecture/dual-stack-plan.md)

- **文本编码器解耦**:将 Mistral-24B 拆为独立可调度单元,同解训练显存、推理显存与推理扩缩容。
- **产物协同**:W&B Artifacts(注册 + 血缘 + alias)+ S3 manifest + SSM 指针 + diffusers hotswap,全复用现有栈。
- **推理框架**:FastAPI + diffusers(MVP)→ Ray Serve / Triton(规模化);vLLM 不适用扩散模型。
- **扩缩容**:SageMaker Async(MVP)→ EKS + Karpenter + KEDA(规模化);HPA 不适用 GPU。

---

## 仓库结构

```
poc/
├── buildspec.yml             CodeBuild 构建脚本(含 Docker Hub 认证)
├── docker/
│   ├── Dockerfile            训练镜像(ai-toolkit pin 4e50535 + optimum-quanto 0.2.7)
│   ├── train_entry.py        ai-toolkit 配置生成 + 训练入口(arch:flux2)
│   └── patch_flux2_te.py     两个 patch:① Mistral CPU 量化后上 GPU ② prepare 前卸载 TE
└── scripts/
    ├── ctl.py                生命周期 CLI(start/stop/status/train --layer/logs)
    ├── 01_setup_infra.py     S3 + ECR + IAM(最小权限)+ CodeBuild
    ├── 02_trigger_build.py   触发 CodeBuild
    ├── 03_upload_dataset.py  上传数据集
    ├── 06_prepare_layers.py  分层:同批图生成 style / char 两套 caption
    ├── 07_compose_experiment.py  多层 LoRA 组合网格实验
    ├── 07_deploy_comfyui.py  部署独立 ComfyUI 推理实例(自动拉取最新 LoRA)
    ├── 08_demo_matrix.py     Demo 矩阵(diffusers 版,早期)
    ├── 04_submit_training.py / 05_monitor.py / 00_cleanup_sagemaker.py
    └── inference/
        ├── comfy_gen.py      ComfyUI API 出图(FLUX.2 fp8 + 分层 LoRA)
        ├── comfy_probe.py    探测 ComfyUI 节点 schema
        └── 09_multiref_infer.py  原生多参考图骨架(🟡)
docs/
├── architecture/dual-stack-plan.md        双栈端到端规划
├── experiments/layered-lora-results.md    分层/组合实验报告(含对照图)
└── superpowers/specs|plans/               设计文档与实施计划
```

---

## 工程要点

两条主要教训:

1. **架构入口**:FLUX.2-dev 必须走 ai-toolkit 专用 `Flux2Model`(`arch:flux2`),经 `torch.device("meta")` + `load_state_dict(assign=True)` 加载;通用 `is_flux:true` 路径(`from_pretrained()+.to()`)与其新架构不兼容。
2. **OOM 诊断**:差几百 MB 的 OOM 应先查"模型未及时卸载"。若 `allocated` 接近上限即权重占满型,降 batch / 分辨率 / grad-accum 等激活类优化无效;真正有效的是卸载非必需模型(如 prepare 阶段卸载文本编码器,一次释放 24GB)。

| 问题 | 解法 |
|------|------|
| SageMaker g6e 配额=0 | 改用 EC2(G/VT 配额独立) |
| us-east-1 g6e 缺货 | 切 us-west-2;DryRun 仅验权限不验容量 |
| FLUX.2-dev + Mistral gated | HF 网页分别申请 |
| 架构入口错误 | `arch: flux2`(非 `is_flux: true`) |
| optimum-quanto 0.2.4 + torch 2.6 fake-impl bug | 升 0.2.7,`--no-deps`(否则拉高 torch) |
| requirements 回退 quanto 0.2.4 | 先装 requirements 再 force-reinstall 0.2.7 |
| Mistral 加载 OOM | patch1:CPU 量化(48→24GB)后上 GPU |
| prepare 阶段 OOM(权重占满型) | patch2:prepare 前卸载文本编码器(省 ~24GB) |
| DataLoader Bus error | `--shm-size=24g`(默认 64MB 不足) |
| 多卡救单卡 OOM | 不可行:ai-toolkit FLUX.2 仅数据并行,`split_model_over_gpus` 锁 FLUX.1 |
| EBS 容量不足 | 扩 350GB + 模型缓存持久化至 `/opt/flux-cache/hf` |
| CodeBuild Docker Hub 429 | buildspec 加认证(token 存 SSM) |
| 本地 arm64 推 ECR 架构错 | 用 CodeBuild(amd64)构建 |

---

## 基础设施

```
S3         flux-poc-<account>-us-east-1/   (datasets/ outputs/ checkpoints/ demo/)
ECR        flux-poc-training:latest        (CodeBuild)
CodeBuild  flux-poc-build                  (us-east-1)
EC2        g6e.4xlarge (us-west-2)         训练长驻 + 推理独立实例
IAM        flux-poc-ec2-role / flux-poc-codebuild-role
SSM        /flux-poc/{hf-token, wandb-key, dockerhub}  (SecureString)
W&B        project flux2-lora-poc
```

成本:停机后 GPU 不计费;EBS 350GB 约 $28/月(换取免重下 90GB)。可将模型缓存做成 EBS snapshot 后删卷进一步节省。
