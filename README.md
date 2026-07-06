# FLUX.2-dev LoRA 训练 / 推理平台

在 AWS 上对 FLUX.2-dev 做 LoRA 微调并提供推理的端到端工程。训练 POC 已跑通,产出可用的游戏 slots 美术风格 LoRA;推理与生产化架构已规划。

> **状态**:训练栈 ✅ 跑通 · 分层 LoRA ✅ Style+Character 双层训成 · 多层组合 Demo 🚧 生成中 · 推理栈 📋 已设计
> **目标**:从手工 POC 演进为生产级双栈(训练 × 在线推理 × LoRA 产物协同)

## 最新进展(2026-07-06)

- **分层 LoRA 跑通**:同一批 18 张 slots 图,用两套 caption 策略训出解耦的 **Style LoRA**(`slotstyle`,详细 caption→学画风)+ **Character LoRA**(`slotchar`,稀疏 caption→学角色身份),各 390MB rank-32。
- **单卡 46GB OOM 根因修复(关键)**:FLUX.2 训练在 `prepare_accelerator` 阶段稳定 OOM(差 ~200MB)。根因是 ai-toolkit 新 commit 只在采样步卸载文本编码器,导致 prepare 时 transformer+Mistral 同时在 GPU(44GB)。修复:patch 在 prepare 前把 TE 卸载到 CPU(`patch_flux2_te.py` PATCH 2),释放 ~24GB。训练显存降到 36GB,稳定跑通。**降分辨率/降 grad-accum 均无效**——纯权重占满型 OOM 只能靠卸载模型解决。详见 memory `reference-gpu-vram-optimization`。
- **多卡不可行**:ai-toolkit FLUX.2 路径不支持模型并行(`split_model_over_gpus` 硬锁 FLUX.1),多卡只是数据并行,救不了单卡 OOM。
- **Demo 生成矩阵**(`08_demo_matrix.py`):单层对照(base/style-only/char-only/combo)× 主题(海盗/龙/自定义美人鱼 IP)× 权重梯度,展示分层 LoRA 微调的特点与优势。

---

## 已验证的结论(POC 实况)

- **模型**:FLUX.2-dev = 32B rectified-flow transformer + **Mistral-Small-3.1-24B** 文本编码器(两个独立模型,共 ~90GB 下载)
- **硬件**:EC2 **g6e.4xlarge**(L40S **46GB** + 128GB RAM,us-west-2)。L40S 46GB 对 FLUX.2 训练和推理都极度吃紧
- **产物**:390MB rank-32 safetensors LoRA(触发词 `SLOTIP`),slots 美术风格迁移**成功**
- **训练收敛**:1500 步训完,实测 **1000 步即收敛**(标题艺术字+装饰+UI 构图全到位),1500 步已饱和

---

## 架构

```
本地 ctl.py  ──SSM──►  EC2 g6e.4xlarge (us-west-2, 长驻实例 stop/start 复用)
                          │
                          ├── docker pull ◄── ECR (us-east-1, 镜像由 CodeBuild 构建)
                          ├── 模型缓存 ◄────── EBS /opt/flux-cache/hf (106GB, 持久, 免重下)
                          ├── 数据集 ◄──────── S3 (us-east-1)
                          ├── docker run (ai-toolkit, arch:flux2)
                          │     ├── transformer fp8 CPU 量化(patch)
                          │     ├── Mistral CPU 量化 → 缓存 text embedding → 卸载编码器
                          │     ├── LoRA 训练(768 分辨率, skip_first_sample)
                          │     └── W&B 实时上报
                          └── s3 sync ──────► S3 outputs (LoRA + 样图)
```

**训练框架**:[ai-toolkit](https://github.com/ostris/ai-toolkit) **main 分支** + `arch: flux2`(专用 Flux2Model)
**镜像构建**:AWS CodeBuild(避开本地带宽 + arm64/amd64 架构问题)
**监控**:W&B project `flux2-lora-poc`
**密钥**:SSM Parameter Store(`/flux-poc/hf-token`、`/flux-poc/wandb-key`、`/flux-poc/dockerhub`)

---

## 快速开始

```bash
# 0. 配置
cp poc/.env.example poc/.env   # 填 AWS 账号/region/subnet/AMI/数据集/实例ID

# 1. 一次性基础设施(S3 + ECR + IAM + CodeBuild)
pip install boto3 python-dotenv
python3 poc/scripts/01_setup_infra.py

# 2. 构建训练镜像(CodeBuild,约 8-10 分钟,需先配 Docker Hub 认证到 SSM)
python3 poc/scripts/02_trigger_build.py

# 3. 上传训练数据集到 S3
python3 poc/scripts/03_upload_dataset.py

# 4. 用 ctl.py 管理实例生命周期(推荐)
python3 poc/scripts/ctl.py start          # 启动长驻实例(等 SSM 就绪)
python3 poc/scripts/ctl.py train          # 后台跑训练(默认 1500 步)
python3 poc/scripts/ctl.py logs train     # 看训练日志
python3 poc/scripts/ctl.py status         # GPU/磁盘/缓存/容器状态
python3 poc/scripts/ctl.py stop           # 停机省钱(EBS+模型缓存保留)
```

`04_submit_training.py` / `05_monitor.py` 是早期"起新实例跑一次"的脚本;现在推荐用 **`ctl.py` + 长驻实例**(模型缓存在 EBS,start 后秒级可用,免重下 90GB)。

---

## 训练配置(已验证可用)

| 参数 | 值 | 说明 |
|------|-----|------|
| 架构入口 | **`arch: flux2`** | 关键:不是 `is_flux:true`(那个对 FLUX.2 不兼容) |
| 基础模型 | black-forest-labs/FLUX.2-dev | gated,需 HF 申请;Mistral-Small-3.1-24B 也 gated |
| LoRA rank | 32 | |
| Steps | 1500(1000 即收敛) | |
| 分辨率 | 768 | 降自 1024,省显存挤进 46GB |
| 量化 | fp8 + low_vram | transformer/Mistral 均 CPU 量化后上 GPU |
| 文本编码 | cache + unload | 编码完卸载 Mistral,训练时只剩 transformer |
| 优化器 / LR | adamw8bit / 1e-4 cosine | |

显存关键 docker 参数:`--shm-size=24g`(DataLoader)、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`(碎片)、HF 缓存挂 EBS。

---

## 推理(规划中)

L40S 46GB 单卡推理在 VAE 解码 + Mistral 编码叠加时逼近 46GB,**只够异步/POC**。两条路:

1. **ComfyUI 独立推理机**(最快验证,详见 `docs/superpowers/specs/2026-06-28-comfyui-inference-design.md`)——用 ComfyUI 官方预量化 fp8 文件,加载一次常驻,出图秒级,避开训练入口每次 ~13 分钟的运行时量化
2. **生产在线 API**(详见双栈规划)——建议上 **g7e RTX PRO 6000 96GB**,让 TE+transformer+VAE 全常驻;Blackwell 缺货期用 L40S + SageMaker Async 队列过渡

---

## 双栈架构规划

完整的训练 × 推理 × 协同端到端规划(基于权威工程实践调研,带出处):

📐 **[docs/architecture/dual-stack-plan.md](docs/architecture/dual-stack-plan.md)**

核心结论:
- **把 Mistral-24B 文本编码器拆成独立可调度单元** —— 同解训练显存、推理显存、推理扩缩容三个问题
- **协同**:W&B Artifacts(注册+血缘+`production` alias)+ S3 manifest + SSM 指针 + diffusers hotswap,全复用现有栈,不引入重型 Registry
- **推理框架**:FastAPI+diffusers(MVP)→ Ray Serve / Triton(规模化);**vLLM 不适用扩散模型**
- **扩缩容**:SageMaker Async(MVP)→ EKS + Karpenter + KEDA(规模化);HPA 不适用 GPU

---

## 仓库结构

```
poc/
├── .env.example              配置模板(账号/region/subnet/AMI/实例ID)
├── buildspec.yml             CodeBuild 构建脚本(含 Docker Hub 认证)
├── docker/
│   ├── Dockerfile            训练镜像(DLAMI base + ai-toolkit main + optimum-quanto 0.2.7)
│   ├── train_entry.py        ai-toolkit 配置生成 + 训练入口(arch:flux2)
│   └── patch_flux2_te.py     patch:Mistral 编码器 CPU 量化后再上 GPU(避 OOM)
└── scripts/
    ├── config.py             共享配置(从 .env 读)
    ├── ctl.py                ⭐ 生命周期 CLI(start/stop/status/train/logs/run)
    ├── 00_cleanup_sagemaker.py
    ├── 01_setup_infra.py     S3 + ECR + IAM + CodeBuild
    ├── 02_trigger_build.py   触发 CodeBuild
    ├── 03_upload_dataset.py
    ├── 04_submit_training.py 起新实例跑训练(早期方式)
    └── 05_monitor.py
docs/
├── architecture/dual-stack-plan.md          ⭐ 双栈端到端规划
└── superpowers/specs|plans/                  设计文档与实施计划
```

---

## 踩坑记录(13 项,完整版见 memory)

核心教训:**用错架构入口(`is_flux:true` vs `arch:flux2`)白费 8 轮调试**。FLUX.2-dev 必须用 ai-toolkit 的专用 `Flux2Model`(`arch:flux2`),它用 `torch.device("meta")` + `load_state_dict(assign=True)` 正确加载;通用 `is_flux` 路径用 `from_pretrained()+.to()` 与 FLUX.2 新架构不兼容。

| 问题 | 解法 |
|------|------|
| SageMaker g6e 配额=0 | 改用 EC2(G/VT 配额独立宽松) |
| us-east-1 全区 g6e 缺货 | 切 us-west-2;DryRun 只验权限不验容量 |
| FLUX.2-dev + Mistral 都是 gated | HF 网页各自申请 |
| **用错架构入口** | `arch: flux2`(非 `is_flux: true`)|
| optimum-quanto 0.2.4 + torch 2.6 fake-impl bug | 升 0.2.7,用 `--no-deps`(否则 torch 被拉到 2.12)|
| requirements 把 quanto 降回 0.2.4 | 先装 requirements 再 force-reinstall 0.2.7 |
| Mistral 加载 OOM | patch CPU 量化 + cache embedding + 卸载编码器 |
| prepare 阶段差 324MB OOM | skip_first_sample + 768 分辨率 |
| DataLoader Bus error | `--shm-size=24g` |
| EBS 装不下双模型+缓存 | 扩 EBS 350GB + 模型缓存持久化到 `/opt/flux-cache/hf` |
| CodeBuild Docker Hub 429 限速 | buildspec 加 Docker Hub 认证(SSM 存 token)|
| 本地 arm64 推 ECR 架构错 | 用 CodeBuild(amd64)构建 |

---

## 基础设施

```
S3:        flux-poc-<account>-us-east-1/  (datasets/ outputs/ checkpoints/)
ECR:       flux-poc-training:latest        (CodeBuild 构建)
CodeBuild: flux-poc-build                   (us-east-1)
EC2:       g6e.4xlarge (us-west-2)          长驻, EBS 350GB + 106G 模型缓存
IAM:       flux-poc-ec2-role / flux-poc-codebuild-role
SSM:       /flux-poc/{hf-token, wandb-key, dockerhub}  (SecureString)
W&B:       project flux2-lora-poc
```

成本提示:停机后 GPU 不计费,但 EBS 350GB 约 ~$28/月持续计费(换"免重下 90GB")。彻底省钱可把模型缓存做成 EBS snapshot 后删卷。
