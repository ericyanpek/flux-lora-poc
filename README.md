# FLUX.2-dev LoRA 训练 / 推理平台

在 AWS 上对 FLUX.2-dev 做 LoRA 微调并提供推理的端到端工程。训练 POC 已跑通,产出可用的游戏 slots 美术风格 LoRA;推理与生产化架构已规划。

> **状态**:训练栈 ✅ 跑通 · 分层 LoRA ✅ Style+Character 双层 · 多层组合 Demo ✅ · 推理栈 📋 已设计
> **目标**:从手工 POC 演进为生产级双栈(训练 × 在线推理 × LoRA 产物协同)

---

## 已验证的结论(POC 实况)

- **模型**:FLUX.2-dev = 32B rectified-flow transformer + **Mistral-Small-3.1-24B** 文本编码器(两个独立模型,共 ~90GB 下载)
- **硬件**:EC2 **g6e.4xlarge**(L40S **46GB** + 128GB RAM,us-west-2)。标称 48GB,PyTorch 进程实际可用仅 **~44.4GB**,对 FLUX.2 训练极度吃紧——显存管理是本项目最大工程难点
- **单 LoRA**:390MB rank-32 safetensors(触发词 `SLOTIP`),slots 美术风格迁移**成功**;1500 步训完,实测 **1000 步即收敛**,之后饱和
- **分层 LoRA**:同一批 18 张图,用两套 caption 策略训出解耦的 **Style LoRA**(`slotstyle`,详细 caption → 学画风)+ **Character LoRA**(`slotchar`,稀疏 caption → 学角色身份)。核心原理:*没写进 caption 的共同特征会被焊进 LoRA*。两层 rank 统一 32(便于后续加权组合/合并)
- **多层组合**:推理时 `set_adapters([style,char], weights)` 加权叠加,生成既保画风又保角色的新资产;`08_demo_matrix.py` 提供单层对照(base / style-only / char-only / combo)× 多主题 × 权重梯度的对比矩阵

---

## 架构

```
本地 ctl.py  ──SSM──►  EC2 g6e.4xlarge (us-west-2, 长驻实例 stop/start 复用)
                          │
                          ├── docker pull ◄── ECR (us-east-1, 镜像由 CodeBuild 构建)
                          ├── 模型缓存 ◄────── EBS /opt/flux-cache/hf (106GB, 持久, 免重下)
                          ├── 数据集 ◄──────── S3 (us-east-1)
                          ├── docker run (ai-toolkit, arch:flux2)
                          │     ├── transformer fp8 CPU 量化 → 上 GPU(patch1)
                          │     ├── Mistral CPU 量化 → 缓存 text embedding
                          │     ├── prepare 前卸载 Mistral 到 CPU(patch2,省 24GB)
                          │     ├── LoRA 训练(训练时 GPU 只剩 transformer ~36GB)
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
| LoRA rank | 32 | 分层各层统一 rank,便于加权组合/合并 |
| Steps | 1500(1000 即收敛) | Character 层 1200 |
| 分辨率 | 640 | 省显存;但注意分辨率对 prepare 阶段 OOM 无效(见下) |
| 量化 | fp8 + low_vram | transformer/Mistral 均 CPU 量化后上 GPU(patch1) |
| 文本编码 | cache + unload | 缓存 embedding 后 **prepare 前**卸载 Mistral(patch2,关键) |
| 优化器 / LR | adamw8bit / 1e-4 cosine | |

**显存管理(本项目核心难点)**:FLUX.2 训练在 `prepare_accelerator` 阶段峰值最高——transformer + Mistral 若同时在 GPU 达 44GB,超 46GB 天花板。两个 patch(见 `patch_flux2_te.py`)解决:①Mistral CPU 量化后再上 GPU;②prepare 前把 Mistral 卸载到 CPU(训练不需要它常驻,已缓存 embedding)。卸载后训练显存稳定在 ~36GB。docker 关键参数:`--shm-size=24g`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`、HF 缓存挂 EBS。

**分层训练**:`06_prepare_layers.py` 从同一批图生成 style/char 两套 caption;`ctl.py train --layer style|char` 分别训练;`08_demo_matrix.py` 做多层组合对比出图。

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
│   └── patch_flux2_te.py     两个 patch:①Mistral CPU 量化后上 GPU ②prepare 前卸载 TE(避 OOM)
└── scripts/
    ├── config.py             共享配置(从 .env 读)
    ├── ctl.py                ⭐ 生命周期 CLI(start/stop/status/train --layer/logs/run)
    ├── 00_cleanup_sagemaker.py
    ├── 01_setup_infra.py     S3 + ECR + IAM + CodeBuild
    ├── 02_trigger_build.py   触发 CodeBuild
    ├── 03_upload_dataset.py  上传单一数据集
    ├── 06_prepare_layers.py  ⭐ 分层:同批图生成 style/char 两套 caption
    ├── 07_compose_experiment.py  多层 LoRA 组合网格实验
    ├── 08_demo_matrix.py     ⭐ Demo 生成矩阵(单层对照 × 主题 × 权重梯度)
    ├── 04_submit_training.py 起新实例跑训练(早期方式)
    └── 05_monitor.py
docs/
├── architecture/dual-stack-plan.md          ⭐ 双栈端到端规划
└── superpowers/specs|plans/                  设计文档与实施计划
```

---

## 踩坑记录(完整版见 memory)

两条核心教训:

1. **用错架构入口白费 8 轮调试**。FLUX.2-dev 必须用 ai-toolkit 的专用 `Flux2Model`(`arch:flux2`),它用 `torch.device("meta")` + `load_state_dict(assign=True)` 正确加载;通用 `is_flux:true` 路径用 `from_pretrained()+.to()` 与 FLUX.2 新架构不兼容。
2. **OOM 差几百 MB 时先查"模型没及时卸载",而非盲目降 batch/分辨率**。看 OOM 报错的 `allocated`:若接近上限说明是权重占满型,激活类优化(分辨率/batch/grad-accum)无效——真正省显存靠卸载不需要的模型(如 prepare 阶段卸载文本编码器,一次省 24GB)。

| 问题 | 解法 |
|------|------|
| SageMaker g6e 配额=0 | 改用 EC2(G/VT 配额独立宽松) |
| us-east-1 全区 g6e 缺货 | 切 us-west-2;DryRun 只验权限不验容量 |
| FLUX.2-dev + Mistral 都是 gated | HF 网页各自申请 |
| **用错架构入口** | `arch: flux2`(非 `is_flux: true`)|
| optimum-quanto 0.2.4 + torch 2.6 fake-impl bug | 升 0.2.7,用 `--no-deps`(否则 torch 被拉到 2.12)|
| requirements 把 quanto 降回 0.2.4 | 先装 requirements 再 force-reinstall 0.2.7 |
| Mistral 加载 OOM(bf16 整个上 GPU)| patch1:先 CPU 量化(48→24GB)再上 GPU |
| **prepare 阶段 OOM(差 ~200MB,权重占满型)** | patch2:prepare 前把文本编码器卸载到 CPU(省 ~24GB)。**注意:降分辨率/降 grad-accum 对权重占满型 OOM 无效**,只能靠卸载模型 |
| DataLoader Bus error | `--shm-size=24g`(默认 64MB 共享内存不够)|
| 想上多卡解决单卡 OOM | 不可行:ai-toolkit FLUX.2 只支持数据并行(每卡一份完整模型),`split_model_over_gpus` 硬锁 FLUX.1 |
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
