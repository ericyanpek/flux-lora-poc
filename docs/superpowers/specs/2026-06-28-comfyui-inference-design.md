# ComfyUI 推理服务部署 — 设计文档

**日期**: 2026-06-28
**状态**: 已确认,待实现
**目标**: 在一台独立于训练机的 g6e EC2 上部署 ComfyUI,用于快速验证 FLUX.2-dev + slot-IP LoRA 的出图效果(海盗等任意主题),交互式反复调 prompt/steps。

## 背景与动机

- POC 训练已跑通,产出 LoRA `slotip-final/flux-lora-poc.safetensors`(1500 步,FLUX.2-dev,`arch:flux2`,触发词 `SLOTIP`)。
- 之前用 ai-toolkit 训练入口跑一次性推理,每次都重新加载+运行时量化(transformer 量化 ~9.5 分钟 + Mistral 24B CPU 量化数分钟),单次出图前置开销 ~13 分钟,无法快速试 prompt。
- 用户决定:**不再与训练机共用推理**,要一个独立、快速、可反复试的推理服务。
- 评估过 SageMaker Endpoint,但其健康检查超时 + 64GB 模型打包 + 自定义 inference.py + flux2 LoRA 格式等多重坑,使其"首次上线"反而最慢。**ComfyUI 是最快的独立推理验证路径**(官方原生支持 FLUX.2,用预量化 fp8 文件,加载一次常驻,出图秒级)。

## 关键事实(已查证)

- ComfyUI 官方支持 FLUX.2(README 列出,有官方 examples 页 `comfyanonymous.github.io/ComfyUI_examples/flux2/`)。
- ComfyUI 需要**它专用的预量化 fp8 文件**,与训练机缓存的 HF bf16 diffusers 格式**不通用**:
  - 扩散主干: `flux2_dev_fp8mixed.safetensors` → `models/diffusion_models/`
  - 文本编码器: `mistral_3_small_flux2_fp8.safetensors` → `models/text_encoders/`
  - VAE: `flux2-vae.safetensors` → `models/vae/`
- **ComfyUI 直接加载 fp8,不在运行时量化** → 避开了那 ~13 分钟量化瓶颈。
- 训练机的模型缓存(106GB)位于 `/opt/dlami/nvme`,是**实例临时盘(ephemeral NVMe/LVM),实例一停即丢失**,不可作为持久来源。
- 用户的 LoRA 是**原生 `diffusion_model.*` 格式**(非 FLUX.1 diffusers 的 `transformer.*` 格式),320 个 key,`double_blocks.* / single_blocks.*` 结构。

## 已确认的设计决策

| 决策点 | 选择 |
|---|---|
| 底模准备 | fp8 文件从 HF 下一次 + 缓存到 S3,之后从 S3 拉(同区快、稳、免 HF gated/限速) |
| Web 访问 | SSM 端口转发(复用现有 egress-only 无入站安全组,不暴露公网) |
| 机型 | g6e.4xlarge(L40S 48GB,16 vCPU/128GB RAM) |
| 运行方式 | 裸机装 ComfyUI(git clone + pip),非容器(避开训练镜像 ai-toolkit 依赖冲突 + flux2 加载坑) |
| LoRA 指向 | slotip-final 最终版(1500 步)单个 LoRA |
| LoRA 兼容性 | 先直接放原文件试加载;ComfyUI 通常能识别 `diffusion_model.*` 原生格式;万一不行再写 key 转换脚本兜底 |

## 架构

```
本地 (Mac)                              AWS (us-west-2)
07_deploy_comfyui.py
  ├── 启动独立 g6e.4xlarge ──────→ 全新 EC2(与训练机 i-00a60dc65d57b9bae 完全隔离)
  │                                    │
  │                                    ├── UserData 自动执行:
  │                                    │   ① 从 S3 拉 fp8 底模 + LoRA → ComfyUI 目录结构
  │                                    │      (首次:从 HF 下 fp8 → sync 到 S3)
  │                                    │   ② git clone ComfyUI + pip 装依赖
  │                                    │   ③ 启动 ComfyUI 监听 127.0.0.1:8188
  │                                    │
  └── SSM 端口转发 ───────────────→ 远端 8188 映射到本地 localhost:8188

浏览器 http://localhost:8188 → 调 prompt/steps/LoRA,秒级出图
```

**与训练机零耦合**:全新实例、独立生命周期,训练机可随时停。

## 组件

### 一次性准备:fp8 底模存 S3

S3 布局(新前缀 `comfyui-models/`,对应 ComfyUI 目录结构):

| 文件 | S3 去向 | ComfyUI 本地目录 |
|---|---|---|
| `flux2_dev_fp8mixed.safetensors` | `s3://flux-poc-984072314535-us-east-1/comfyui-models/diffusion_models/` | `models/diffusion_models/` |
| `mistral_3_small_flux2_fp8.safetensors` | `.../comfyui-models/text_encoders/` | `models/text_encoders/` |
| `flux2-vae.safetensors` | `.../comfyui-models/vae/` | `models/vae/` |
| `flux-lora-poc.safetensors`(已在 S3 outputs/slotip-final/) | 拷到 `.../comfyui-models/loras/` | `models/loras/` |

机制:首次起推理机时,UserData 检测 S3 是否已有 fp8 文件;无则从 HF 下载并 `aws s3 sync` 固化到 S3;有则直接从 S3 拉。之后每次起机都走 S3。

### 新增文件(2 个,对标现有流水线)

| 文件 | 职责 |
|---|---|
| `poc/scripts/07_deploy_comfyui.py` | 本地编排:起 g6e.4xlarge、注入 UserData、打印 SSM 端口转发命令。结构对标 `04_submit_training.py`(复用 SG / subnet / 容量回退 / IAM instance profile)。**不自动关机**(推理服务按需常开) |
| `poc/scripts/comfyui_userdata.sh`(或在 07 内内联生成) | 实例内引导:拉模型 → 装 ComfyUI → 启动服务,日志写 `/var/log/comfyui-setup.log` |

复用现有资产:无入站安全组 `flux-poc-training-sg`、4-AZ 容量回退、`flux-poc-ec2-role`(已有 S3 读权限,需确认含 comfyui-models 前缀写权限用于首次固化)。

### 访问与使用

- 脚本结尾打印 SSM 端口转发命令:
  `aws ssm start-session --target <iid> --document-name AWS-StartPortForwardingSession --parameters '{"portNumber":["8188"],"localPortNumber":["8188"]}'`
- 浏览器 `http://localhost:8188`,加载 FLUX.2 工作流,选 LoRA `flux-lora-poc`,触发词 `SLOTIP`,反复试 prompt,秒级出图。

## 错误处理

- **容量回退**:沿用 04 的多 AZ 重试逻辑(InsufficientInstanceCapacity 时换 AZ)。
- **IAM profile 传播延迟**:沿用 04 的重试。
- **首次下载 fp8 失败 / HF gated**:UserData 需要 HF_TOKEN(从 SSM `/flux-poc/hf-token` 取,复用现有机制);下载失败则日志报错,实例保留供调试。
- **LoRA 加载失败(key 格式不匹配)**:先验证直接加载;失败则编写 `convert_lora_keys.py` 把 `diffusion_model.*` 转成 ComfyUI 期望格式,作为兜底(列入计划但非首选路径)。
- **ComfyUI 启动失败**:日志 `/var/log/comfyui-setup.log`,实例保留(不自动关机),可 SSM 进入排查。

## 成本与生命周期

- 推理机是**常驻服务**,不像训练机跑完自停 → **用户需手动管理生命周期**,用完手动停。
- 脚本打印停止/终止提示;**不内置自动关机**。
- 落地前置:训练机 `i-00a60dc65d57b9bae` 已跑 16h+,建议先停止止血(独立动作,不阻塞本方案)。

## 验证标准(完成定义)

1. 推理机成功启动,ComfyUI 在 8188 监听。
2. SSM 端口转发后,本地浏览器能打开 ComfyUI。
3. FLUX.2 fp8 底模 + slotip LoRA 成功加载(无 key 错误)。
4. 用海盗主题 prompt(含 `SLOTIP` 触发词)出图成功,单张耗时秒级~几十秒(底模常驻后)。
5. fp8 文件已固化到 S3,二次起机直接从 S3 拉,无需重下 HF。

## 范围外(YAGNI)

- 不做 HTTP API 封装(那是后续 SageMaker Async Endpoint 生产方案的事)。
- 不做模型版本协同/manifest(已单独讨论,另作专题)。
- 不做多 checkpoint 切换(本次只验证 1500 步最终版)。
- 不做自动扩缩/缩容到 0。
