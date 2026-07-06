# FLUX.2 LoRA 双栈架构规划:训练栈 × 推理栈 × 协同

> 端到端规划。基于三路并行调研(训练栈/推理栈/协同,均带权威出处)+ 当前 POC 实况。
> 目标定位:**生产级在线 API + 双栈对等 + LoRA 产物管理分发为协同核心**。
> 日期:2026-06-28

---

## 〇、当前状态锚点(规划的起点)

已跑通(POC):
- 训练:ai-toolkit + FLUX.2-dev,EC2 g6e.4xlarge(L40S 46GB),CodeBuild→ECR 镜像,W&B 监控,SSM 存密钥,EBS 缓存 90GB 双模型
- 产物:390MB rank-32 safetensors LoRA(游戏美术风格),存 S3
- 已验证的关键约束:FLUX.2 = transformer(32B) + Mistral-Small-3.x-24B 编码器(版本号说明见 README 脚注),**L40S 46GB 显存对训练和推理都极度吃紧**

未做:推理服务、产物版本化、训练/推理协同、自动化。

---

## 一、最重要的一个架构判断(先看这个)

调研给出一个贯穿双栈的核心结论,直接决定硬件和架构:

> **把 Mistral-24B 文本编码器拆成独立的、可单独调度的单元** —— 这是同时解决"训练显存吃紧""推理显存吃紧""推理扩缩容"三个问题的关键动作。

- 训练侧:已经在用(`cache_text_embeddings` + 编码完卸载),这是 POC 跑通的关键。
- 推理侧:把 Mistral TE 做成独立常驻"编码服务"(或用 FLUX.2 的 remote text-encoder 模式),扩散 transformer 只管去噪。两者各自占显存、各自扩缩容。
- 出处:FLUX.2 官方 diffusers 文档对 <80GB 卡都要求 offload,40-48GB 卡建议 8-bit + TE 可卸载([flux2_dev_hf.md](https://github.com/black-forest-labs/flux2/blob/main/docs/flux2_dev_hf.md))。

**硬件结论**:
- **L40S 46GB 单卡**:只够 POC / 异步降级出图(fp8 + TE 卸载 + batch=1)。你 POC 推理 OOM 正是 VAE 解码 35-42GB + Mistral 编码叠加逼近 46GB([apatero FLUX.2 显存分解](https://apatero.com/blog/flux-2-memory-optimization-62gb-vram-spike-fix-guide-2025))。
- **生产在线 API 建议 g7e RTX PRO 6000 96GB**:让 TE + fp8 transformer + VAE 全常驻,腾 40GB+ 做 batch/compile/多 LoRA。Blackwell 缺货期用 L40S + 异步队列过渡。

---

## 二、训练栈规划

### 现在就做(低摩擦,全基于现有栈)

| 项 | 做法 | 出处 |
|---|---|---|
| **复现三件套** | 每个 W&B run 的 config 记 `config@gitsha + image@sha256 + dataset@sha256 + base_model@hfcommit` | [W&B Config](https://docs.wandb.ai/guides/track/config/) |
| **镜像锁定** | AWS DLC 作 base(已 pin CUDA/torch)+ uv 锁依赖 + 按 digest 引用(非 latest) | [AWS DLC](https://github.com/aws/deep-learning-containers) / [uv](https://docs.astral.sh/uv/concepts/projects/sync/) |
| **基模缓存** | 90GB 权重走 EBS 快照/S3 预拉 + 锁 HF commit(已部分做:EBS 缓存) | — |
| **实例生命周期** | checkpoint→S3 + autostop/Spot,评估 SkyPilot 薄层(起→跑→续→关+spot 容错一条命令) | [SkyPilot](https://docs.skypilot.co/en/latest/examples/managed-jobs.html) |
| **数据/产物版本化** | W&B Artifacts(S3 reference + alias + lineage),零额外基础设施 | [W&B Artifacts](https://docs.wandb.ai/guides/artifacts/) |
| **超参扫描** | W&B Sweeps(grid/random/bayes) | [W&B Sweeps](https://docs.wandb.ai/guides/sweeps/) |
| **单任务编排** | Step Functions 状态机(起实例→训→登记→关) | [SFN×SageMaker](https://docs.aws.amazon.com/step-functions/latest/dg/sample-train-model.html) |

### 规模化后再做
- 编排升级 Metaflow(Python-first,迁移成本低)或 Flyte(K8s,强类型)
- Spot 训练用 `price-capacity-optimized` + 多 AZ,省 70-90%([EC2 Spot 最佳实践](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-best-practices.html))
- 数据涨到数千张再上 lakeFS;托管化用 SageMaker Managed Spot Training

---

## 三、推理栈规划

### 框架选型(关键:vLLM 不适用扩散模型)

| 阶段 | 方案 | 出处 |
|---|---|---|
| **MVP** | FastAPI + diffusers,封装 SageMaker Async,异步队列出图 | [Ray Serve SD](https://docs.ray.io/en/latest/serve/tutorials/stable-diffusion.html) |
| **规模化** | Ray Serve(原生 replica/autoscaler/`@serve.batch`)或 Triton(TensorRT 极致延迟 + 显式 load/unload) | [Triton 配置](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html) |

### 多 LoRA 动态热加载(diffusers 原生,无需重启)

- `load_lora_weights(path, adapter_name=...)` 加载 → `set_adapters(...)` 切换/混合
- **hotswap**:`load_lora_weights(..., hotswap=True)` + `enable_lora_hotswap(target_rank=32)`(compile 后不重编译),FLUX LoRA 在 H100 上 7.89s→3.55s([HF lora-fast](https://huggingface.co/blog/lora-fast))
- **两个硬限制**:① hotswap 不支持 text-encoder LoRA;② 后续 LoRA 的 target 层须是首个的子集,先加载 target 最多的。当前游戏美术 LoRA 训练含 CLIP/TE 侧(`strength_clip>0`),生产化前需确认改为 transformer-only,否则 hotswap 路径不成立(待验证项)
- 出处:[diffusers PEFT 推理](https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference)

### 显存与冷启动
- fp8 量化(甜点位,FLUX 整体 ~18-20GB,~95% 质量)
- Mistral TE 编码完卸载 / remote TE(本地只剩 ~18GB)
- 冷启动:safetensors 流式 + CUDA graph 缓存 + 镜像瘦身 + NVMe + 启动预热 dummy 推理(热镜像 25-85s vs 冷镜像 6-10 分钟,[Spheron 冷启动](https://www.spheron.network/blog/keda-knative-gpu-autoscaling-kubernetes-llm-cold-start/))

### 批处理与扩缩容
- 扩散用 **dynamic batching**(Triton `max_batch_size>1` 自动凑批),**不是** continuous batching(那是 LLM 的)
- 扩缩容:**HPA 不行**(GPU 满时 CPU 仅 5-8%);用 KEDA(队列深度)/ Knative(scale-to-0)
- **AWS MVP 首选 SageMaker Async Inference**:队列 + scale-to-0 + S3 + SNS,生图天然异步,几乎零运维([SageMaker Async](https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference.html))
- 规模化:EKS + Karpenter(供 g6e/g7e)+ KEDA,生产 endpoint minScale≥1 避免冷启动

---

## 四、协同:LoRA 产物管理与分发(你最看重的核心)

> **实现状态(2026-07-06)**:本节是**生产化目标设计,尚未实现**。当前训练↔推理的实际协调契约是
> `07_deploy_comfyui.py` 的 **latest-SUCCESS-by-S3-LastModified** 扫描(过滤 `status.txt==SUCCESS`,
> 按 S3 LastModified 取最新,缺失则硬报错)——一个刻意精简的 POC 契约,尚无 W&B Artifacts 注册、
> manifest JSON、SSM 指针或 hotswap。下文的三件套/灰度/回滚是演进方向,不代表已落地。

### 架构决策:不引入重型 Registry

> **W&B Artifacts(注册+血缘+`production` alias 信号)+ S3(产物真相源)+ EventBridge/Lambda(评测分发)+ S3 manifest + SSM 指针(推理契约)+ diffusers hotswap(热加载)**

全部复用现有 AWS+W&B 栈,不引入 MLflow/SageMaker Registry(后者只在已用 SageMaker Pipelines 时才划算)。

### 通信契约三件套(训练栈↔推理栈的接口)

1. **Manifest(S3 JSON,真相源)**— 推理服务"有哪些 LoRA 可用"全靠它,含 adapter_name/slot/trigger_word/s3_uri/sha256/wandb_version/rollout(灰度)/default_scale
2. **指针(SSM Parameter Store)**— `/<env>/inference/lora-manifest-version = N`,推理服务只感知这个数字变化
3. **信号(事件)**— 主路径:W&B Automation(加 `production` alias → webhook);兜底:EventBridge(S3 Object Created)

### 端到端生命周期

```
训练(EC2 g6e + ai-toolkit) → adapter.safetensors + metadata.json
   │ s3 cp (+sha256)
   ▼
S3 ──(Object Created)──► EventBridge ──► Lambda「注册器」
   │  W&B log_artifact(ref S3) + use_artifact(dataset 血缘), alias="candidate"
   ▼
评测门禁(Step Functions / CodeBuild GPU): golden prompts + 固定 seed
   │  CLIP score(trigger契合) | 角色自相似 | vs上版回归 | 黑图检测
   │  通过?
   ├─是→ W&B alias="production" ──(Automation webhook)──► Lambda「分发器」
   │         更新 S3 manifest(version++) + SSM 指针 + rollout=canary
   │         ▼
   │     推理服务轮询 SSM → 拉 manifest → 校验 sha256 → load_lora_weights(hotswap=True)
   │         canary 10%→v7 / 90%→v6,监控失败率/延迟/CLIP
   │         回滚 = alias 指回 + manifest++ + SSM bump(秒级 set_adapters 切回)
   └─否→ alias="rejected" + Slack 告警
```

出处:[W&B Automations](https://docs.wandb.ai/guides/core/automations/) / [S3→EventBridge](https://docs.aws.amazon.com/AmazonS3/latest/userguide/EventBridge.html) / [diffusers 评测](https://huggingface.co/docs/diffusers/main/en/conceptual/evaluation) / [SageMaker 部署护栏](https://docs.aws.amazon.com/sagemaker/latest/dg/deployment-guardrails-blue-green.html)

### LoRA 元数据(钉死在 W&B artifact + S3 metadata.json,两份一致)
关键字段:`base_model@revision`(头号坑:FLUX.2 升版让旧 LoRA 失配)、dataset W&B artifact 引用、训练超参、wandb_run、eval 指标、sha256。

---

## 五、双栈目标架构总图

```
═══════════════════ 训练栈 ═══════════════════      ═══════════════════ 推理栈 ═══════════════════
控制面: Step Functions / W&B Sweeps                  入口: API GW/ALB → SQS/SageMaker Async (KEDA 监控队列)
   │ {config@sha, image@sha, data@sha, base@commit}     │ 返回 request_id (异步)
   ▼                                                    ▼
GPU(EC2 g6e Spot, SkyPilot 管生命周期)              ┌─ Text-Encoder Service (Mistral-24B fp8, 独立常驻/扩缩) ─┐
  容器(ECR@digest): DLC base + uv.lock + ai-toolkit    │              │ embed                                 │
  基模(90GB)← EBS快照/S3, TE编码缓存, ckpt→S3          │              ▼                                       │
   │ metrics              │ 产物                        │  Diffusion Service (FLUX.2 fp8 + VAE, torch.compile)  │
   ▼                      ▼                             │   LoRA Manager: enable_lora_hotswap(rank32)           │
W&B Runs ◀──lineage── W&B Artifact(LoRA, alias)        │   set_adapters/hotswap, LRU 缓存                       │
                          │ + S3 权威副本               │              ▲ 按需拉取(390MB)                        │
                          │                             └──────────────┼────────────────────────────────────────┘
                          ▼                                            │
        ┌──────── 协同层(复用 AWS+W&B)────────┐                       │
        │ S3 manifest(真相源) + SSM 指针(信号)  │◀──────────────────────┘ 推理轮询 SSM→拉 manifest→hotswap
        │ EventBridge/W&B webhook → Lambda 注册/评测/分发              │
        │ 评测门禁(CLIP/角色一致/回归) → alias=production → 灰度/回滚    │
        └──────────────────────────────────────┘
节点供给: Karpenter → g7e 96GB(生产)/ g6e 46GB(训练+批处理降级)
```

---

## 六、落地路线图(分阶段,避免一次到位)

### 阶段 1:训练栈生产化(1-2 周,纯现有栈)
1. 复现三件套写进 W&B config
2. uv 锁依赖 + DLC base + digest 引用
3. checkpoint→S3 + autostop(消除手工关机烧钱)
4. LoRA/数据用 W&B Artifacts 版本化 + lineage
- **产出**:可复现、可追溯的训练,产物自动版本化

### 阶段 2:协同链路(1-2 周,这是你最看重的)
1. metadata.json 规范(含 base_model@revision、sha256)
2. S3 ObjectCreated → Lambda 注册器(写 W&B Artifact + candidate alias)
3. 评测门禁 MVP:CLIP score + 黑图检测(CodeBuild GPU 容器)
4. manifest(S3)+ SSM 指针契约
- **产出**:训练产出 LoRA 自动注册、评测、生成可用 manifest

### 阶段 3:推理栈 MVP(2-3 周)
1. FastAPI + diffusers,fp8 + TE 卸载,L40S 单卡先跑异步
2. SageMaker Async Inference(队列 + scale-to-0)
3. 推理服务轮询 SSM → 拉 manifest → load_lora_weights 热加载
4. **同时申请 g7e 配额/监控容量**,Blackwell 到货切生产硬件
- **产出**:端到端可用的异步生图 API,多 LoRA 切换

### 阶段 4:生产强化(规模化时)
- g7e 96GB + TE 独立服务 + dynamic batching
- hotswap + torch.compile 零重编译切换
- EKS + Karpenter + KEDA 自动扩缩容
- 灰度(canary)+ baking period + CloudWatch 自动回滚
- 编排升级 Metaflow/Flyte

---

## 七、关键风险与已知坑

| 风险 | 影响 | 缓解 |
|---|---|---|
| **g7e Blackwell 缺货** | 生产硬件拿不到 | L40S + 异步队列过渡,持续监控容量;TE 拆独立服务降低单卡压力 |
| **base_model 升版** | 旧 LoRA 失配 | metadata 锁 `base_model@hfcommit`,推理校验 |
| **hotswap 不支持 TE LoRA** | 若 LoRA 含 TE 部分无法热切 | 训练时只训 transformer 的 LoRA(当前就是) |
| **L40S 推理 OOM** | 在线服务不稳 | fp8 + TE 卸载 + batch=1 + 异步;根治靠 g7e |
| **ai-toolkit 依赖脆弱** | 镜像重建漂移 | uv.lock + 锁 ai-toolkit git commit + digest 引用 |

---

## 全部来源
训练栈:[W&B Config](https://docs.wandb.ai/guides/track/config/)/[Sweeps](https://docs.wandb.ai/guides/sweeps/)/[Artifacts](https://docs.wandb.ai/guides/artifacts/) · [PyTorch 复现](https://docs.pytorch.org/docs/stable/notes/randomness.html) · [DVC](https://doc.dvc.org/use-cases/versioning-data-and-models) · [lakeFS](https://lakefs.io/blog/data-versioning/) · [Metaflow](https://docs.metaflow.org/introduction/why-metaflow) · [Flyte](https://flyte.org/blog/why-flyte) · [SFN×SageMaker](https://docs.aws.amazon.com/step-functions/latest/dg/sample-train-model.html) · [EC2 Spot](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-best-practices.html) · [SageMaker Managed Spot](https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html) · [SkyPilot](https://docs.skypilot.co/en/latest/examples/managed-jobs.html) · [AWS DLC](https://github.com/aws/deep-learning-containers) · [uv](https://docs.astral.sh/uv/concepts/projects/sync/)

推理栈:[FLUX.2 diffusers 文档](https://github.com/black-forest-labs/flux2/blob/main/docs/flux2_dev_hf.md) · [FLUX.2 模型卡](https://huggingface.co/black-forest-labs/FLUX.2-dev) · [FLUX.2 硬件要求](https://deepwiki.com/black-forest-labs/flux2/2.3-hardware-requirements) · [apatero 显存优化](https://apatero.com/blog/flux-2-memory-optimization-62gb-vram-spike-fix-guide-2025) · [diffusers hotswap](https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference) · [HF lora-fast](https://huggingface.co/blog/lora-fast) · [Ray Serve SD](https://docs.ray.io/en/latest/serve/tutorials/stable-diffusion.html) · [Triton](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html) · [SageMaker Async](https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference.html) · [Spheron 冷启动](https://www.spheron.network/blog/keda-knative-gpu-autoscaling-kubernetes-llm-cold-start/)

协同:[W&B Automations](https://docs.wandb.ai/guides/core/automations/) · [MLflow Registry](https://mlflow.org/docs/latest/ml/model-registry/) · [SageMaker Registry](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html) · [SageMaker EventBridge](https://docs.aws.amazon.com/sagemaker/latest/dg/automating-sagemaker-with-eventbridge.html) · [S3 EventBridge](https://docs.aws.amazon.com/AmazonS3/latest/userguide/EventBridge.html) · [SageMaker 部署护栏](https://docs.aws.amazon.com/sagemaker/latest/dg/deployment-guardrails-blue-green.html) · [diffusers 评测](https://huggingface.co/docs/diffusers/main/en/conceptual/evaluation) · [LoRAX](https://github.com/predibase/lorax) · [vLLM LoRA](https://docs.vllm.ai/en/latest/features/lora.html)
