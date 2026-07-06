# FLUX.2 原生多参考图推理 —— 设计(第二条互补主线)

> **状态**:🟡 **部分验证**(2026-07-06 GPU 实跑)——身份保持 ✅,场景跟随 ⚠️。尚未达标,不标 ✅。
> **定位**:与已跑通的分层 LoRA(见 `docs/experiments/layered-lora-results.md`)**并列的第二条能力主线**,不是替代。
> **选型**:ComfyUI + FLUX.2-dev(与现有推理栈一致,`ReferenceLatent` 节点已实跑生效);diffusers Klein-KV 降为备选。

---

## 0. 验证结果(2026-07-06 GPU 实跑)

在推理机上以美人鱼参考图 + 多个新场景 prompt 实跑(脚本 `09_multiref_infer.py`,产物 S3 `demo/multiref2/`):

- ✅ **角色身份保持极好**:参考图的脸型、发色、珍珠冠、青绿鱼尾、金三叉戟在多张输出中高度一致。`ReferenceLatent` 注入生效,身份锚定强。
- ⚠️ **场景不跟随 prompt**:要求"沉船 / 夕阳水面 / 珊瑚花园"等新场景,输出仍近似复刻参考图构图,prompt 场景描述被压制。

**原因分析(重要,避免误判)**:
- **不是"缺场景 LoRA/模型不认识场景"**。底模 FLUX.2-dev 原生认识沙滩、沉船、夕阳等通用概念——证据:base 配置(无任何 LoRA)纯 prompt 出图时,场景词全部生效(见 `layered-lora-results.md` 的 base 列,海盗有船甲板、龙有洞穴)。
- **真正原因是参考图注入权重过高**,`ReferenceLatent` 当前接法下参考图主导构图,盖过文字 prompt。即"参考图音量 100、文字音量 20",退化为近似 img2img 复制。这是身份保持 vs 场景自由度的平衡问题,非能力缺失。

**下一步(调优方向,非训练)**:
1. 配合 guidance / denoise 强度调节,降低参考图对构图的锁定。
2. 改用 FluxKontext 部分注入(只借角色特征,不锁构图)。
3. 参考图裁剪到仅角色主体 / 降分辨率再注入,减少背景干扰。

---

## 1. 为什么要这条主线

分层 LoRA 已验证能做"风格迁移 + 角色泛化",但**角色一致性(同一角色跨场景/姿态保持身份)有两个短板**:
1. Character LoRA 与 Style LoRA 用同一批图训练,未能干净解耦(见实验报告第 3 节)。
2. 每换一个新角色都要重训一个 LoRA,不适合 Demo 现场"给一张定妆图立刻复用"。

FLUX.2 **底模自带**多参考图能力(官方模型卡:"No need for finetuning: character, object and style reference without additional training",最多 10 张参考图)。这条路**无需训练、可现场反复调**,正面解掉角色一致性问题——是对 LoRA 主线的互补,不是竞争。

---

## 2. 选型:ComfyUI + FLUX.2-dev(与现有推理栈一致)

**首选方案 = ComfyUI + FLUX.2-dev,复用现有推理机,不引入新框架/新模型。**

依据:
- FLUX.2-**dev** 底模**原生支持**多参考图,是 BFL 官方一级特性。模型卡原文:"No need for finetuning: character, object and style reference without additional training in one model",并列出 "multi-reference editing"。**不是"只有 Klein 能做参考图"**。
- 本项目已确立的核心结论:diffusers / ai-toolkit 在 46GB 上做 FLUX.2 推理会 segfault/OOM,**ComfyUI(fp8 底模)才是可靠路径**。多参考图应沿用同一栈。
- 已在推理机(g6e.4xlarge)的 ComfyUI 上核实存在 **`ReferenceLatent`** 节点(FLUX.2 / Kontext 系列做参考图 conditioning 的标准节点),配 `EmptyFlux2LatentImage` / `Flux2Scheduler` 等。**无需 diffusers、无需额外权重**——底模、VAE、TE 都已加载。
- 额外收益:参考图定角色 + 现有 Style LoRA 定画风,可在同一 workflow 里叠加(探索项)。

**workflow 结构(ComfyUI)**:
```
LoadImage(参考图) → (可选 FluxKontextImageScale) → VAEEncode → ReferenceLatent ┐
CLIPTextEncode(prompt) ─────────────────────────────────────────────────────┼→ (conditioning)
UNETLoader(flux2 fp8) + CLIPLoader(flux2) + VAELoader ────────────────────────┘
        → KSampler → VAEDecode → SaveImage
```
`ReferenceLatent` 把参考图的 latent 注入 conditioning;多张参考图可串接多个 `ReferenceLatent`(对应官方"多参考")。

### 备选:diffusers Klein-KV(仅当需要极速交互)

`Flux2KleinKVPipeline`(diffusers)是文档明确标注 "reference image conditioning + K/V cache" 的 pipeline,默认 4 步、显存低,**适合现场高频交互**。但它需要另一套 gated 权重 `FLUX.2-klein-9b-kv`(Qwen3 编码器,非 Mistral),且回到了 diffusers 栈。**仅在"4 步极速交互"成为硬需求时才考虑**,不作为首选。参考图入参为 `image=`(单张 PIL 或 list)。

---

## 3. 显存与机器

- 复用现有 ComfyUI 推理机(g6e.4xlarge),fp8 底模已加载,`--lowvram` 分时 offload,与文生图路径同栈同显存表现(GPU 峰值 <42GB)。
- 不新增实例、不下载新权重——这是选 ComfyUI+dev 相对 Klein-KV 的最大成本优势。

---

## 4. 脚本职责(`09_multiref_infer.py`)

沿用 `comfy_gen.py` 的 ComfyUI API 驱动方式(POST `/prompt` + 轮询 `/history`):
1. 载入 1~N 张角色定妆参考图。
2. 构造含 `ReferenceLatent` 的 FLUX.2 workflow,+ prompt 生成"同角色 × 新场景/新姿态"。
3. 固定 seed 输出对照:参考图 → 多个新场景。
4. 产物存 S3 `demo/multiref/`,供 README 引用。

**验收(达标才可标 ✅)**:一组"同一角色 × 多场景"样图,角色身份跨场景保持。达标前 README 标 🟡。

---

## 5. 未决 / 待实测

- `ReferenceLatent` 对 FLUX.2-dev 的确切接线(单张 vs 多张串接)与身份保持强度——**GPU 实测**。
- 参考图 + Style LoRA 叠加(参考图定角色 + LoRA 定画风)——探索项。
- 若需 4 步极速交互再评估 Klein-KV 备选路径。
