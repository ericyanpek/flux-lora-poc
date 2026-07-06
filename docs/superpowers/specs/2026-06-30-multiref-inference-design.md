# FLUX.2 原生多参考图推理 —— 设计(第二条互补主线)

> **状态**:📋 设计 + 🟡 脚本骨架(`poc/scripts/inference/09_multiref_infer.py`),**未 GPU 验证**。
> **定位**:与已跑通的分层 LoRA(见 `docs/experiments/layered-lora-results.md`)**并列的第二条能力主线**,不是替代。
> **来源**:外部走查(2026-07-06 HANDOFF)T6;API 已按 diffusers `main` 文档核实(2026-07-06)。

---

## 1. 为什么要这条主线

分层 LoRA 已验证能做"风格迁移 + 角色泛化",但**角色一致性(同一角色跨场景/姿态保持身份)有两个短板**:
1. Character LoRA 与 Style LoRA 用同一批图训练,未能干净解耦(见实验报告第 3 节)。
2. 每换一个新角色都要重训一个 LoRA,不适合 Demo 现场"给一张定妆图立刻复用"。

FLUX.2 **底模自带**多参考图能力(官方模型卡:"No need for finetuning: character, object and style reference without additional training",最多 10 张参考图)。这条路**无需训练、可现场反复调**,正面解掉角色一致性问题——是对 LoRA 主线的互补,不是竞争。

---

## 2. API 事实(已按 diffusers main 文档核实,2026-07-06)

diffusers 的 FLUX.2 有 4 个 pipeline(`src/diffusers/pipelines/flux2/`):

| Pipeline | 文本编码器 | 参考图支持 | 说明 |
|----------|-----------|-----------|------|
| `Flux2Pipeline` | Mistral3 (dev) | `image=` 接受 `list[PIL.Image]` | dev 主 pipeline;`image` 文档描述偏 img2img "starting point",**多参考图语义需实测确认** |
| `Flux2KleinPipeline` | Qwen3 (klein) | `image=` | Klein 9B 变体 |
| `Flux2KleinKVPipeline` | Qwen3 (klein) | **`image=` 明确为 reference conditioning** | KV-cache 缓存参考图 token;**文档明确写"reference image conditioning",是多参考图的正主**;默认 4 步,快 |
| `Flux2KleinInpaintPipeline` | Qwen3 | — | inpaint |

**关键结论**:
- 参考图入参统一是 **`image=`**(单张 `PIL.Image` 或 `list[PIL.Image]`),不是 `reference_images=`。走查里的猜测方向对。
- **角色一致性的最佳 pipeline 是 `Flux2KleinKVPipeline`**(文档唯一明确标注"reference image conditioning + K/V cache"的),官方示例:
  ```py
  from diffusers import Flux2KleinKVPipeline
  from PIL import Image
  pipe = Flux2KleinKVPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-9b-kv", torch_dtype=torch.bfloat16)
  pipe.to("cuda")
  ref = Image.open("reference.png")
  img = pipe("A cat dressed like a wizard", image=ref, num_inference_steps=4).images[0]
  ```
- ⚠️ **dev 版(`Flux2Pipeline`)用 `image=[多张]` 做多参考图的确切行为,骨架里标 🟡,GPU 上跑通前不声称。** Klein-KV 是文档背书最强的路径,但需要另一套 Klein 权重(Qwen3 编码器,非 Mistral)。

---

## 3. 显存与机器

- dev `Flux2Pipeline` 全量 bf16 在 46GB 上此前实测 segfault/OOM(见 `poc/scripts/inference/` 失败尝试);要跑需 fp8/量化或更大卡。
- **Klein 9B 显存需求断崖下降**,46GB 甚至更小卡可跑,且 KV pipeline 默认 4 步、快——**Demo 现场交互的最优解可能是 Klein-KV**。
- 复用现有 ComfyUI 推理机(g6e.4xlarge)或新起 Klein 专用机。ComfyUI 侧也有对应的参考图节点(后续可对齐)。

---

## 4. 脚本骨架职责(`09_multiref_infer.py`)

1. 载入 1~N 张角色定妆参考图(本地或 S3)。
2. 用 `image=[refs]` + prompt 生成"同角色 × 新场景/新姿态"。
3. 固定 seed 输出一组对照:参考图 → 多个新场景。
4. 产物存 S3 `demo/multiref/`,供 README 引用。

**验收(达标才可标 ✅)**:一组"同一角色 × 多场景"样图,现场可改 prompt 重出且角色身份稳定。达标前 README 标 🟡。

---

## 5. 未决 / 待实测

- dev `Flux2Pipeline` `image=[多张]` 是否真做多参考融合(vs 仅取第一张当 img2img 起点)——**必须 GPU 实测**。
- Klein-KV 需要 `FLUX.2-klein-9b-kv` 权重(gated,Qwen3 编码器)——下载 + 显存待测。
- 与分层 LoRA 的组合(参考图定角色 + Style LoRA 定画风)是否可叠加——探索项。
