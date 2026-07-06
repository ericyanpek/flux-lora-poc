# 客户 Demo 目标 — 差距分析与实施路线

> ⚠️ **状态更新(2026-07-06)**:本文档为 **2026-06-29 的早期差距分析**,部分标为"要做/未实现"的能力现已完成并有产物。以最新为准:
> - 分层 LoRA(Style + Character)训练 **✅ 已完成** → 见 [实验报告](../experiments/layered-lora-results.md)
> - 多层组合 **✅ 已完成**(ComfyUI `LoraLoader` 加权叠加)→ 见实验报告对照矩阵
> - 独立推理服务 **✅ 已跑通**(ComfyUI 独立机)→ 见 `poc/scripts/07_deploy_comfyui.py`
> - 原生多参考图 **🟡 部分验证**(身份保持 ✅ / 场景跟随待调优)
> 下文的能力现状表(第 45 行起)保留作历史记录,请对照上述更新阅读。

> 目标:为客户构建**企业级游戏美术 AI 生产平台 Demo**(不只是训一个模型)。FLUX.2 + 多层 LoRA(Style/UI/Character),保持美术风格/UI/Icon/IP 角色一致性,快速生成新资产。
> 本文档 = 对照三大 Demo 目标的"已有 / 要做 / 风险",带技术验证依据。日期 2026-06-29。

---

## 〇、一个改变方案的关键发现(先看这个)

调研验证了一条**直接影响架构**的事实:

> **FLUX.2 原生支持最多 10 张参考图(multi-reference),是官方一级特性,专为"无需训练的角色/IP/风格一致性"设计。** 模型卡原文:"No need for finetuning: character, object and style reference without additional training in one model"([FLUX.2-dev 模型卡](https://huggingface.co/black-forest-labs/FLUX.2-dev)、[BFL 博客](https://bfl.ai/blog/flux-2))。

**对方案的影响**:
- 客户原计划用"Character LoRA"保角色一致性。但 FLUX.2 的**原生多参考图**才是最成熟路径,应作为角色一致性的**主力**,Character LoRA 退为补充。
- **IP-Adapter 不要纳入 FLUX.2 方案** —— FLUX.2 上基本无可用 IP-Adapter(FLUX.1 的不兼容新架构),原生多参考已覆盖该场景。客户文档里"预留 IP-Adapter"应改为"用 FLUX.2 原生多参考替代 IP-Adapter"。
- **多层 LoRA 叠加是唯一不成熟环节**(见风险章节),需要实验验证,不能假设直接可用。

调整后的角色一致性技术分工:

| 需求 | 推荐主力 | 备选 |
|------|----------|------|
| 整体美术风格 | **Style LoRA**(烧进权重,全局基调) | — |
| UI 布局/Icon 规范 | **UI LoRA** + 结构化 caption | 参考图 |
| **IP 角色一致性** | **FLUX.2 原生多参考图**(给定妆图) | Character LoRA 补充 |

---

## 一、对照三大 Demo 目标的差距

### 目标 1:FLUX.2 vs FLUX.1 能力对比(Prompt Follow / 角色一致性 / 多元素 / 稳定性)

| 项 | 现状 | 差距 |
|----|------|------|
| FLUX.2 训练跑通 | ✅ 已有(SLOTIP LoRA) | — |
| FLUX.1 对比基线 | ❌ 无 | 客户说可忽略对比 FLUX.1 本身,但**展示 FLUX.2 的提升点**仍需准备对照样图 |
| 对比维度样图 | ❌ 无 | 需准备:同 prompt 下 Prompt Follow、多元素生成、角色一致性(用多参考图)的展示图 |

**要做**:准备一组"能力展示"样图(不必真跑 FLUX.1),用 FLUX.2 展示:复杂 prompt 跟随、一次生成多个游戏美术元素、用多参考图保持角色。可结合客户已有的 FLUX.1 旧案例作演进叙事。

### 目标 2:训练流程展示(资产 → 清洗/标注 → LoRA FT → 评估 → Registry)

| 环节 | 现状 | 差距 |
|------|------|------|
| 游戏美术资产数据 | 🟡 仅 18 张游戏美术图 | 需扩充,且按 Style/UI/Character 分类组织 |
| **数据清洗/自动标注** | ❌ 手工 caption | ai-toolkit 内置 **Qwen3-VL captioner**(角色/背景)+ **Ideogram4 结构化标注**(UI/icon,带 bbox)——[出处](https://deepwiki.com/ostris/ai-toolkit/20.2-captioner-backends)。需接入自动标注流程 |
| LoRA Fine-tuning | ✅ 跑通(单 LoRA) | 需扩展到**分层训练**(Style/UI/Character 各一个数据集+trigger word) |
| **模型评估** | ❌ 无 | 需评测门禁:CLIP score(trigger 契合)+ 角色一致性 + 黑图检测([diffusers 评测](https://huggingface.co/docs/diffusers/main/en/conceptual/evaluation)) |
| **Model Registry** | 🟡 已设计未实现 | W&B Artifacts + S3 manifest(双栈规划已设计) |

**要做**:数据分层组织 + 自动标注接入 + 分层 LoRA 训练 + 评测门禁 + Registry 落地。

### 目标 3:推理流程展示(Prompt + 多层 LoRA 组合 + 多参考图扩展)

| 环节 | 现状 | 差距 |
|------|------|------|
| 单 LoRA 推理 | 🟡 训练入口可出图(慢,~13 分钟量化) | 需独立推理服务 |
| **独立推理服务** | 🟡 ComfyUI 方案已设计未实现 | ComfyUI 原生支持 FLUX.2 + 多 LoRA + 多参考图,出图秒级 |
| **多层 LoRA 组合** | ❌ 无 | diffusers `set_adapters([style,ui,char], weights)` 或 ComfyUI 多 LoRA 节点。**需实验验证叠加效果**(风险) |
| **多参考图(角色一致性)** | ❌ 无 | FLUX.2 原生,`Flux2Pipeline(image=[refs])` 最多 10 张 |
| IP-Adapter | — | **不做**,原生多参考替代 |

**要做**:独立推理服务(ComfyUI 优先)+ 多层 LoRA 组合 + 多参考图能力。

---

## 二、需要新建的能力(按 Demo 价值排序)

### A. 分层 LoRA 训练(Style / UI / Character)— Demo 核心
当前只有一个混合 LoRA。需要:
1. **数据分层**:把游戏资产按 Style(整体画风)/ UI(布局元素)/ Character(IP 角色)分三组数据集,各自 trigger word(如 `slotstyle` / `slotui` / `charXXX`)
2. **三次独立训练**:复用现有 `ctl.py train` 链路,每组数据训一个 rank-32 LoRA(**rank 统一**,为后续可能的 TIES 合并留路)
3. **自动标注**:接入 ai-toolkit 内置 Qwen3-VL(角色/背景)+ Ideogram4 结构化(UI/icon)

### B. 多层 LoRA 组合推理 — Demo 核心(⚠️ 最高风险)
- diffusers:`set_adapters(["style","ui","char"], adapter_weights=[w1,w2,w3])`
- **已知问题**:多 LoRA 互相渗透/风格冲突,style LoRA 会污染 character 的非目标区域([diffusers 文档](https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference))
- **缓解**:scale scheduling(按去噪步动态调权重)、或 experimental 的 TIES/DARE 合并(要求 rank 一致)
- **必须先做小实验**:两两叠加 + 权重网格扫描,验证稳定性,再上三层

### C. FLUX.2 原生多参考图 — 角色一致性主力
- `Flux2Pipeline(image=[ref1, ref2, ...], prompt=...)`,最多 10 张
- Demo 价值高:给一张角色定妆图,在新场景/新姿态复用,**无需训练**
- ComfyUI 原生支持(FLUX.2 工作流)

### D. 评测门禁 + Model Registry — 企业级叙事
- 评测:golden prompts + 固定 seed → CLIP score(trigger 契合)+ 角色自相似度 + 黑图检测
- Registry:W&B Artifacts(版本+血缘+alias)+ S3 manifest + SSM 指针(双栈规划已设计)

### E. 独立推理服务 — Demo 可演示性
- ComfyUI(已有设计文档):FLUX.2 官方支持,预量化 fp8 文件,加载一次常驻,出图秒级,多 LoRA + 多参考图节点齐全
- 比训练入口推理(每次 ~13 分钟量化)快得多,适合现场演示反复调

---

## 三、技术风险与应对

| 风险 | 等级 | 应对 |
|------|------|------|
| **多层 LoRA 叠加冲突** | 🔴 高 | 先两两叠加实验 + 权重网格;角色一致性主力改用原生多参考图,降低对 LoRA 叠加的依赖 |
| **L40S 46GB 推理显存** | 🟡 中 | ComfyUI 用预量化 fp8 文件(避运行时量化);生产上 g7e 96GB |
| **g7e Blackwell 缺货** | 🟡 中 | Demo 阶段 L40S + ComfyUI 够用;生产再上 g7e |
| 数据量不足(分层后每层更少) | 🟡 中 | 每层至少 15-30 张代表性资产;Style 层可多些 |
| 角色一致性达不到客户预期 | 🟡 中 | 多参考图 + Character LoRA 双管;管理客户预期(一致性是"高度相似"非"像素级") |

---

## 四、Demo 实施路线(分阶段)

### Phase 1:分层训练能力(1-2 周)
- 数据按 Style/UI/Character 分组 + 自动标注(Qwen3-VL / Ideogram4)
- 三个 LoRA 独立训练(复用 ctl.py,rank 统一 32)
- 产出:三个分层 LoRA + W&B 版本化

### Phase 2:推理与组合验证(1-2 周,含核心风险验证)
- ComfyUI 独立推理机(已有设计)落地
- **多层 LoRA 叠加小实验**(两两 + 权重网格,验证稳定性)
- FLUX.2 原生多参考图验证(角色一致性)
- 产出:可演示的"prompt + 多层 LoRA + 参考图"生成

### Phase 3:企业级叙事补全(1 周)
- 评测门禁(CLIP + 一致性 + 黑图)
- Model Registry(W&B Artifacts + S3 manifest)
- FLUX.2 能力展示样图(Prompt Follow / 多元素 / 角色一致性)

### Phase 4:Demo 串讲与打磨
- 端到端故事线:资产 → 标注 → 分层训练 → 评估 → Registry → 多层组合推理 → 新资产
- 对照李佳 FLUX.1 旧案例讲"能力演进"

---

## 五、与现有资产的关系

**已有可复用**:
- 训练链路(ctl.py + CodeBuild + EBS 缓存 + arch:flux2 patch)— Phase 1 直接用
- 双栈架构规划(`dual-stack-plan.md`)— 协同/Registry/推理选型的依据
- ComfyUI 推理设计(`2026-06-28-comfyui-inference-design.md`)— Phase 2 推理落地
- SLOTIP LoRA — 作为 Style 层的起点

**需新建**:分层数据组织、自动标注接入、多层组合推理、多参考图、评测门禁、Registry 实现。

---

## 来源
- [FLUX.2-dev 模型卡(无需微调的角色/风格参考)](https://huggingface.co/black-forest-labs/FLUX.2-dev)
- [BFL FLUX.2 博客(10 张参考图)](https://bfl.ai/blog/flux-2)
- [diffusers Flux2Pipeline(多参考图 image 列表)](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux2)
- [diffusers 多 LoRA set_adapters / scale scheduling / TIES-DARE](https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference)
- [ai-toolkit captioner(Qwen3-VL / Ideogram4 结构化)](https://deepwiki.com/ostris/ai-toolkit/20.2-captioner-backends)
- [diffusers 评测(CLIP score)](https://huggingface.co/docs/diffusers/main/en/conceptual/evaluation)
- [ComfyUI FLUX.2-dev 教程](https://docs.comfy.org/tutorials/flux/flux-2-dev)
- 配套:[双栈架构规划](./dual-stack-plan.md)
