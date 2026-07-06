# 分层 LoRA 训练 + 多层组合验证 — 设计文档

**日期**: 2026-06-29
**状态**: 待评审
**目标**: 为客户 Demo 关键路径(Phase 1 分层训练 + Phase 2 多层组合)产出可执行设计。验证"分层 LoRA 方法论 + 多层组合机制",证明企业级游戏美术 AI 生产平台的核心能力。

> 关联:`docs/architecture/demo-gap-analysis.md`(差距分析)、`docs/architecture/dual-stack-plan.md`(双栈规划)

---

## 一、范围与成功标准

**范围**:用现有 18 张游戏美术角色图(数据集 `slot-ip-v1`,统一厚涂 3D 卡通风,纯角色 symbol、无 UI 元素),演示分层 LoRA 的完整闭环。

- **做实**:Style 层 + Character 层(各一个 rank-32 LoRA)
- **方法论展示**:第三层(UI/Symbol/背景)作为可扩展占位 —— 分层方法论对任意层通用,今天用现有数据演示两层闭环
- **核心交付**:① 分层训练流程跑通 ② 两层 LoRA 组合的权重/冲突验证实验 ③ 可量化评估方法 + 推荐配置

**成功标准**:能演示"同一批数据 → 不同 caption 策略 → 两个解耦 LoRA → 加权组合生成既保画风又保角色的新图",并有实验数据说明组合权重如何调、冲突如何诊断。

**非目标**:出完美生产模型(数据仅 18 张);UI 层真实训练;IP-Adapter(FLUX.2 无,用原生多参考替代,不在本期);生产推理服务。

---

## 二、Phase 1:分层训练

### 核心机制
同一批 18 张图,用**两套不同 caption** 训出两个解耦 LoRA。原理(调研强共识):**没写进 caption 的共同特征会被"焊死"进 LoRA,写进 caption 的会被解耦成可控变量**(来源:[Civitai LoRA 指南](https://civitai.com/articles/4/making-a-lora-is-like-baking-a-cake))。

| 维度 | Style 层 | Character 层 |
|------|----------|--------------|
| trigger | `slotstyle` | `slotchar` |
| caption 策略 | 详细描述内容(角色/物体/构图/颜色),**不写画风** | 稀疏,**删除角色固有特征**,只留 trigger |
| 学到什么 | 纯画风(厚涂/高饱和/金币/描边),内容可控 | 角色身份焊进权重 |
| rank/alpha | 32/32 | 32/32(统一,为后续可合并:`add_weighted_adapter` 要求 identical rank) |
| steps | 1500-2000 | 1000-1500 + `sample_every:250` 挑 checkpoint(角色需早停防过拟合) |
| lr | 1e-4 cosine | 1e-4 cosine |
| 数据 | 18 张全用 | 18 张全用(身份层靠 caption 稀疏化锁共性风格特征) |

来源:[ai-toolkit](https://github.com/ostris/ai-toolkit)、[Civitai FLUX 教程](https://education.civitai.com/quickstart-guide-to-flux-1/)、[HF diffusers LoRA](https://huggingface.co/docs/diffusers/main/en/training/lora)

### 实现
复用现有 `ctl.py train` + `train_entry.py`,扩展:
- caption 集切换:为 Style/Character 各准备一套 `.txt`(同图不同 caption),放不同 S3 前缀(`datasets/slot-ip-v1-style/`、`datasets/slot-ip-v1-char/`)
- 参数化:`train_entry.py` 已支持 env 传 `TRIGGER_WORD`/`STEPS`/`RANK`;新增按层命名输出
- 产物:`s3://.../outputs/lora-style-<ts>/`、`lora-char-<ts>/`,各登记 W&B Artifact(alias `candidate`)

### 数据准备
- **自动标注**:ai-toolkit 内置 Qwen3-VL captioner 生成基础 caption,再人工/脚本按层策略调整(Style 详细、Character 稀疏)。来源:[ai-toolkit captioner](https://deepwiki.com/ostris/ai-toolkit/20.2-captioner-backends)
- 当前已有的 18 张 caption 偏详细(适合 Style 层);Character 层需生成稀疏版

---

## 三、Phase 2:多层组合验证实验

### 实验环境
diffusers 脚本化(非 ComfyUI):固定 seed 集(8 个)+ 固定 prompt 集,可量化、可复现、可批量网格扫描。
- 加载:`load_lora_weights(style)` + `load_lora_weights(char)`
- 组合:`set_adapters(["style","char"], adapter_weights=[ws, wc])`,固定整体 `scale=1.0` 只扫 adapter_weights
- 来源:[diffusers 多 LoRA](https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference)

### 实验阶段

**阶段 0 — 单层基线**:Style/Character 各自激活,权重扫 {0.6, 0.8, 1.0},固定 seed 出图,定每层单独最佳权重 w*。

**阶段 1 — 两两组合**:Style+Character 在各自 w* 附近做 3×3 网格({0.6,0.8,1.0}²),固定 seed。记录哪个权重区出现冲突。

**阶段 2 — 量化评估**(per-concept,调研指出必须分维度,否则掩盖"某概念被洗掉"):
- **Character 保真**:生成图 vs 角色参考的 CLIP image-image 相似度
- **Style 命中**:CLIP text-image score(prompt 含 slotstyle)或 vs 风格参考集
- **整体**:VLM(GPT-4V 类)rubric 打分(每概念 0-5 + 画质 0-5 + 污染/畸变标记)+ 人工把关
- 注:CLIP score 仅作同组内相对排序,不作绝对合格线
- 来源:[diffusers 评测](https://huggingface.co/docs/diffusers/main/en/conceptual/evaluation)、[Multi-LoRA Composition ACL 2024](https://arxiv.org/abs/2402.16843)

**阶段 3 — 补救**(若阶段 2 显示冲突,按成本递增):
1. **scale scheduling**(优先,官方支持):Character 前期高(锁身份)后期低(避污染),`callback_on_step_end` 每步调权重。官方有 FLUX 示例(1.5→0.2 衰减)
2. 分块加权:dict 形式对不同 transformer 块给不同 scale
3. (最后手段)TIES/DARE 合并:experimental,rank 须一致(已统一 32),density 0.5-0.8
- 来源:[diffusers scale scheduling](https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference)、[PEFT merging](https://huggingface.co/blog/peft_merging)

### 已知冲突模式与诊断
- **冲突模式**:style 风格污染 character 区域、特征互相覆盖、整体崩坏(来源:[ACL 2024 论文](https://arxiv.org/abs/2402.16843)证实加权和随 LoRA 数增加退化)
- **诊断法**:逐个关闭法 —— 用阶段 0 单层基线图对照,定位是哪层在干扰;`get_active_adapters()` 确认实际生效

### 交付
实验报告:权重组合 → per-concept 分数 → 推荐配置;Demo 时讲"如何科学调多层 LoRA"。

---

## 四、文件与产物结构

```
poc/
├── scripts/
│   ├── train_entry.py          (扩展:按层 caption/trigger/rank/输出命名)
│   ├── ctl.py                  (复用 train 命令)
│   ├── 06_prepare_layers.py    (新增:生成 Style/Character 两套 caption + 上传 S3)
│   └── 07_compose_experiment.py (新增:Phase 2 多层组合网格实验 + 评估)
├── dataset/
│   ├── slot-ip-v1-style/       (18 图 + 详细 caption)
│   └── slot-ip-v1-char/        (18 图 + 稀疏 caption)
docs/
└── experiments/
    └── layered-lora-results.md (Phase 2 实验报告,执行后产出)

S3: outputs/lora-style-<ts>/, outputs/lora-char-<ts>/
W&B: artifacts lora-style / lora-char (alias candidate)
```

---

## 五、风险与应对

| 风险 | 等级 | 应对 |
|------|------|------|
| 18 张训不出干净 Character 层(身份不一致) | 🔴 高 | 选同类角色子集;或承认是"风格变体"层,Demo 讲方法论;早停挑最佳 checkpoint |
| 两层组合冲突无法调和 | 🟡 中 | scale scheduling;实在不行用单层 + 多参考图演示一致性 |
| caption 稀疏化效果不明显(同图同内容) | 🟡 中 | 这是核心实验变量,做 A/B(详细 vs 稀疏 caption)对比,本身就是 Demo 内容 |
| L40S 46GB 推理 OOM | 🟢 低 | 已验证 fp8+卸载+768 可行;组合推理显存与单层相近 |
| FLUX.2 分层 LoRA 无成熟先例 | 🟡 中 | 调研数值来自 FLUX.1/SDXL 迁移;rank32/lr1e-4 作基线,靠样图校准 |

---

## 六、与现有资产关系

**复用**:ctl.py + CodeBuild + EBS 缓存 + arch:flux2 patch(训练链路);SLOTIP LoRA(作 Style 层起点参考)。
**新建**:06_prepare_layers.py(分层数据)、07_compose_experiment.py(组合实验)、train_entry.py 扩展。

---

## 来源汇总
- [ai-toolkit](https://github.com/ostris/ai-toolkit) / [captioner](https://deepwiki.com/ostris/ai-toolkit/20.2-captioner-backends)
- [Civitai LoRA 指南(caption 焊死原理)](https://civitai.com/articles/4/making-a-lora-is-like-baking-a-cake) / [FLUX 教程](https://education.civitai.com/quickstart-guide-to-flux-1/)
- [HF diffusers LoRA 训练](https://huggingface.co/docs/diffusers/main/en/training/lora) / [多 LoRA 组合](https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference) / [合并](https://huggingface.co/docs/diffusers/main/en/using-diffusers/merge_loras) / [评测](https://huggingface.co/docs/diffusers/main/en/conceptual/evaluation)
- [Multi-LoRA Composition (ACL 2024)](https://arxiv.org/abs/2402.16843) / [PEFT merging](https://huggingface.co/blog/peft_merging)
