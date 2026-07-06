# 分层 LoRA 训练 + 多层组合 —— 实验报告

> **状态**:✅ 已跑通,有产物。2026-07-06。
> **产物位置**:S3 `s3://flux-poc-<account>-us-east-1/demo/comfyui-matrix/`;本地 `demo_output/comfyui-matrix/`。
> **对应计划**:`docs/superpowers/plans/2026-06-29-layered-lora.md` Task 4/5/7。

本报告是外部走查(2026-07-06 HANDOFF)要求的"分层 LoRA 真训出且解耦"与"多层组合"的产物背书。**结论如实记录,包括未完全达标的部分。**

---

## 1. 实验设置

### 训练产物(两个 rank-32 LoRA,各 390MB)

| 层 | 触发词 | caption 策略 | S3 路径 |
|----|--------|-------------|---------|
| Style | `slotstyle` | 详细 caption(把角色/物体都写进去)→ 模型学"没被描述的共性" = 画风 | `outputs/lora-style-20260706-112005/flux-lora-poc.safetensors` |
| Character | `slotchar` | 稀疏 caption(只留主体名)→ 把更多视觉特征"焊进" LoRA | `outputs/lora-char-20260706-134937/flux-lora-poc.safetensors` |

两层都从**同一批 18 张游戏美术图**训练,只是 caption 策略不同(见 `poc/scripts/06_prepare_layers.py`)。均 rank-32,便于加权叠加。

### 推理设置

- 独立 g6e.4xlarge ComfyUI 推理机(官方 fp8 预量化底模 + `LoraLoader` 串接),见 `poc/scripts/inference/comfy_gen.py`。
- 固定 seed=42、25 步、guidance=3.5、1024×1024、euler/simple。**唯一变量是 LoRA 配置**,以便直接对照。

### 对照矩阵:3 主题 × 4 配置 = 12 图

- **主题**:`pirate`(通用新角色:海盗狼船长)、`dragon`(通用新角色:胖龙)、`mermaid`(**自定义 IP**:美人鱼)。
- **配置**:
  - `base` — 无 LoRA(原始 FLUX.2)
  - `style` — 仅 Style LoRA(强度 1.0)
  - `char` — 仅 Character LoRA(强度 1.0)
  - `combo` — Style 0.9 + Character 0.8 叠加

---

## 2. 观察结果(逐主题)

### mermaid(自定义 IP,最能说明问题)

base → style → char → combo(固定 seed=42,唯一变量是 LoRA 配置):

| base | style | char | combo |
|------|-------|------|-------|
| ![](images/mermaid_base.png) | ![](images/mermaid_style.png) | ![](images/mermaid_char.png) | ![](images/mermaid_combo.png) |

| 配置 | 观察 |
|------|------|
| base | 通用可爱美人鱼立绘,画质好但只是"独立角色",无游戏氛围 |
| style | 同一美人鱼被渲染成**目标游戏美术风格**:出现游戏面板网格、数字符号、UI 排布。**风格迁移明确生效** |
| char | 也带目标游戏美学 + 更强的面板/图标框架(见下"解耦"讨论) |
| combo | 兼具画风与角色:美人鱼 + 完整游戏面板 + 金币/珍珠氛围,可直接作为素材 |

### dragon / pirate(通用新角色,验证泛化)

| dragon base | dragon combo | pirate combo |
|-------------|--------------|--------------|
| ![](images/dragon_base.png) | ![](images/dragon_combo.png) | ![](images/pirate_combo.png) |

- `base`:柔和插画风,漂亮但普通。
- `combo`:训练集里**从没有过**的龙/海盗狼,照样套上目标游戏美术风格(霓虹高饱和 + 转轮框架 + UI 文字 + 宝箱/金币)。**证明风格能泛化到新角色 = 可复用生产力。**

---

## 3. 诚实结论(达标 / 未达标)

**✅ 达标:**
1. 两个 LoRA 真训出、真加载、真生效——同 seed 下 base 与 style 输出 md5 不同,肉眼差异显著(不是静默跳过)。
2. 风格迁移对**训练集外的新角色**泛化成功(dragon/pirate)。
3. 多层 combo 叠加可用,无加载报错(fp8 底模 + bf16 LoRA 兼容性 OK,见 `reference_comfyui_inference` memory)。

**⚠️ 未完全达标 / 需注意:**
1. **两层未完全解耦**。因为 style 和 char 用**同一批 18 张图**训练,`char`-only 也带明显的目标画风,不是"纯角色身份、无画风"。要真正解耦需让 char 层用**跨风格的同角色图**训练——当前数据集不具备。目前 style/char 的差异更多是"caption 密度导致的特征保留程度"差异,而非"画风 vs 身份"的干净切分。
2. **触发词渗进画面文字**。触发词有时被渲染成画面里的文字(如乱码字样)。因目标美术本身含大量文字/数字元素,模型把触发词当成了可渲染文本。生产中应换成不易被当文字的触发词,或在 caption 中更明确隔离。
3. **未做定量评分**。本轮是视觉对照,没跑 CLIP-score / 人工 rubric 网格。`07_compose_experiment.py` 的 3×3 权重网格定量评估仍是 TODO。

---

## 4. 对 Demo 的建议

- **主打**:mermaid 的 base→style→combo 三联对照 + dragon/pirate 的 base→combo 泛化对照。故事线"同一角色,加了我们训练的 LoRA 后变成可直接用的游戏素材"最直观。
- **补充路线**:角色一致性(同角色多场景)另有一条不依赖训练的路径——FLUX.2 原生多参考图,见 `docs/superpowers/specs/2026-06-30-multiref-inference-design.md`(第二条互补主线)。
- **不要过度声称**"完美解耦的风格/角色双层"——按第 3 节口径说明"分层可训、组合可用、完全解耦受限于单一数据集"。

---

## 5. 复现

```bash
# 训练两层(需 GPU 训练机)
python3 poc/scripts/06_prepare_layers.py            # 生成 style/char 两套 caption
python3 poc/scripts/ctl.py train --layer style
python3 poc/scripts/ctl.py train --layer char

# 出对照矩阵(在 ComfyUI 推理机上)
python3 poc/scripts/07_deploy_comfyui.py            # 自动拉 latest style/char LoRA
# 每个 config 一次:
python3 poc/scripts/inference/comfy_gen.py --config base  --out /exp/base
python3 poc/scripts/inference/comfy_gen.py --config style --out /exp/style
python3 poc/scripts/inference/comfy_gen.py --config char  --out /exp/char
python3 poc/scripts/inference/comfy_gen.py --config combo --out /exp/combo
```
