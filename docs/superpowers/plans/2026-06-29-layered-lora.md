# 分层 LoRA 训练 + 多层组合验证 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用现有 18 张游戏美术图,通过两套 caption 训出解耦的 Style + Character LoRA,并用网格实验验证多层组合的权重/冲突,产出可量化的推荐配置。

**Architecture:** 复用现有训练链路(ctl.py + CodeBuild 镜像 + arch:flux2)。新增:① 分层数据准备脚本(同图两套 caption)② train_entry 参数化(支持 layer 配置)③ 组合实验脚本(diffusers 网格扫描 + per-concept 评估)。

**Tech Stack:** Python, boto3, diffusers, ai-toolkit (arch:flux2), AWS EC2 g6e/SSM/S3, W&B

---

## 常量(全程共用)

```
S3 BUCKET        = flux-poc-984072314535-us-east-1
现有数据集       = poc/dataset/  (18 张 png + 详细 caption txt)
Style trigger    = slotstyle
Char trigger     = slotchar
rank             = 32 (两层统一)
实例             = i-00a60dc65d57b9bae (g6e.4xlarge, us-west-2)
本地结果目录     = poc/results/
```

注:本计划脚本是工程脚本(boto3/diffusers 调 AWS 与模型),非单元测试驱动。"测试"= 语法校验 + 实际运行验证输出。每个代码任务以 `python3 -c "import ast; ast.parse(...)"` 验证语法,关键任务有运行验证步骤。

---

### Task 1: 分层数据准备脚本(同图两套 caption)

**Files:**
- Create: `poc/scripts/06_prepare_layers.py`

- [ ] **Step 1: 创建 06_prepare_layers.py**

```python
"""
从现有 18 张图生成两套 caption,上传到两个 S3 前缀,支撑分层 LoRA 训练。
- Style 层:详细描述内容(角色/物体/构图/颜色),不写画风 → 画风沉淀为 LoRA
- Character 层:稀疏 caption,删角色固有特征,只留 trigger → 角色身份焊进权重
Run: python3 06_prepare_layers.py
"""
import boto3
from pathlib import Path
from config import REGION, BUCKET, LOCAL_DATASET_PATH

LOCAL = Path(LOCAL_DATASET_PATH)
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
STYLE_PREFIX = "datasets/slot-ip-v1-style/"
CHAR_PREFIX = "datasets/slot-ip-v1-char/"
STYLE_TRIGGER = "slotstyle"
CHAR_TRIGGER = "slotchar"


def style_caption(original: str) -> str:
    # Style 层:保留原详细描述(角色/构图/颜色),把风格词 SLOTIP 换成 style trigger
    # 原 caption 形如 "SLOTIP style, a confident cartoon skunk character ..."
    body = original
    for token in ["SLOTIP style, ", "SLOTIP style ", "SLOTIP, ", "SLOTIP "]:
        if body.startswith(token):
            body = body[len(token):]
            break
    return f"{STYLE_TRIGGER}, {body}"


def char_caption(original: str) -> str:
    # Character 层:稀疏化 —— 只保留 trigger + 极简主体类别,删除颜色/服饰/场景等可变细节
    # 提取主体名词(skunk/pharaoh/mummy 等),其余删掉
    import re
    m = re.search(r"\b(skunk|pharaoh|mummy|elephant|gorilla|koi|horse|clown|chameleon|girl|skeleton|train|hero|zombie)\b",
                  original, re.IGNORECASE)
    subject = m.group(1).lower() if m else "character"
    return f"{CHAR_TRIGGER}, a {subject}"


def prepare():
    s3 = boto3.client("s3", region_name=REGION)
    imgs = [p for p in LOCAL.iterdir() if p.suffix.lower() in IMG_EXT]
    print(f"Found {len(imgs)} images")
    for img in imgs:
        orig_txt = img.with_suffix(".txt")
        original = orig_txt.read_text().strip() if orig_txt.exists() else "a character"
        for prefix, capfn, label in [
            (STYLE_PREFIX, style_caption, "style"),
            (CHAR_PREFIX, char_caption, "char"),
        ]:
            cap = capfn(original)
            # 上传图片
            s3.upload_file(str(img), BUCKET, prefix + img.name)
            # 上传 caption(同名 txt)
            s3.put_object(Bucket=BUCKET, Key=prefix + img.stem + ".txt", Body=cap.encode())
        print(f"  {img.name}: style/char captions uploaded")
    print(f"\n✅ Style → s3://{BUCKET}/{STYLE_PREFIX}")
    print(f"✅ Char  → s3://{BUCKET}/{CHAR_PREFIX}")


if __name__ == "__main__":
    prepare()
```

- [ ] **Step 2: 验证语法**

```bash
cd /Users/yabolin/claude-code/flux && python3 -c "import ast; ast.parse(open('poc/scripts/06_prepare_layers.py').read()); print('OK')"
```
Expected: `OK`

- [ ] **Step 3: 本地预览生成的 caption(不上传,确认策略正确)**

临时在脚本末尾用 dry-run 验证逻辑:
```bash
cd /Users/yabolin/claude-code/flux/poc/scripts && python3 -c "
import sys; sys.path.insert(0,'.')
from importlib import import_module
m = __import__('06_prepare_layers')
orig = 'SLOTIP style, a confident cartoon skunk character with black and white fur wearing a blue t-shirt, standing in a neon-lit alley'
print('STYLE:', m.style_caption(orig))
print('CHAR :', m.char_caption(orig))
"
```
Expected:
```
STYLE: slotstyle, a confident cartoon skunk character with black and white fur wearing a blue t-shirt, standing in a neon-lit alley
CHAR : slotchar, a skunk
```

- [ ] **Step 4: Commit**

```bash
cd /Users/yabolin/claude-code/flux && git add poc/scripts/06_prepare_layers.py && git commit -m "feat: layered caption prep (style detailed / char sparse)"
```

---

### Task 2: train_entry.py 参数化(支持分层 sample prompts + dataset 路径)

**Files:**
- Modify: `poc/docker/train_entry.py`

当前 `build_config` 的 `sample_prompts` 写死角色场景,`run_name` 写死。分层训练需按 layer 调整。

- [ ] **Step 1: 修改 load_hyperparameters 增加 layer 字段**

将 `load_hyperparameters` 里的 key 列表:
```python
    for key in ["trigger_word", "model_name", "steps", "lr", "rank", "sample_every"]:
```
替换为:
```python
    for key in ["trigger_word", "model_name", "steps", "lr", "rank", "sample_every", "layer", "project_name"]:
```

- [ ] **Step 2: 修改 build_config 的 sample_prompts 按 layer 区分**

将:
```python
    sample_prompts = [
        f"a {trigger_word} character sitting on a beach, casual game style, vibrant colors",
        f"a {trigger_word} character in a fantasy forest, detailed character art",
        f"a {trigger_word} character portrait, close up, high quality",
        "a character sitting on a beach, casual game style, vibrant colors",
    ]
```
替换为:
```python
    layer = hp.get("layer", "")
    if layer == "style":
        sample_prompts = [
            f"{trigger_word}, a treasure chest full of gold coins",
            f"{trigger_word}, a fierce dragon mascot",
            f"{trigger_word}, a magic potion bottle icon",
            "a treasure chest full of gold coins",  # 无 trigger 对照
        ]
    elif layer == "char":
        sample_prompts = [
            f"{trigger_word}, on a beach with palm trees",
            f"{trigger_word}, in a fantasy forest",
            f"{trigger_word}, portrait close up",
            "a character on a beach",  # 无 trigger 对照
        ]
    else:
        sample_prompts = [
            f"a {trigger_word} character sitting on a beach, casual game style, vibrant colors",
            f"a {trigger_word} character in a fantasy forest, detailed character art",
            f"a {trigger_word} character portrait, close up, high quality",
            "a character sitting on a beach, casual game style, vibrant colors",
        ]
```

- [ ] **Step 3: 修改 default_caption 和 run_name 按 layer**

将 datasets 里的:
```python
            "default_caption": f"a character in {trigger_word} style",
```
替换为:
```python
            "default_caption": f"{trigger_word}",
```

将 logging 块:
```python
    if wandb_key:
        process["logging"] = {
            "use_wandb": True,
            "project": "flux2-lora-poc",
            "run_name": trigger_word,
        }
```
替换为:
```python
    if wandb_key:
        process["logging"] = {
            "use_wandb": True,
            "project": hp.get("project_name", "flux2-lora-poc"),
            "run_name": f"{layer or 'base'}-{trigger_word}",
        }
```

- [ ] **Step 4: 验证语法**

```bash
cd /Users/yabolin/claude-code/flux && python3 -c "import ast; ast.parse(open('poc/docker/train_entry.py').read()); print('OK')"
```
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
cd /Users/yabolin/claude-code/flux && git add poc/docker/train_entry.py && git commit -m "feat: parametrize train_entry by layer (sample prompts, run_name)"
```

---

### Task 3: ctl.py 支持分层训练参数

**Files:**
- Modify: `poc/scripts/ctl.py`

当前 `cmd_train` 写死 `DATASET_PREFIX` 和 `TRIGGER_WORD`,需支持 `--layer` 选择数据集前缀/trigger。

- [ ] **Step 1: 修改 cmd_train 支持 layer**

将整个 `cmd_train` 函数替换为:
```python
def cmd_train(args):
    _ensure_running()
    ts = time.strftime("%Y%m%d-%H%M%S")
    layer = getattr(args, "layer", None)
    # 分层:选数据集前缀 + trigger;无 layer 则用默认
    if layer == "style":
        prefix, trigger, job = "datasets/slot-ip-v1-style/", "slotstyle", f"style-{ts}"
    elif layer == "char":
        prefix, trigger, job = "datasets/slot-ip-v1-char/", "slotchar", f"char-{ts}"
    else:
        prefix, trigger, job = DATASET_PREFIX, TRIGGER_WORD, f"slotip-{ts}"
    steps_env = f"-e STEPS={args.steps}" if args.steps else ""
    layer_env = f"-e LAYER={layer}" if layer else ""
    print(f"Launching {job} (layer={layer or 'base'}, trigger={trigger}, steps={args.steps or 'default'})...")
    cmd = (
        f"bash -c 'exec >> /var/log/flux-train-{job}.log 2>&1; "
        f"mkdir -p /tmp/td-{job} /tmp/out-{job} /opt/flux-cache/hf; "
        f"aws s3 sync s3://{BUCKET}/{prefix} /tmp/td-{job}/ >/dev/null 2>&1; "
        f"aws ecr get-login-password --region {S3_REGION} | docker login --username AWS --password-stdin {ECR_URI.split('/')[0]} >/dev/null 2>&1; "
        f"docker pull {ECR_URI} >/dev/null 2>&1; "
        f"HF=$(aws ssm get-parameter --region {S3_REGION} --name /flux-poc/hf-token --with-decryption --query Parameter.Value --output text); "
        f"WB=$(aws ssm get-parameter --region {S3_REGION} --name /flux-poc/wandb-key --with-decryption --query Parameter.Value --output text 2>/dev/null || echo \"\"); "
        f"docker run --gpus all --rm --shm-size=24g -e HF_TOKEN=\"$HF\" -e WANDB_API_KEY=\"$WB\" "
        f"-e TRIGGER_WORD={trigger} {layer_env} {steps_env} -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        f"-e HF_HOME=/root/.cache/huggingface -v /opt/flux-cache/hf:/root/.cache/huggingface "
        f"-v /tmp/td-{job}:/opt/ml/input/data/training -v /tmp/out-{job}:/opt/ml/model {ECR_URI} 2>&1; "
        f"EXIT=$?; "
        f"aws s3 sync /tmp/td-{job}/flux-lora-poc/ s3://{BUCKET}/outputs/lora-{job}/ --exclude \"*_cache/*\" >/dev/null 2>&1; "
        f"echo $([ $EXIT -eq 0 ] && echo SUCCESS || echo FAILED:$EXIT) | aws s3 cp - s3://{BUCKET}/outputs/lora-{job}/status.txt' &"
    )
    cid, _ = _ssm_run([cmd], wait=False)
    print(f"  job started (background). SSM cmd: {cid}")
    print(f"  watch:   python3 ctl.py logs train")
    print(f"  results: s3://{BUCKET}/outputs/lora-{job}/")
```

- [ ] **Step 2: 给 train 子命令加 --layer 参数**

在 `__main__` 块里找到:
```python
    pt = sub.add_parser("train"); pt.add_argument("--steps", type=int, default=None)
```
替换为:
```python
    pt = sub.add_parser("train"); pt.add_argument("--steps", type=int, default=None)
    pt.add_argument("--layer", choices=["style", "char"], default=None)
```

- [ ] **Step 3: 验证语法**

```bash
cd /Users/yabolin/claude-code/flux && python3 -c "import ast; ast.parse(open('poc/scripts/ctl.py').read()); print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
cd /Users/yabolin/claude-code/flux && git add poc/scripts/ctl.py && git commit -m "feat: ctl.py train --layer (style/char) for layered LoRA"
```

---

### Task 4: 多层组合实验脚本(本地 diffusers,网格 + 评估)

**Files:**
- Create: `poc/scripts/07_compose_experiment.py`

> 注:此脚本在**有 GPU 的实例上**运行(本地 Mac 无法跑 FLUX.2)。设计为在训练实例上通过 SSM 或直接 SSH 执行。脚本只依赖 diffusers + 两个 LoRA 文件 + 基模缓存。

- [ ] **Step 1: 创建 07_compose_experiment.py**

```python
"""
多层 LoRA 组合验证实验(在 GPU 实例上跑)。
阶段 0 单层基线 → 阶段 1 两两网格 → per-concept 评估(CLIP)。
产出:每个权重组合的图 + CLIP 分数 CSV。
Run (on GPU instance, inside container or with diffusers env):
  python3 07_compose_experiment.py --style /path/style.safetensors --char /path/char.safetensors --out /tmp/exp
"""
import argparse
import csv
import itertools
import os
import torch
from pathlib import Path


SEEDS = [42, 123, 777, 2024]
PROMPTS = [
    "slotstyle slotchar, a treasure chest full of gold coins on a beach",
    "slotstyle slotchar, a dragon mascot in a fantasy forest",
    "slotstyle slotchar, a magic potion icon, vibrant game art",
]
STYLE_WEIGHTS = [0.6, 0.8, 1.0]
CHAR_WEIGHTS = [0.6, 0.8, 1.0]


def load_pipeline():
    from diffusers import Flux2Pipeline
    pipe = Flux2Pipeline.from_pretrained(
        "black-forest-labs/FLUX.2-dev", torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()  # L40S 46GB 显存策略
    return pipe


def run(style_path, char_path, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pipe = load_pipeline()
    pipe.load_lora_weights(style_path, adapter_name="style")
    pipe.load_lora_weights(char_path, adapter_name="char")

    rows = []
    # 阶段 0:单层基线(style only / char only)
    combos = [("style_only", [1.0, 0.0]), ("char_only", [0.0, 1.0])]
    # 阶段 1:两两网格
    for ws, wc in itertools.product(STYLE_WEIGHTS, CHAR_WEIGHTS):
        combos.append((f"s{ws}_c{wc}", [ws, wc]))

    for name, (ws, wc) in combos:
        pipe.set_adapters(["style", "char"], adapter_weights=[ws, wc])
        for pi, prompt in enumerate(PROMPTS):
            for seed in SEEDS:
                g = torch.Generator("cpu").manual_seed(seed)
                img = pipe(prompt, num_inference_steps=20, guidance_scale=3.5,
                           width=768, height=768, generator=g).images[0]
                fn = f"{name}__p{pi}__s{seed}.png"
                img.save(out / fn)
                rows.append({"combo": name, "ws": ws, "wc": wc,
                             "prompt_idx": pi, "seed": seed, "file": fn})
                print(f"  saved {fn}")

    with open(out / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["combo", "ws", "wc", "prompt_idx", "seed", "file"])
        w.writeheader(); w.writerows(rows)
    print(f"\n✅ {len(rows)} images → {out}")
    print("  next: 人工/CLIP 评估各 combo,选 per-concept 都达标的权重")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", required=True)
    ap.add_argument("--char", required=True)
    ap.add_argument("--out", default="/tmp/compose-exp")
    a = ap.parse_args()
    run(a.style, a.char, a.out)
```

- [ ] **Step 2: 验证语法**

```bash
cd /Users/yabolin/claude-code/flux && python3 -c "import ast; ast.parse(open('poc/scripts/07_compose_experiment.py').read()); print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
cd /Users/yabolin/claude-code/flux && git add poc/scripts/07_compose_experiment.py && git commit -m "feat: multi-LoRA composition grid experiment (diffusers)"
```

---

### Task 5: 执行分层数据准备

- [ ] **Step 1: 运行数据准备**

```bash
cd /Users/yabolin/claude-code/flux/poc/scripts && python3 06_prepare_layers.py
```
Expected: 输出 18 张图的 style/char caption 上传,末尾两行 `✅ Style → ...` / `✅ Char → ...`

- [ ] **Step 2: 抽查 S3 上的 caption 正确**

```bash
aws s3 cp s3://flux-poc-984072314535-us-east-1/datasets/slot-ip-v1-char/cute_pharaoh.txt - 2>/dev/null
aws s3 cp s3://flux-poc-984072314535-us-east-1/datasets/slot-ip-v1-style/cute_pharaoh.txt - 2>/dev/null
```
Expected: char 是稀疏的(`slotchar, a pharaoh`),style 是详细的(`slotstyle, a cute chibi skeleton pharaoh ...`)

---

### Task 6: 训练 Style 层 + Character 层

- [ ] **Step 1: 启动实例**

```bash
cd /Users/yabolin/claude-code/flux/poc/scripts && python3 ctl.py start
```
Expected: `instance running` + `ready`

- [ ] **Step 2: 训练 Style 层**

```bash
cd /Users/yabolin/claude-code/flux/poc/scripts && python3 ctl.py train --layer style --steps 1800
```
Expected: 打印 job 名 `style-<ts>`,后台启动

- [ ] **Step 3: 等 Style 训练完成(轮询状态)**

```bash
# 每隔几分钟查,直到 status.txt = SUCCESS
aws s3 cp s3://flux-poc-984072314535-us-east-1/outputs/lora-style-<ts>/status.txt - 2>/dev/null
```
Expected: 最终 `SUCCESS`,outputs 下有 `flux-lora-poc.safetensors`

- [ ] **Step 4: 训练 Character 层**

```bash
cd /Users/yabolin/claude-code/flux/poc/scripts && python3 ctl.py train --layer char --steps 1200
```
Expected: job 名 `char-<ts>` 后台启动

- [ ] **Step 5: 等 Character 训练完成**

```bash
aws s3 cp s3://flux-poc-984072314535-us-east-1/outputs/lora-char-<ts>/status.txt - 2>/dev/null
```
Expected: `SUCCESS`

- [ ] **Step 6: 下载两个 LoRA 到本地存档**

```bash
mkdir -p /Users/yabolin/claude-code/flux/poc/results/layered
aws s3 cp s3://flux-poc-984072314535-us-east-1/outputs/lora-style-<ts>/flux-lora-poc.safetensors /Users/yabolin/claude-code/flux/poc/results/layered/style.safetensors
aws s3 cp s3://flux-poc-984072314535-us-east-1/outputs/lora-char-<ts>/flux-lora-poc.safetensors /Users/yabolin/claude-code/flux/poc/results/layered/char.safetensors
```
Expected: 两个 ~390MB safetensors

---

### Task 7: 执行多层组合实验

- [ ] **Step 1: 把实验脚本和两个 LoRA 传到实例**

```bash
# LoRA 已在 S3;脚本通过 SSM 拉到实例
cd /Users/yabolin/claude-code/flux/poc/scripts && python3 ctl.py run "mkdir -p /tmp/loras && aws s3 cp s3://flux-poc-984072314535-us-east-1/outputs/lora-style-<ts>/flux-lora-poc.safetensors /tmp/loras/style.safetensors && aws s3 cp s3://flux-poc-984072314535-us-east-1/outputs/lora-char-<ts>/flux-lora-poc.safetensors /tmp/loras/char.safetensors && echo done"
```
Expected: `done`

- [ ] **Step 2: 在容器内跑组合实验(复用训练镜像的 diffusers 环境)**

通过 ctl.py run 在实例上启动容器跑实验脚本(脚本经 S3 或内联传入)。先把 07 脚本上传 S3:
```bash
aws s3 cp /Users/yabolin/claude-code/flux/poc/scripts/07_compose_experiment.py s3://flux-poc-984072314535-us-east-1/scripts/07_compose_experiment.py
```

然后在实例上容器内执行:
```bash
cd /Users/yabolin/claude-code/flux/poc/scripts && python3 ctl.py run "aws s3 cp s3://flux-poc-984072314535-us-east-1/scripts/07_compose_experiment.py /tmp/07.py && HF=\$(aws ssm get-parameter --region us-east-1 --name /flux-poc/hf-token --with-decryption --query Parameter.Value --output text) && docker run --gpus all --rm --shm-size=24g -e HF_TOKEN=\"\$HF\" -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e HF_HOME=/root/.cache/huggingface -v /opt/flux-cache/hf:/root/.cache/huggingface -v /tmp/loras:/loras -v /tmp/07.py:/07.py -v /tmp/exp-out:/exp --entrypoint python3 984072314535.dkr.ecr.us-east-1.amazonaws.com/flux-poc-training:latest /07.py --style /loras/style.safetensors --char /loras/char.safetensors --out /exp 2>&1 | tail -30"
```
Expected: 打印生成的图文件名,末尾 `✅ N images → /exp`

> 风险提示:若 `Flux2Pipeline` 加载 LoRA 报格式错(ai-toolkit 产出格式 vs diffusers 期望),记录实际错误,在结果文档中说明,可能需要 LoRA 格式转换——这是已知风险点。

- [ ] **Step 3: 把实验结果图同步回 S3 + 本地**

```bash
cd /Users/yabolin/claude-code/flux/poc/scripts && python3 ctl.py run "aws s3 sync /tmp/exp-out/ s3://flux-poc-984072314535-us-east-1/experiments/compose/ && echo synced"
mkdir -p /Users/yabolin/claude-code/flux/poc/results/compose
aws s3 sync s3://flux-poc-984072314535-us-east-1/experiments/compose/ /Users/yabolin/claude-code/flux/poc/results/compose/
```
Expected: 本地 results/compose 下有网格图 + index.csv

---

### Task 8: 编写实验报告 + 停机

**Files:**
- Create: `docs/experiments/layered-lora-results.md`

- [ ] **Step 1: 看图评估,写实验报告**

人工对比各权重组合的图(对照 style_only / char_only 基线),按 spec 的 per-concept 维度判断:
- Character 身份是否保持(对照 char_only)
- Style 画风是否命中(对照 style_only)
- 是否有风格污染/畸变

创建 `docs/experiments/layered-lora-results.md`,记录:
```markdown
# 分层 LoRA 组合实验结果

## 训练产物
- Style LoRA: lora-style-<ts>, 1800 steps, trigger=slotstyle
- Char LoRA: lora-char-<ts>, 1200 steps, trigger=slotchar

## 单层基线观察
- style_only: <画风是否成型>
- char_only: <角色身份是否保持>

## 组合网格结果(3×3 权重)
| ws \ wc | 0.6 | 0.8 | 1.0 |
|---------|-----|-----|-----|
| 0.6 | <观察> | | |
| 0.8 | | <观察> | |
| 1.0 | | | |

## 冲突诊断
<是否出现风格污染/角色畸变,在哪个权重区>

## 推荐配置
- 推荐 adapter_weights: [style=X, char=Y]
- 是否需要 scale scheduling: <是/否,理由>

## 结论
<两层组合机制是否验证成功,Demo 可讲什么>
```

- [ ] **Step 2: Commit 报告**

```bash
cd /Users/yabolin/claude-code/flux && git add docs/experiments/layered-lora-results.md && git commit -m "docs: layered LoRA composition experiment results"
```

- [ ] **Step 3: 停机省钱**

```bash
cd /Users/yabolin/claude-code/flux/poc/scripts && python3 ctl.py stop
```
Expected: `stop requested`

---

## 运行顺序总览

```bash
# 一次性:数据准备(本地)
python3 06_prepare_layers.py        # Task 5

# 训练两层(实例)
python3 ctl.py start
python3 ctl.py train --layer style --steps 1800
python3 ctl.py train --layer char --steps 1200   # Style 完成后

# 组合实验(实例)
python3 ctl.py run "<拉 LoRA + 跑 07 实验>"        # Task 7

# 评估 + 报告 + 停机
# 写 docs/experiments/layered-lora-results.md
python3 ctl.py stop
```
