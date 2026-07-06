"""
Demo 生成矩阵 —— 展示分层 LoRA 微调的特点与优势(在 GPU 实例容器内跑)。

三个维度:
1. 单层对照: base(无LoRA) / 仅Style / 仅Character / Style+Character 组合
2. 主题: 通用新角色(海盗、龙) + 自定义 IP(美人鱼)—— 展示泛化 + 风格迁移
3. 权重梯度: 组合配置下 style/char 权重档位,展示可控性

输出: 每张图命名 <theme>__<config>.png + index.csv,便于拼对照图。
Run (容器内, diffusers 环境):
  python3 08_demo_matrix.py --style /loras/style.safetensors --char /loras/char.safetensors --out /exp
"""
import argparse
import csv
from pathlib import Path
import torch

SEED = 42  # 固定 seed,让不同配置可直接对照(唯一变量是 LoRA)

# 主题:prompt 里的 trigger 会按 config 动态加,这里只写主体描述
THEMES = {
    "pirate":  "a cartoon wolf pirate captain with an eyepatch and hat, treasure chest, gold coins, ship deck background",
    "dragon":  "a cute chubby dragon guarding a pile of gold coins, glowing gems, cave background",
    "mermaid": "a cute cartoon mermaid character with shimmering teal fish tail, pearl crown, holding a golden trident, gold coins and pearls around, coral reef background",
}

STYLE_TRIG = "slotstyle"
CHAR_TRIG = "slotchar"
COMMON_TAIL = "glossy 3D cartoon slot game art, vibrant colors, high quality"

# 单层对照 4 配置: (名字, style权重, char权重, 是否加style_trig, 是否加char_trig)
CONFIGS = [
    ("base",       0.0, 0.0, False, False),   # 无 LoRA,原始 FLUX.2
    ("style_only", 1.0, 0.0, True,  False),   # 仅风格
    ("char_only",  0.0, 1.0, False, True),    # 仅角色
    ("combo",      0.9, 0.8, True,  True),    # 组合(推荐权重)
]

# 组合权重梯度(只对 mermaid 跑,展示可控性),(name, ws, wc)
WEIGHT_SWEEP = [
    ("w_s07_c07", 0.7, 0.7),
    ("w_s09_c08", 0.9, 0.8),
    ("w_s10_c10", 1.0, 1.0),
]


def build_prompt(theme_body, use_style, use_char):
    trigs = []
    if use_style:
        trigs.append(STYLE_TRIG)
    if use_char:
        trigs.append(CHAR_TRIG)
    prefix = (" ".join(trigs) + ", ") if trigs else ""
    return f"{prefix}{theme_body}, {COMMON_TAIL}"


def load_pipeline():
    # Flux2Pipeline 需要项目训练容器里含 FLUX.2 的 diffusers 构建
    from diffusers import Flux2Pipeline
    pipe = Flux2Pipeline.from_pretrained(
        "black-forest-labs/FLUX.2-dev", torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()
    return pipe


def gen(pipe, prompt, out_path):
    g = torch.Generator("cpu").manual_seed(SEED)
    img = pipe(prompt, num_inference_steps=20, guidance_scale=3.5,
               width=768, height=768, generator=g).images[0]
    img.save(out_path)


def run(style_path, char_path, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pipe = load_pipeline()
    pipe.load_lora_weights(style_path, adapter_name="style")
    pipe.load_lora_weights(char_path, adapter_name="char")

    rows = []

    def set_w(ws, wc):
        pipe.set_adapters(["style", "char"], adapter_weights=[ws, wc])

    # 维度1+2: 每个主题 × 4 配置(单层对照)
    for theme, body in THEMES.items():
        for cname, ws, wc, us, uc in CONFIGS:
            set_w(ws, wc)
            prompt = build_prompt(body, us, uc)
            fn = f"{theme}__{cname}.png"
            gen(pipe, prompt, out / fn)
            rows.append({"theme": theme, "config": cname, "ws": ws, "wc": wc, "prompt": prompt, "file": fn})
            print(f"  saved {fn}")

    # 维度3: 美人鱼的权重梯度(展示可控性)
    for wname, ws, wc in WEIGHT_SWEEP:
        set_w(ws, wc)
        prompt = build_prompt(THEMES["mermaid"], True, True)
        fn = f"mermaid__{wname}.png"
        gen(pipe, prompt, out / fn)
        rows.append({"theme": "mermaid", "config": wname, "ws": ws, "wc": wc, "prompt": prompt, "file": fn})
        print(f"  saved {fn}")

    with open(out / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["theme", "config", "ws", "wc", "prompt", "file"])
        w.writeheader(); w.writerows(rows)
    print(f"\n✅ {len(rows)} images → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", required=True)
    ap.add_argument("--char", required=True)
    ap.add_argument("--out", default="/exp")
    a = ap.parse_args()
    run(a.style, a.char, a.out)
