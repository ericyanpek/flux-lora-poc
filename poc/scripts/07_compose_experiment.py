"""
多层 LoRA 组合验证实验(在 GPU 实例上跑)。
阶段 0 单层基线 → 阶段 1 两两网格 → 出图 + index.csv 供评估。
Run (on GPU instance, inside container with diffusers env):
  python3 07_compose_experiment.py --style /loras/style.safetensors --char /loras/char.safetensors --out /exp
"""
import argparse
import csv
import itertools
from pathlib import Path
import torch


SEEDS = [42, 123, 777, 2024]
PROMPTS = [
    "slotstyle slotchar, a treasure chest full of gold coins on a beach",
    "slotstyle slotchar, a dragon mascot in a fantasy forest",
    "slotstyle slotchar, a magic potion icon, vibrant game art",
]
STYLE_WEIGHTS = [0.6, 0.8, 1.0]
CHAR_WEIGHTS = [0.6, 0.8, 1.0]


def load_pipeline():
    # Flux2Pipeline requires the FLUX.2-enabled diffusers build inside the project
    # training container — NOT available in stock PyPI diffusers. Run this script
    # inside that container (see plan Task 7), not on a bare diffusers env.
    from diffusers import Flux2Pipeline
    pipe = Flux2Pipeline.from_pretrained(
        "black-forest-labs/FLUX.2-dev", torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()
    return pipe


def run(style_path, char_path, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pipe = load_pipeline()
    pipe.load_lora_weights(style_path, adapter_name="style")
    pipe.load_lora_weights(char_path, adapter_name="char")

    rows = []
    combos = [("style_only", [1.0, 0.0]), ("char_only", [0.0, 1.0])]
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
