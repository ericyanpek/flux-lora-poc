"""
单 LoRA Demo 出图 —— 用 ai-toolkit 的 StableDiffusion loader(已验证 46GB 可用路径)。
加载 base FLUX.2 + 一个 LoRA,对一批 slots 主题 prompt 出图。
用于展示单层 LoRA(style 或 char)的效果。

Run INSIDE the training container:
  python3 single_lora_demo.py --lora /loras/style.safetensors --trigger slotstyle --tag style --out /exp
  python3 single_lora_demo.py --lora /loras/char.safetensors  --trigger slotchar  --tag char  --out /exp
"""
import os, sys, gc, argparse
sys.path.insert(0, "/ai-toolkit")
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--lora", required=True)
ap.add_argument("--trigger", required=True)   # slotstyle 或 slotchar
ap.add_argument("--tag", required=True)        # 输出文件名前缀 style/char
ap.add_argument("--out", default="/exp")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

# slots 主题 prompt:通用新角色 + 自定义美人鱼 IP,统一带 trigger
T = a.trigger
TAIL = "glossy 3D cartoon slot game art, vibrant colors, gold coins, high quality"
PROMPTS = [
    (f"{T}, a cartoon wolf pirate captain with eyepatch and hat, treasure chest, {TAIL}", "pirate"),
    (f"{T}, a cute chubby dragon guarding a pile of gold coins, glowing gems, {TAIL}", "dragon"),
    (f"{T}, a cute mermaid with shimmering teal fish tail, pearl crown, holding a golden trident, coral reef, pearls, {TAIL}", "mermaid"),
]
SEED, STEPS, GUIDANCE, W, H = 42, 25, 3.5, 768, 768

from toolkit.config_modules import ModelConfig, GenerateImageConfig
from toolkit.stable_diffusion_model import StableDiffusion

print(f"=== Loading FLUX.2 + LoRA ({a.tag}) via ai-toolkit loader ===", flush=True)
model_config = ModelConfig(
    name_or_path="black-forest-labs/FLUX.2-dev",
    arch="flux2",
    quantize=True, quantize_te=True, low_vram=True, dtype="bf16",
    lora_path=a.lora,     # 单 LoRA 走 load_model 内的 fuse 路径
)
sd = StableDiffusion(device="cuda:0", model_config=model_config, dtype="bf16")
sd.load_model()
print("=== Model + LoRA loaded ===", flush=True)

gen_configs = []
for prompt, name in PROMPTS:
    out_path = f"{a.out}/{name}__{a.tag}.png"
    gen_configs.append(GenerateImageConfig(
        prompt=prompt, width=W, height=H, num_inference_steps=STEPS,
        guidance_scale=GUIDANCE, seed=SEED, output_path=out_path,
    ))
print(f"=== Generating {len(gen_configs)} images ({a.tag}) ===", flush=True)
sd.generate_images(gen_configs)
print(f"=== Done ({a.tag}) ===", flush=True)
print(os.listdir(a.out), flush=True)
