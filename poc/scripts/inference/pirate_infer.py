"""
Minimal FLUX.2 + LoRA inference using ai-toolkit's loader (same path as training).
Loads base model once, applies the LoRA, generates pirate-theme images for
multiple checkpoints so we can compare. Runs INSIDE the training container.
"""
import os, sys, gc
sys.path.insert(0, "/ai-toolkit")
import torch

LORA_DIR = "/tmp/training-data/flux-lora-poc"
OUT_DIR = "/tmp/pirate-eval"
os.makedirs(OUT_DIR, exist_ok=True)

# Pirate-theme eval prompts (with trigger word SLOTIP)
PROMPTS = [
    "SLOTIP style, a fierce pirate captain with a tricorn hat and golden coins, treasure chest overflowing with gold, ship deck background, vibrant slot game key art",
    "SLOTIP style, a pirate ship sailing on stormy seas with golden treasure glowing, dramatic lighting, casino slot game art",
    "SLOTIP style, a cute pirate parrot holding a gold coin, tropical island and treasure, glossy 3D render",
]
# Compare these checkpoints (training steps)
CKPTS = {
    "1000": f"{LORA_DIR}/flux-lora-poc_000001000.safetensors",
    "1500": f"{LORA_DIR}/flux-lora-poc.safetensors",
}
SEED = 42
STEPS = 25
GUIDANCE = 3.5
W = H = 768

from toolkit.config_modules import ModelConfig, GenerateImageConfig
from toolkit.stable_diffusion_model import StableDiffusion

print("=== Loading FLUX.2 base model (ai-toolkit loader, quantized) ===", flush=True)
model_config = ModelConfig(
    name_or_path="black-forest-labs/FLUX.2-dev",
    arch="flux2",
    quantize=True,
    quantize_te=True,
    low_vram=True,
    dtype="bf16",
)
sd = StableDiffusion(device="cuda:0", model_config=model_config, dtype="bf16")
sd.load_model()
print("=== Base model loaded ===", flush=True)

for tag, ckpt in CKPTS.items():
    if not os.path.exists(ckpt):
        print(f"!! checkpoint missing: {ckpt}", flush=True)
        continue
    print(f"=== Applying LoRA ckpt {tag}: {ckpt} ===", flush=True)
    # load LoRA weights onto the network
    from toolkit.lora_special import LoRASpecialNetwork
    try:
        sd.load_lora(ckpt)  # if available in this ai-toolkit version
    except Exception as e:
        print(f"   sd.load_lora not available ({e}); trying network attach", flush=True)
        raise

    gen_configs = []
    for i, p in enumerate(PROMPTS):
        out_path = f"{OUT_DIR}/pirate_{i}_ckpt{tag}.png"
        gen_configs.append(GenerateImageConfig(
            prompt=p, width=W, height=H, num_inference_steps=STEPS,
            guidance_scale=GUIDANCE, seed=SEED, output_path=out_path,
        ))
    print(f"=== Generating {len(gen_configs)} images for ckpt {tag} ===", flush=True)
    sd.generate_images(gen_configs)
    print(f"=== Done ckpt {tag} ===", flush=True)

    # unload this LoRA before next ckpt
    try:
        sd.unload_lora()
    except Exception:
        pass
    gc.collect(); torch.cuda.empty_cache()

print("=== ALL DONE ===", flush=True)
print(os.listdir(OUT_DIR), flush=True)
