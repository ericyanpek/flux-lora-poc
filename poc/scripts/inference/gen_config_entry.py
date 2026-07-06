"""
⚠️ KNOWN-FAILED(保留作探索记录):此路径在 46GB L40S 上 Mistral .to() OOM,不可用。
   可靠的推理路径是 inference/comfy_gen.py(独立 ComfyUI + 官方 fp8 底模)。

用 ai-toolkit 训练入口的 SAME 路径出图(sd_trainer + steps=0 + skip_first_sample=False)。
走 Flux2Model 加载(和训练完全相同),规避通用 StableDiffusion / diffusers
Flux2Pipeline 在 46GB 上的段错误。

通过 env 传入:
  LORA_PATH   要加载的单个 LoRA safetensors(容器内路径)
  TRIGGER     触发词(slotstyle / slotchar)
  TAG         输出子目录名(style / char)
Run INSIDE container:
  LORA_PATH=/loras/style.safetensors TRIGGER=slotstyle TAG=style python3 gen_config_entry.py
输出: /opt/ml/model/<TAG>/samples/  (由 ai-toolkit 写到 training_folder/<name>/samples)
"""
import os, sys, yaml
sys.path.insert(0, "/ai-toolkit")

LORA_PATH = os.environ["LORA_PATH"]
TRIGGER = os.environ.get("TRIGGER", "slotstyle")
TAG = os.environ.get("TAG", "style")
OUT = f"/opt/ml/model/{TAG}"
os.makedirs(OUT, exist_ok=True)

TAIL = "glossy 3D cartoon slot game art, vibrant colors, gold coins, high quality"
prompts = [
    f"{TRIGGER}, a cartoon wolf pirate captain with eyepatch and hat, treasure chest, {TAIL}",
    f"{TRIGGER}, a cute chubby dragon guarding a pile of gold coins, glowing gems, {TAIL}",
    f"{TRIGGER}, a cute mermaid with shimmering teal fish tail, pearl crown, golden trident, coral reef, pearls, {TAIL}",
]

# 复用训练验证过的 model 块;network 走 sd_trainer 的 LoRASpecialNetwork(与训练同机制)
# steps=0 + skip_first_sample=False => 训练循环启动即出一次 sample 然后结束
config = {
    "job": "extension",
    "config": {
        "name": TAG,
        "process": [{
            "type": "sd_trainer",
            "training_folder": OUT,
            "device": "cuda:0",
            "network": {
                "type": "lora",
                "linear": 32,
                "linear_alpha": 32,
                # 加载已训练的 LoRA 权重(ai-toolkit: NetworkConfig.pretrained_lora_path)
                "pretrained_lora_path": LORA_PATH,
            },
            "model": {
                "arch": "flux2",
                "name_or_path": "black-forest-labs/FLUX.2-dev",
                "quantize": True,
                "quantize_te": True,
                "low_vram": True,
            },
            "train": {
                "batch_size": 1,
                "steps": 0,
                "gradient_accumulation_steps": 1,
                "train_unet": True,
                "train_text_encoder": False,
                "unload_text_encoder": True,
                "skip_first_sample": False,   # 关键:启动即出 sample
                "lr": 1e-4,
                "optimizer": "adamw8bit",
                "dtype": "bf16",
                "disable_sampling": False,
            },
            "save": {"save_every": 100000, "save_format": "safetensors", "max_step_saves_to_keep": 1},
            "sample": {
                "sample_every": 100000,   # 不靠周期,靠 first sample
                "width": 768, "height": 768,
                "prompts": prompts,
                "neg": "", "seed": 42, "guidance_scale": 3.5,
                "sample_steps": 25, "walk_seed": False,
            },
            "datasets": [],
        }],
    },
}

cfg_path = "/tmp/gen_config.yaml"
with open(cfg_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False)
print("=== generated gen config ===", flush=True)
print(open(cfg_path).read(), flush=True)

hf = os.environ.get("HF_TOKEN", "")
if hf:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf

from toolkit.job import get_job
job = get_job(cfg_path)
job.run()
job.cleanup()
print(f"=== DONE, images in {OUT} ===", flush=True)
import glob
print(glob.glob(f"{OUT}/**/*.jpg", recursive=True) + glob.glob(f"{OUT}/**/*.png", recursive=True), flush=True)
