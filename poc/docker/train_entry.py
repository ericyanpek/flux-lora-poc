"""
Bridge script: reads env vars, builds ai-toolkit YAML config, runs training,
saves output to /opt/ml/model/.
On EC2: env vars injected via docker run -e. On SageMaker: reads hyperparameters.json.
"""
import json
import os
import sys
import yaml

sys.path.insert(0, "/ai-toolkit")

HYPERPARAM_PATH = "/opt/ml/input/config/hyperparameters.json"
TRAINING_DATA_PATH = "/opt/ml/input/data/training"
OUTPUT_PATH = "/opt/ml/model"
CHECKPOINT_PATH = "/opt/ml/checkpoints"


def load_hyperparameters():
    # EC2 path: env vars take precedence; SageMaker path: read JSON file
    hp = {}
    if os.path.exists(HYPERPARAM_PATH):
        with open(HYPERPARAM_PATH) as f:
            raw = json.load(f)
        hp = {k: v.strip('"') if isinstance(v, str) else v for k, v in raw.items()}
    # env vars override file (EC2 mode)
    for key in ["trigger_word", "model_name", "steps", "lr", "rank", "sample_every", "layer", "project_name", "resolution"]:
        env_val = os.environ.get(key.upper())
        if env_val:
            hp[key] = env_val
    return hp


def build_config(hp: dict) -> dict:
    trigger_word = hp.get("trigger_word", "GAMECATV1")
    steps = int(hp.get("steps", "1500"))
    lr = float(hp.get("lr", "1e-4"))
    rank = int(hp.get("rank", "32"))
    sample_every = int(hp.get("sample_every", "250"))
    model_name = hp.get("model_name", "black-forest-labs/FLUX.2-dev")
    wandb_key = os.environ.get("WANDB_API_KEY", "")

    # resolution: 逗号分隔支持多分辨率 bucket(ai-toolkit 会按 list 各训一遍)。
    # 默认 640(保持 slot 训练不变);icon 传 "640,1024" 做动态分辨率。
    res_raw = str(hp.get("resolution", "640"))
    resolution = [int(x) for x in res_raw.split(",") if x.strip()]
    # sample 尺寸用最大分辨率(1024 图标细节更清晰)
    sample_dim = max(resolution)

    layer = hp.get("layer", "")
    if layer == "icon":
        # 中国风 Q版图标风格 LoRA:验证画风是否泛化到训练集外的新道具主体
        sample_prompts = [
            f"{trigger_word}, a sword, glossy material, isolated on white background",
            f"{trigger_word}, a treasure map scroll, glossy material, isolated on white background",
            f"{trigger_word}, a jade ring, dominant green with gold accents, isolated on white background",
            "a sword icon, isolated on white background",
        ]
    elif layer == "style":
        sample_prompts = [
            f"{trigger_word}, a treasure chest full of gold coins",
            f"{trigger_word}, a fierce dragon mascot",
            f"{trigger_word}, a magic potion bottle icon",
            "a treasure chest full of gold coins",
        ]
    elif layer == "char":
        sample_prompts = [
            f"{trigger_word}, on a beach with palm trees",
            f"{trigger_word}, in a fantasy forest",
            f"{trigger_word}, portrait close up",
            "a character on a beach",
        ]
    else:
        sample_prompts = [
            f"a {trigger_word} character sitting on a beach, casual game style, vibrant colors",
            f"a {trigger_word} character in a fantasy forest, detailed character art",
            f"a {trigger_word} character portrait, close up, high quality",
            "a character sitting on a beach, casual game style, vibrant colors",
        ]

    process = {
        "type": "sd_trainer",
        "training_folder": TRAINING_DATA_PATH,
        # NOTE: ai-toolkit ignores output_folder; actual artifacts go to
        # training_folder/<config name>/ (= /opt/ml/input/data/training/flux-lora-poc/).
        # ctl.py syncs from that path, not /opt/ml/model.
        "output_folder": OUTPUT_PATH,
        "device": "cuda:0",
        "model": {
            "arch": "flux2",            # use ai-toolkit's dedicated Flux2Model (meta-device load + assign weights)
            "name_or_path": model_name,
            "quantize": True,           # fp8 quantize transformer
            "quantize_te": True,        # fp8 quantize Mistral-Small-3.1-24B text encoder
            "low_vram": True,           # keep transformer on CPU during quantization (L40S 48GB headroom)
        },
        "train": {
            "batch_size": 1,
            "steps": steps,
            "gradient_accumulation_steps": 1,   # 4->1: reduce prepare-stage VRAM (was OOM by 324MB on L40S 46GB)
            "train_unet": True,
            "train_text_encoder": False,
            "unload_text_encoder": True,   # precompute embeddings then move Mistral to CPU — frees ~24GB VRAM during training
            "skip_first_sample": True,     # skip baseline sample before training (holds VRAM, causes OOM at prepare)
            "lr": lr,
            "optimizer": "adamw8bit",
            "lr_scheduler": "cosine",
            "gradient_checkpointing": True,
            "noise_scheduler": "flowmatch",
            "dtype": "bf16",
        },
        "network": {
            "type": "lora",
            "linear": rank,
            "linear_alpha": rank,
        },
        "save": {
            "save_every": sample_every,
            "save_format": "safetensors",
            "max_step_saves_to_keep": 4,
        },
        "sample": {
            "sample_every": sample_every,
            "width": sample_dim,
            "height": sample_dim,
            "prompts": sample_prompts,
            "neg": "",
            "seed": 42,
            "guidance_scale": 3.5,
            "sample_steps": 20,
            "walk_seed": False,
        },
        "datasets": [{
            "folder_path": TRAINING_DATA_PATH,
            "caption_ext": "txt",
            "resolution": resolution,         # list → ai-toolkit 多分辨率 bucket(如 [640,1024]);默认 [640]
            "default_caption": f"{trigger_word}",
            "cache_latents_to_disk": True,    # precompute VAE latents
            "cache_text_embeddings": True,    # precompute Mistral embeddings, then unload encoder
            # flip_aug removed: augmentations are incompatible with latent caching
        }],
    }

    if wandb_key:
        process["logging"] = {
            "use_wandb": True,
            "project": hp.get("project_name", "flux2-lora-poc"),
            "run_name": f"{layer or 'base'}-{trigger_word}",
        }

    return {
        "job": "extension",
        "config": {
            "name": "flux-lora-poc",
            "process": [process],
        },
    }


def main():
    hp = load_hyperparameters()
    safe_hp = {k: "***" if k == "hf_token" else v for k, v in hp.items()}
    print(f"Hyperparameters: {safe_hp}")

    hf_token = hp.get("hf_token", os.environ.get("HF_TOKEN", ""))
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        print("HF token set")
    else:
        print("WARNING: No HF_TOKEN — FLUX.2-dev download will fail if license-gated")

    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if wandb_key:
        print("W&B logging enabled")
    else:
        print("W&B key not set — logging disabled")

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    config = build_config(hp)
    config_path = "/tmp/train_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print("Generated ai-toolkit config:")
    with open(config_path) as f:
        print(f.read())

    from toolkit.job import get_job
    job = get_job(config_path)
    job.run()
    job.cleanup()
    print("Training complete")


if __name__ == "__main__":
    main()
