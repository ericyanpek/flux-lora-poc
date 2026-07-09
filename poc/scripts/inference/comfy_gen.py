"""
ComfyUI API 出图 —— FLUX.2-dev fp8 底模 + 可选分层 LoRA(style/char)。

在推理实例上跑(ComfyUI 已在 127.0.0.1:8188)。构造 FLUX.2 workflow 图,
POST /prompt 提交,轮询 /history 完成,把输出 PNG 收集到 --out 并可选上传 S3。

FLUX.2 workflow 关键节点(已由 comfy_probe.py 确认):
  UNETLoader(flux2_dev_fp8mixed, fp8_e4m3fn) -> MODEL
  CLIPLoader(mistral_3_small_flux2_fp8, type=flux2) -> CLIP
  VAELoader(flux2-vae) -> VAE
  [可选] LoraLoader 串接 MODEL+CLIP
  CLIPTextEncode(正) / CLIPTextEncode(负 空) -> COND
  FluxGuidance(正cond, guidance) -> COND
  EmptySD3LatentImage(w,h) -> LATENT
  KSampler(model, pos, neg, latent, seed, steps, cfg=1.0, sampler, scheduler) -> LATENT
  VAEDecode -> IMAGE -> SaveImage(prefix)

用法(两种模式二选一):
  # 预设模式:CONFIGS × THEMES 批量出图,固定 seed,做可复现对照矩阵
  comfy_gen.py --config base   --out /exp/base
  comfy_gen.py --config style  --out /exp/style
  comfy_gen.py --config char   --out /exp/char
  comfy_gen.py --config combo  --out /exp/combo

  # 自定义模式:任意 prompt(原样使用,触发词自己写)+ 任意 LoRA 组合,出单图
  comfy_gen.py --prompt "slotstyle, a cyberpunk fox, gold coins" \
               --lora slotstyle.safetensors:1.0 --out /exp/custom
  comfy_gen.py --prompt "a plain castle" --out /exp/base-custom   # 不带 --lora = 纯底模
"""
import argparse, json, os, time, urllib.request, urllib.error, urllib.parse, uuid

SERVER = "http://127.0.0.1:8188"

UNET = "flux2_dev_fp8mixed.safetensors"
CLIP = "mistral_3_small_flux2_fp8.safetensors"
VAE = "flux2-vae.safetensors"
LORA_STYLE = "slotstyle.safetensors"
LORA_CHAR = "slotchar.safetensors"

STYLE_TRIG = "slotstyle"
CHAR_TRIG = "slotchar"
TAIL = "glossy 3D cartoon slot game art, vibrant colors, gold coins, high quality"

# 三个主题:2 通用新角色(海盗狼、龙) + 1 自定义 IP(美人鱼)
THEMES = {
    "pirate":  "a cartoon wolf pirate captain with an eyepatch and hat, treasure chest, ship deck background",
    "dragon":  "a cute chubby dragon guarding a pile of gold coins, glowing gems, cave background",
    "mermaid": "a cute mermaid character with shimmering teal fish tail, pearl crown, holding a golden trident, coral reef, pearls",
}

# config -> (用style_trig, 用char_trig, style权重, char权重)
CONFIGS = {
    "base":  (False, False, 0.0, 0.0),
    "style": (True,  False, 1.0, 0.0),
    "char":  (False, True,  0.0, 1.0),
    "combo": (True,  True,  0.9, 0.8),
}

SEED, STEPS, GUIDANCE, W, H = 42, 25, 3.5, 1024, 1024


def build_prompt(theme_body, use_style, use_char):
    trigs = []
    if use_style:
        trigs.append(STYLE_TRIG)
    if use_char:
        trigs.append(CHAR_TRIG)
    prefix = (", ".join(trigs) + ", ") if trigs else ""
    return f"{prefix}{theme_body}, {TAIL}"


def build_workflow(prompt_text, loras, seed, out_prefix):
    """返回 ComfyUI API 格式的 prompt graph(dict of nodes)。
    loras: [(lora_name, weight), ...] —— 按顺序串接,weight<=0 的跳过。
    空列表 = 纯底模(base)。"""
    g = {}
    g["1"] = {"class_type": "UNETLoader",
              "inputs": {"unet_name": UNET, "weight_dtype": "fp8_e4m3fn"}}
    g["2"] = {"class_type": "CLIPLoader",
              "inputs": {"clip_name": CLIP, "type": "flux2"}}
    g["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": VAE}}

    model_out = ["1", 0]
    clip_out = ["2", 0]

    # 串接 LoRA(每个 LoraLoader 同时改 MODEL + CLIP)
    nid = 10
    for lora_name, w in loras:
        if w <= 0:
            continue
        g[str(nid)] = {"class_type": "LoraLoader",
                       "inputs": {"lora_name": lora_name,
                                  "strength_model": w, "strength_clip": w,
                                  "model": model_out, "clip": clip_out}}
        model_out = [str(nid), 0]
        clip_out = [str(nid), 1]
        nid += 1

    g["20"] = {"class_type": "CLIPTextEncode",
               "inputs": {"text": prompt_text, "clip": clip_out}}
    g["21"] = {"class_type": "CLIPTextEncode",
               "inputs": {"text": "", "clip": clip_out}}
    g["22"] = {"class_type": "FluxGuidance",
               "inputs": {"guidance": GUIDANCE, "conditioning": ["20", 0]}}
    g["30"] = {"class_type": "EmptySD3LatentImage",
               "inputs": {"width": W, "height": H, "batch_size": 1}}
    g["40"] = {"class_type": "KSampler",
               "inputs": {"model": model_out, "positive": ["22", 0],
                          "negative": ["21", 0], "latent_image": ["30", 0],
                          "seed": seed, "steps": STEPS, "cfg": 1.0,
                          "sampler_name": "euler", "scheduler": "simple",
                          "denoise": 1.0}}
    g["50"] = {"class_type": "VAEDecode",
               "inputs": {"samples": ["40", 0], "vae": ["3", 0]}}
    g["60"] = {"class_type": "SaveImage",
               "inputs": {"filename_prefix": out_prefix, "images": ["50", 0]}}
    return g


def submit(graph, client_id):
    data = json.dumps({"prompt": graph, "client_id": client_id}).encode()
    req = urllib.request.Request(f"{SERVER}/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        resp = json.load(urllib.request.urlopen(req))
    except urllib.error.HTTPError as e:
        print("SUBMIT ERROR:", e.read().decode()[:2000])
        raise
    return resp["prompt_id"]


def wait(prompt_id, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            h = json.load(urllib.request.urlopen(f"{SERVER}/history/{prompt_id}"))
        except Exception:
            time.sleep(2)
            continue
        if prompt_id in h:
            entry = h[prompt_id]
            status = entry.get("status", {})
            if status.get("completed") or status.get("status_str") == "success":
                return entry
            if status.get("status_str") == "error":
                print("EXEC ERROR:", json.dumps(status, indent=2)[:3000])
                return entry
        time.sleep(3)
    raise TimeoutError(f"prompt {prompt_id} timed out")


def collect(entry, out_dir):
    """从 history entry 的 outputs 拷贝 PNG 到 out_dir。"""
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for node_id, out in entry.get("outputs", {}).items():
        for img in out.get("images", []):
            fn = img["filename"]
            sub = img.get("subfolder", "")
            typ = img.get("type", "output")
            url = f"{SERVER}/view?filename={urllib.parse.quote(fn)}&subfolder={urllib.parse.quote(sub)}&type={typ}"
            dest = os.path.join(out_dir, fn)
            with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
                f.write(r.read())
            saved.append(dest)
    return saved


def config_to_loras(config):
    """把预设 config 翻译成 build_workflow 需要的 [(lora_name, weight), ...]。"""
    use_style, use_char, ws, wc = CONFIGS[config]
    loras = []
    if use_style:
        loras.append((LORA_STYLE, ws))
    if use_char:
        loras.append((LORA_CHAR, wc))
    return loras


def run_preset(args, client_id):
    """预设模式:CONFIGS × THEMES 批量出图(可复现对照矩阵)。"""
    use_style, use_char, _, _ = CONFIGS[args.config]
    loras = config_to_loras(args.config)
    themes = args.themes.split(",")
    all_saved = []
    for theme in themes:
        body = THEMES[theme]
        prompt_text = build_prompt(body, use_style, use_char)
        prefix = f"{theme}__{args.config}"
        graph = build_workflow(prompt_text, loras, args.seed, prefix)
        print(f"[{args.config}/{theme}] prompt: {prompt_text}", flush=True)
        pid = submit(graph, client_id)
        print(f"  submitted {pid}, waiting...", flush=True)
        entry = wait(pid)
        saved = collect(entry, args.out)
        print(f"  saved: {saved}", flush=True)
        all_saved += saved
    print(f"\n=== {args.config}: {len(all_saved)} images in {args.out} ===", flush=True)
    for s in all_saved:
        print(" ", s)


def run_custom(args, client_id):
    """自定义模式:用户任意 prompt + 任意 LoRA 组合出一张图。
    prompt 原样使用(不追加 THEMES/TAIL);触发词由用户自行写进 prompt。"""
    # --lora name:weight 可多次;省略则纯底模(base)
    loras = []
    for spec in (args.lora or []):
        if ":" in spec:
            name, w = spec.rsplit(":", 1)
            loras.append((name, float(w)))
        else:
            loras.append((spec, 1.0))
    graph = build_workflow(args.prompt, loras, args.seed, "custom")
    lora_desc = ", ".join(f"{n}@{w}" for n, w in loras) or "(base, no LoRA)"
    print(f"[custom] loras: {lora_desc}", flush=True)
    print(f"[custom] prompt: {args.prompt}", flush=True)
    pid = submit(graph, client_id)
    print(f"  submitted {pid}, waiting...", flush=True)
    entry = wait(pid)
    saved = collect(entry, args.out)
    print(f"\n=== custom: {len(saved)} images in {args.out} ===", flush=True)
    for s in saved:
        print(" ", s)


def main():
    ap = argparse.ArgumentParser(
        description="ComfyUI 出图。两种模式:预设(--config,批量对照矩阵)或自定义(--prompt,单图任意提示词)。")
    ap.add_argument("--config", choices=list(CONFIGS),
                    help="预设模式:base/style/char/combo,对 --themes 批量出图")
    ap.add_argument("--prompt", help="自定义模式:任意提示词(原样使用,触发词自己写进去),出单图")
    ap.add_argument("--lora", action="append",
                    help="自定义模式的 LoRA,格式 文件名[:权重](默认权重1.0),可重复。"
                         "例:--lora slotstyle.safetensors:0.9 --lora slotchar.safetensors:0.8")
    ap.add_argument("--out", required=True)
    ap.add_argument("--themes", default="pirate,dragon,mermaid", help="预设模式的主题(逗号分隔)")
    ap.add_argument("--seed", type=int, default=SEED, help=f"随机种子(默认 {SEED})")
    args = ap.parse_args()

    if bool(args.config) == bool(args.prompt):
        ap.error("二选一:--config(预设批量)或 --prompt(自定义单图)")

    client_id = str(uuid.uuid4())
    if args.prompt:
        run_custom(args, client_id)
    else:
        run_preset(args, client_id)


if __name__ == "__main__":
    main()
