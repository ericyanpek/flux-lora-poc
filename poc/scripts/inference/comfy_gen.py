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

用法:
  comfy_gen.py --config base   --out /exp/base
  comfy_gen.py --config style  --out /exp/style
  comfy_gen.py --config char   --out /exp/char
  comfy_gen.py --config combo  --out /exp/combo
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


def build_workflow(prompt_text, use_style, use_char, ws, wc, seed, out_prefix):
    """返回 ComfyUI API 格式的 prompt graph(dict of nodes)。"""
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
    if use_style and ws > 0:
        g[str(nid)] = {"class_type": "LoraLoader",
                       "inputs": {"lora_name": LORA_STYLE,
                                  "strength_model": ws, "strength_clip": ws,
                                  "model": model_out, "clip": clip_out}}
        model_out = [str(nid), 0]
        clip_out = [str(nid), 1]
        nid += 1
    if use_char and wc > 0:
        g[str(nid)] = {"class_type": "LoraLoader",
                       "inputs": {"lora_name": LORA_CHAR,
                                  "strength_model": wc, "strength_clip": wc,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--themes", default="pirate,dragon,mermaid")
    args = ap.parse_args()

    use_style, use_char, ws, wc = CONFIGS[args.config]
    client_id = str(uuid.uuid4())
    themes = args.themes.split(",")

    all_saved = []
    for theme in themes:
        body = THEMES[theme]
        prompt_text = build_prompt(body, use_style, use_char)
        prefix = f"{theme}__{args.config}"
        graph = build_workflow(prompt_text, use_style, use_char, ws, wc, SEED, prefix)
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


if __name__ == "__main__":
    main()
