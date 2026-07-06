"""
FLUX.2 原生多参考图推理 —— ComfyUI API 驱动(与 comfy_gen.py 同栈)。

无需训练:给 1~N 张参考图 → 在新场景/新姿态下复用同一主体。
FLUX.2-dev 原生支持(官方一级特性),ComfyUI 用 ReferenceLatent 节点注入参考图 latent。

对应设计:docs/superpowers/specs/2026-06-30-multiref-inference-design.md

workflow:
  UNETLoader(flux2 fp8) + CLIPLoader(type=flux2) + VAELoader
  LoadImage(ref) → FluxKontextImageScale → VAEEncode → ReferenceLatent.latent ┐
  CLIPTextEncode(prompt) ──────────────────────────────────► ReferenceLatent.conditioning
  多张参考图 = 串接多个 ReferenceLatent
  ReferenceLatent → FluxGuidance → KSampler(+EmptySD3LatentImage) → VAEDecode → SaveImage

用法(推理机上,ComfyUI 已在 127.0.0.1:8188;参考图需先放到 /opt/ComfyUI/input/):
  python3 09_multiref_infer.py --refs ref_mermaid.png --out /exp/multiref \
      --prompts "the same mermaid character riding a seahorse through a coral city" \
                "the same mermaid character sitting on a treasure chest, front view"
"""
import argparse, json, os, time, urllib.request, urllib.error, urllib.parse, uuid

SERVER = "http://127.0.0.1:8188"
UNET = "flux2_dev_fp8mixed.safetensors"
CLIP = "mistral_3_small_flux2_fp8.safetensors"
VAE = "flux2-vae.safetensors"

SEED, STEPS, GUIDANCE, W, H = 42, 25, 3.5, 1024, 1024


def build_workflow(ref_names, prompt_text, seed, out_prefix):
    """FLUX.2 多参考图 workflow。ref_names: ComfyUI input/ 下的文件名列表。"""
    g = {}
    g["1"] = {"class_type": "UNETLoader",
              "inputs": {"unet_name": UNET, "weight_dtype": "fp8_e4m3fn"}}
    g["2"] = {"class_type": "CLIPLoader",
              "inputs": {"clip_name": CLIP, "type": "flux2"}}
    g["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": VAE}}

    # 正向文本 conditioning
    g["20"] = {"class_type": "CLIPTextEncode",
               "inputs": {"text": prompt_text, "clip": ["2", 0]}}
    g["21"] = {"class_type": "CLIPTextEncode",
               "inputs": {"text": "", "clip": ["2", 0]}}

    # 每张参考图:LoadImage → FluxKontextImageScale → VAEEncode → ReferenceLatent(串接)
    cond_out = ["20", 0]
    nid = 30
    for i, ref in enumerate(ref_names):
        load_id, scale_id, enc_id, refl_id = str(nid), str(nid + 1), str(nid + 2), str(nid + 3)
        g[load_id] = {"class_type": "LoadImage", "inputs": {"image": ref}}
        g[scale_id] = {"class_type": "FluxKontextImageScale",
                       "inputs": {"image": [load_id, 0]}}
        g[enc_id] = {"class_type": "VAEEncode",
                     "inputs": {"pixels": [scale_id, 0], "vae": ["3", 0]}}
        g[refl_id] = {"class_type": "ReferenceLatent",
                      "inputs": {"conditioning": cond_out, "latent": [enc_id, 0]}}
        cond_out = [refl_id, 0]   # 串接:下一张参考图接在这条 conditioning 上
        nid += 4

    g["80"] = {"class_type": "FluxGuidance",
               "inputs": {"guidance": GUIDANCE, "conditioning": cond_out}}
    g["81"] = {"class_type": "EmptySD3LatentImage",
               "inputs": {"width": W, "height": H, "batch_size": 1}}
    g["90"] = {"class_type": "KSampler",
               "inputs": {"model": ["1", 0], "positive": ["80", 0],
                          "negative": ["21", 0], "latent_image": ["81", 0],
                          "seed": seed, "steps": STEPS, "cfg": 1.0,
                          "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}}
    g["95"] = {"class_type": "VAEDecode",
               "inputs": {"samples": ["90", 0], "vae": ["3", 0]}}
    g["99"] = {"class_type": "SaveImage",
               "inputs": {"filename_prefix": out_prefix, "images": ["95", 0]}}
    return g


def submit(graph, client_id):
    data = json.dumps({"prompt": graph, "client_id": client_id}).encode()
    req = urllib.request.Request(f"{SERVER}/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req))["prompt_id"]
    except urllib.error.HTTPError as e:
        print("SUBMIT ERROR:", e.read().decode()[:2000])
        raise


def wait(prompt_id, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            h = json.load(urllib.request.urlopen(f"{SERVER}/history/{prompt_id}"))
        except Exception:
            time.sleep(2); continue
        if prompt_id in h:
            entry = h[prompt_id]
            st = entry.get("status", {})
            if st.get("completed") or st.get("status_str") == "success":
                return entry
            if st.get("status_str") == "error":
                print("EXEC ERROR:", json.dumps(st, indent=2)[:3000])
                return entry
        time.sleep(3)
    raise TimeoutError(f"prompt {prompt_id} timed out")


def collect(entry, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for _, out in entry.get("outputs", {}).items():
        for img in out.get("images", []):
            url = (f"{SERVER}/view?filename={urllib.parse.quote(img['filename'])}"
                   f"&subfolder={urllib.parse.quote(img.get('subfolder',''))}"
                   f"&type={img.get('type','output')}")
            dest = os.path.join(out_dir, img["filename"])
            with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
                f.write(r.read())
            saved.append(dest)
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", nargs="+", required=True,
                    help="ComfyUI input/ 下的参考图文件名(1~10 张)")
    ap.add_argument("--prompts", nargs="+", required=True, help="新场景 prompt(每个出一张)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    client_id = str(uuid.uuid4())
    all_saved = []
    for i, prompt in enumerate(args.prompts):
        prefix = f"multiref_{i:02d}"
        graph = build_workflow(args.refs, prompt, SEED, prefix)
        print(f"[{i}] refs={args.refs} :: {prompt[:70]}", flush=True)
        pid = submit(graph, client_id)
        print(f"    submitted {pid}, waiting...", flush=True)
        entry = wait(pid)
        saved = collect(entry, args.out)
        print(f"    saved: {saved}", flush=True)
        all_saved += saved

    print(f"\n=== {len(all_saved)} images in {args.out} ===", flush=True)
    print("核对:同一主体身份是否跨场景保持;达标后将 README/spec 从 🟡 转 ✅。", flush=True)


if __name__ == "__main__":
    main()
