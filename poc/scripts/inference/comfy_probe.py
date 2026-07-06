"""Probe ComfyUI object_info for the FLUX.2 node names we need to build a workflow."""
import json, urllib.request

d = json.load(urllib.request.urlopen("http://127.0.0.1:8188/object_info"))
ks = list(d.keys())
print("=== FLUX/Flux nodes ===")
for k in ks:
    if "lux" in k.lower():
        print(" ", k)
print("=== loaders present ===")
for n in ["UNETLoader", "CLIPLoader", "VAELoader", "LoraLoader", "LoraLoaderModelOnly",
          "DualCLIPLoader", "EmptyLatentImage", "EmptySD3LatentImage",
          "KSampler", "VAEDecode", "SaveImage", "CLIPTextEncode",
          "ModelSamplingFlux", "FluxGuidance", "BasicGuider", "SamplerCustomAdvanced"]:
    print(f"  {n}: {n in d}")

# UNETLoader + CLIPLoader input options (weight_dtype, clip type list)
for n in ["UNETLoader", "CLIPLoader"]:
    if n in d:
        print(f"=== {n} required inputs ===")
        print(json.dumps(d[n]["input"].get("required", {}), indent=1)[:900])
