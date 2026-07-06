"""
FLUX.2 原生多参考图推理 —— 骨架(🟡 未 GPU 验证)。

第二条互补主线:无需训练,给 1~N 张角色定妆图 → 生成"同角色 × 新场景"。
对应设计:docs/superpowers/specs/2026-06-30-multiref-inference-design.md

⚠️ 状态:此脚本为骨架,尚未在 GPU 上跑通。API 已按 diffusers main 文档核实
(2026-07-06):参考图入参是 `image=`(单张 PIL 或 list[PIL]),不是 reference_images=。
但 dev 版 Flux2Pipeline 的 `image=[多张]` 多参考融合语义仍需实测确认(文档描述偏
img2img starting point)。Klein-KV pipeline 是文档明确标注"reference image
conditioning"的正主。跑通并产出对照样图前,勿在 README 标 ✅。

两条路径(用 --pipeline 选):
  dev     : Flux2Pipeline(Mistral 编码器,FLUX.2-dev)—— image=[refs]
  klein-kv: Flux2KleinKVPipeline(Qwen3 编码器,FLUX.2-klein-9b-kv)—— 文档背书最强,
            KV-cache 缓存参考图 token,默认 4 步,显存友好,可能是 Demo 现场最优

用法(GPU 机器上,diffusers 环境):
  python3 09_multiref_infer.py --pipeline klein-kv \
      --refs char_ref.png --out /exp/multiref \
      --prompts "the same character riding a dragon over a castle" \
                "the same character as a pirate captain on a ship deck"
"""
import argparse
import os
import sys


def load_refs(paths):
    from PIL import Image
    refs = []
    for p in paths:
        if not os.path.exists(p):
            sys.exit(f"reference image not found: {p}")
        refs.append(Image.open(p).convert("RGB"))
    return refs


def build_pipeline(kind, dtype):
    import torch
    if kind == "dev":
        # ⚠️ FLUX.2-dev 全量 bf16 在 46GB 上此前 segfault/OOM;需 fp8/量化或更大卡。
        from diffusers import Flux2Pipeline
        pipe = Flux2Pipeline.from_pretrained(
            "black-forest-labs/FLUX.2-dev", torch_dtype=dtype)
    elif kind == "klein-kv":
        # 文档明确的 reference-image-conditioning pipeline(KV-cache,4 步,显存友好)。
        # 需 gated 权重 FLUX.2-klein-9b-kv(Qwen3 编码器,非 Mistral)。
        from diffusers import Flux2KleinKVPipeline
        pipe = Flux2KleinKVPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-9b-kv", torch_dtype=dtype)
    else:
        sys.exit(f"unknown pipeline kind: {kind}")
    pipe.enable_model_cpu_offload()  # 省显存;46GB 上必要
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", choices=["dev", "klein-kv"], default="klein-kv")
    ap.add_argument("--refs", nargs="+", required=True, help="角色定妆参考图(1~10 张)")
    ap.add_argument("--prompts", nargs="+", required=True, help="新场景 prompt(每个出一张)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=None, help="dev 默认50;klein-kv 默认4")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    os.makedirs(args.out, exist_ok=True)
    refs = load_refs(args.refs)
    # klein-kv 官方示例传单张;多张走 list。dev 多参考图语义待实测。
    image_arg = refs if len(refs) > 1 else refs[0]

    print(f"=== building {args.pipeline} pipeline (refs={len(refs)}) ===", flush=True)
    pipe = build_pipeline(args.pipeline, torch.bfloat16)

    steps = args.steps if args.steps is not None else (4 if args.pipeline == "klein-kv" else 50)
    saved = []
    for i, prompt in enumerate(args.prompts):
        gen = torch.Generator("cpu").manual_seed(args.seed)
        # ⚠️ 未验证:确认 image= 对所选 pipeline 真做多参考融合而非仅 img2img。
        out = pipe(prompt, image=image_arg, num_inference_steps=steps, generator=gen)
        img = out.images[0]
        dest = os.path.join(args.out, f"multiref_{i:02d}.png")
        img.save(dest)
        saved.append(dest)
        print(f"  [{i}] {prompt[:60]}... -> {dest}", flush=True)

    print(f"\n=== {len(saved)} images in {args.out} ===", flush=True)
    print("提醒:核对同一角色身份是否跨场景保持;达标后才把 README 多参考图标 ✅。", flush=True)


if __name__ == "__main__":
    main()
