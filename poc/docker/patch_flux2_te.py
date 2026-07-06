"""
Two ai-toolkit patches for FLUX.2 LoRA training on a single 46GB L40S.

PATCH 1 (flux2_model.py): quantize the Mistral text encoder on CPU BEFORE moving
it to GPU. Upstream loads full bf16 encoder (~48GB) onto GPU first then quantizes
— OOMs a 46GB card. We reorder: quantize on CPU (48GB -> 24GB fp8), then to GPU.

PATCH 2 (BaseSDTrainProcess.py): unload the text encoder to CPU BEFORE
prepare_accelerator moves the transformer to GPU. Upstream (commit 4e50535,
2026-06-28) only unloads the TE inside the sample step (deep in the train loop),
so at prepare time BOTH transformer (fp8 ~12GB) and Mistral (~24GB) sit on GPU =
~44GB, OOMing by ~200MB. Since we cache_text_embeddings + unload_text_encoder,
the TE is not needed on GPU during training — unload it before prepare.

Run at image build time. Idempotent: exits cleanly if already patched.
"""
import sys

FLUX2 = "/ai-toolkit/extensions_built_in/diffusion_models/flux2/flux2_model.py"
BASE = "/ai-toolkit/jobs/process/BaseSDTrainProcess.py"

# ---------------- PATCH 1: quantize TE on CPU before GPU ----------------
with open(FLUX2) as f:
    src1 = f.read()

p1_orig = '''        text_encoder.to(self.device_torch, dtype=dtype)

        flush()

        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing Mistral")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype))
            freeze(text_encoder)
            flush()'''

p1_new = '''        flush()

        if self.model_config.quantize_te:
            # PATCHED: quantize on CPU first to avoid OOM moving full bf16 encoder to GPU
            self.print_and_status_update("Quantizing Mistral on CPU")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype))
            freeze(text_encoder)
            flush()

        text_encoder.to(self.device_torch, dtype=dtype)
        flush()'''

if "Quantizing Mistral on CPU" in src1:
    print("PATCH 1: already applied, skipping")
elif p1_orig not in src1:
    print("PATCH 1 ERROR: target block not found in flux2_model.py — upstream changed")
    sys.exit(1)
else:
    with open(FLUX2, "w") as f:
        f.write(src1.replace(p1_orig, p1_new))
    print("PATCH 1: applied — Mistral quantized on CPU before GPU move")

# ---------------- PATCH 2: unload TE before prepare_accelerator ----------------
with open(BASE) as f:
    src2 = f.read()

# Insert a TE-unload right before the transformer is prepared onto the GPU.
# Anchor on the exact prepare line (present in commit 4e50535).
p2_anchor = '''        if self.sd.unet is not None:
            self.sd.unet = self.accelerator.prepare(self.sd.unet)'''

p2_new = '''        # PATCHED: unload text encoder to CPU before moving transformer to GPU.
        # With cache_text_embeddings the TE is not needed during training; leaving
        # it on GPU here (upstream only unloads it in the sample step) OOMs a 46GB card.
        if self.train_config.unload_text_encoder or self.is_caching_text_embeddings:
            try:
                self.sd.text_encoder_to('cpu')
            except Exception as _e:
                print(f"PATCH2 warn: text_encoder_to(cpu) failed: {_e}")
            from toolkit.basic import flush as _flush
            _flush()
        if self.sd.unet is not None:
            self.sd.unet = self.accelerator.prepare(self.sd.unet)'''

if "PATCHED: unload text encoder to CPU before moving transformer" in src2:
    print("PATCH 2: already applied, skipping")
elif p2_anchor not in src2:
    print("PATCH 2 ERROR: prepare_accelerator anchor not found in BaseSDTrainProcess.py — upstream changed")
    sys.exit(1)
else:
    with open(BASE, "w") as f:
        f.write(src2.replace(p2_anchor, p2_new))
    print("PATCH 2: applied — TE unloaded to CPU before prepare_accelerator")

print("PATCH: all patches processed")
