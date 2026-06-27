"""
Patch ai-toolkit's flux2_model.py to quantize the Mistral text encoder on CPU
BEFORE moving it to GPU. Upstream loads the full bf16 encoder (~48GB) onto the
GPU first, then quantizes — which OOMs a 46GB L40S. We reorder: quantize on CPU
(48GB -> 24GB fp8), then move to GPU.

Run at image build time. Idempotent: exits cleanly if already patched.
"""
import re
import sys

TARGET = "/ai-toolkit/extensions_built_in/diffusion_models/flux2/flux2_model.py"

with open(TARGET) as f:
    src = f.read()

# Original block: unconditional .to(GPU) then conditional quantize
original = '''        text_encoder.to(self.device_torch, dtype=dtype)

        flush()

        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing Mistral")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype))
            freeze(text_encoder)
            flush()'''

# Patched block: quantize on CPU first (if enabled), then move to GPU
patched = '''        flush()

        if self.model_config.quantize_te:
            # PATCHED: quantize on CPU first to avoid OOM moving full bf16 encoder to GPU
            self.print_and_status_update("Quantizing Mistral on CPU")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype))
            freeze(text_encoder)
            flush()

        text_encoder.to(self.device_torch, dtype=dtype)
        flush()'''

if patched.split("\n")[2].strip() in src and "Quantizing Mistral on CPU" in src:
    print("PATCH: already applied, skipping")
    sys.exit(0)

if original not in src:
    print("PATCH ERROR: target block not found — flux2_model.py may have changed upstream")
    print("Looking for .to(device_torch) before quantize_te block")
    sys.exit(1)

src = src.replace(original, patched)

with open(TARGET, "w") as f:
    f.write(src)

print("PATCH: applied successfully — Mistral now quantized on CPU before GPU move")
