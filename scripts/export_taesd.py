#!/usr/bin/env python3
"""
Export TAESD (Tiny AutoEncoder for Stable Diffusion) to 2 ONNX files.

TAESD is a distilled VAE for SD 1.5 — drop-in replacement for the
standard 80M-param VAE encoder/decoder with a ~1M-param encoder + 1M
decoder. ~3× faster decode at slight quality cost. Pairs with LCM
(Tier 1.8) for compounded SD inpaint speedup on CPU/iGPU.

    encoder.onnx   ~5 MB   image (1, 3, 512, 512) → latent (1, 4, 64, 64)
    decoder.onnx   ~5 MB   latent (1, 4, 64, 64)  → image  (1, 3, 512, 512)

Important behavioral difference vs standard VAE:
- TAESD does NOT apply the 0.18215 scaling factor in/out. Standard
  diffusers VAE does. The runtime path detects the TAESD backend and
  skips the scaling step in both encode and decode.

Source: https://github.com/madebyollin/taesd (MIT license).

Usage:
    pip install diffusers torch optimum onnx
    python scripts/export_taesd.py

Output goes into ./out/taesd-fp16/ . Upload the 2 files to a Prunr
GitHub release; SHA256 + size print to stdout for the registry entry.
"""

import os
import sys
import shutil
from pathlib import Path

OUT_DIR = Path("out/taesd-fp16")
TAESD_REPO = "madebyollin/taesd"


def main() -> int:
    require("torch")
    require("diffusers")
    require("optimum")
    require("onnx")

    import torch
    from diffusers import AutoencoderTiny

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Loading {TAESD_REPO} (FP16)…")
    # AutoencoderTiny is the diffusers wrapper for TAESD. Loading at
    # FP16 matches our SD bundle's UNet precision so latents flow
    # through without dtype casts at the ORT boundary.
    vae = AutoencoderTiny.from_pretrained(TAESD_REPO, torch_dtype=torch.float16)
    vae.eval()

    # Prepare static example inputs at the canonical SD 1.5 tile size.
    # Dynamic axes are NOT enabled — our pipeline always crops to
    # 512×512 before VAE so a fixed-shape export plays well with
    # OpenVINO's graph compiler.
    sample_image = torch.randn(1, 3, 512, 512, dtype=torch.float16)
    sample_latent = torch.randn(1, 4, 64, 64, dtype=torch.float16)

    enc_path = OUT_DIR / "encoder.onnx"
    print(f"[2/3] Exporting encoder → {enc_path}")
    # Use a thin wrapper so the exported graph has just `encode`'s
    # forward — diffusers' default __call__ chains encode+decode.
    class EncoderWrap(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.encoder = vae.encoder
        def forward(self, x):
            return self.encoder(x)

    torch.onnx.export(
        EncoderWrap(vae),
        sample_image,
        str(enc_path),
        input_names=["sample"],
        output_names=["latent_sample"],
        opset_version=17,
        do_constant_folding=True,
    )

    dec_path = OUT_DIR / "decoder.onnx"
    print(f"[3/3] Exporting decoder → {dec_path}")
    class DecoderWrap(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.decoder = vae.decoder
        def forward(self, latent):
            return self.decoder(latent)

    torch.onnx.export(
        DecoderWrap(vae),
        sample_latent,
        str(dec_path),
        input_names=["latent_sample"],
        output_names=["sample"],
        opset_version=17,
        do_constant_folding=True,
    )

    # SHA256 + size for the registry.
    print("\n=== SHA256 (paste into prunr-models registry) ===")
    import hashlib
    for part in ["encoder", "decoder"]:
        path = OUT_DIR / f"{part}.onnx"
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {part}.onnx  {size_mb:.1f} MB  {h}")

    print(f"\nDone. Bundle at {OUT_DIR.resolve()}")
    print("Upload these 2 .onnx files to your prunr GitHub release.")
    return 0


def require(module: str) -> None:
    try:
        __import__(module)
    except ImportError:
        print(
            f"error: missing Python package '{module}'.\n"
            f"  pip install diffusers torch optimum onnx",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
