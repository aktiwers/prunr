#!/usr/bin/env python3
"""
Export an LCM-distilled SD 1.5 Inpaint pipeline to 4 ONNX files.

Output layout matches the existing SD 1.5 Inpaint bundle so prunr's
dispatcher can swap backends with no per-file routing changes:

    text_encoder.onnx
    vae_encoder.onnx
    vae_decoder.onnx
    unet.onnx

The LCM-LoRA (https://huggingface.co/latent-consistency/lcm-lora-sdv1-5)
is a small adapter that modifies SD 1.5 to converge in ~4 timesteps
instead of 20-30. We merge the LoRA weights into the base SD 1.5
Inpaint UNet at fp16 precision (~860M params) and export the result.

Run on a machine with:
- Python 3.10+
- diffusers, transformers, torch, peft
- ~8 GB free RAM during export
- ~5 GB free disk for the FP16 ONNX bundle

Output goes into  ./out/sd-15-lcm-inpaint-fp16/ . Upload that
directory's contents to your prunr GitHub release as the SdV15LcmInpaintFp16
multi-part bundle.

Usage:
    pip install diffusers transformers torch peft accelerate optimum onnx onnxruntime
    python scripts/export_lcm_inpaint.py

Notes:
- The base inpaint checkpoint is `runwayml/stable-diffusion-inpainting`.
  If that's gated, swap in `botp/stable-diffusion-v1-5-inpainting` (mirror).
- The LCM-LoRA is `latent-consistency/lcm-lora-sdv1-5`.
- Both are FP16. Mixed-precision is intentional: text_encoder and VAE
  stay FP32 to avoid color-shift artifacts at the [-1, 1] boundary;
  UNet (the bottleneck) is FP16.
- `optimum-cli export onnx` would handle this in one shot if HuggingFace
  ever ships an LCM-aware exporter; until then we drive it manually so
  the LoRA-merge step happens before the conversion.
"""

import os
import sys
import shutil
from pathlib import Path

OUT_DIR = Path("out/sd-15-lcm-inpaint-fp16")
BASE_INPAINT = "runwayml/stable-diffusion-inpainting"
LCM_LORA = "latent-consistency/lcm-lora-sdv1-5"


def main() -> int:
    require("torch")
    require("diffusers")
    require("transformers")
    require("peft")
    require("optimum")
    require("onnx")

    import torch
    from diffusers import StableDiffusionInpaintPipeline, LCMScheduler
    from optimum.exporters.onnx import main_export

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading base inpaint pipeline: {BASE_INPAINT}")
    # FP16 throughout the pipeline. CPU-only export is fine and avoids
    # CUDA driver ambiguity; ONNX export does its own tracing pass.
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        BASE_INPAINT,
        torch_dtype=torch.float16,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )

    print(f"[2/4] Loading LCM-LoRA: {LCM_LORA}")
    pipe.load_lora_weights(LCM_LORA)

    print("[3/4] Fusing LoRA weights into UNet (in-place)…")
    # `fuse_lora` bakes the LoRA delta into the base UNet weights so we
    # can ONNX-export without needing the LoRA adapter at runtime.
    # Without this, the exporter trips over the PEFT modules.
    pipe.fuse_lora()
    pipe.unload_lora_weights()

    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # Save the merged pipeline to disk so optimum can find a checkpoint
    # to export. main_export expects a HF model id or local path; the
    # local-path route avoids a network round-trip on the export pass.
    merged_dir = OUT_DIR.parent / "_merged_pipeline"
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    pipe.save_pretrained(merged_dir, safe_serialization=True)

    # Free the python pipeline before invoking the exporter — optimum
    # loads its own copy and we don't want to hold ~4 GB twice.
    del pipe
    import gc
    gc.collect()

    print(f"[4/4] Exporting to ONNX → {OUT_DIR}")
    main_export(
        model_name_or_path=str(merged_dir),
        output=OUT_DIR,
        task="stable-diffusion",
        # SD 1.5 inpaint sample shape — 9-channel UNet input (4 latent +
        # 1 mask + 4 masked-image latent) at 64x64 latent resolution =
        # 512x512 image.
        device="cpu",
        framework="pt",
        # FP16 ops for UNet; rest stays FP32 by default, matching our
        # standard SD bundle's mixed-precision layout.
        dtype="fp16",
    )

    # Sanity-check the four expected ONNX files actually landed.
    expected = ["text_encoder", "vae_encoder", "vae_decoder", "unet"]
    for part in expected:
        candidates = list(OUT_DIR.rglob(f"*{part}*.onnx"))
        if not candidates:
            print(f"  MISSING: {part}.onnx not found in {OUT_DIR}", file=sys.stderr)
            return 2
        # Flatten into the bundle root so prunr's loader (which reads
        # by literal filename) finds them without sub-directory hops.
        root_target = OUT_DIR / f"{part}.onnx"
        if candidates[0] != root_target:
            shutil.move(candidates[0], root_target)
            print(f"  ✓ {part}.onnx (moved to bundle root)")
        else:
            print(f"  ✓ {part}.onnx")

    # Compute SHA256 of each part for the prunr-models registry entry.
    print("\n=== SHA256 (paste into prunr-models registry) ===")
    import hashlib
    for part in expected:
        path = OUT_DIR / f"{part}.onnx"
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {part}.onnx  {size_mb:.1f} MB  {h}")

    # Clean up the intermediate merged pipeline.
    shutil.rmtree(merged_dir, ignore_errors=True)
    print(f"\nDone. Bundle at {OUT_DIR.resolve()}")
    print("Upload these 4 .onnx files to your prunr GitHub release.")
    return 0


def require(module: str) -> None:
    """Friendlier failure than ImportError when the user is missing a dep."""
    try:
        __import__(module)
    except ImportError:
        print(
            f"error: missing Python package '{module}'.\n"
            f"  pip install diffusers transformers torch peft accelerate optimum onnx",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
