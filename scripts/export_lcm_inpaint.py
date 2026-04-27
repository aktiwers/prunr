#!/usr/bin/env python3
"""
Export LCM-distilled SD 1.5 Inpaint to 4 ONNX files via direct torch.onnx.

The earlier optimum-based exporter has API drift with diffusers >= 0.30
(NormalizedConfig.__init__ kwargs clash). We drive each part by hand
through torch.onnx.export — same idiom as scripts/export_taesd.py —
which sidesteps optimum entirely.

Output:
    text_encoder.onnx
    vae_encoder.onnx
    vae_decoder.onnx
    unet.onnx

Run on a host with:
- diffusers, transformers, peft, torch, onnx
- ~8 GB free disk during export
- ~6 GB free RAM (UNet is ~860M params at FP16)

The LCM-LoRA (latent-consistency/lcm-lora-sdv1-5) is FUSED into the
base inpaint UNet before export so the runtime side has nothing
LoRA-aware to deal with.

Usage:
    pip install diffusers transformers peft torch onnx accelerate
    python scripts/export_lcm_inpaint.py
"""

import hashlib
import sys
from pathlib import Path

OUT_DIR = Path("out/sd-15-lcm-inpaint-fp16")
BASE_INPAINT = "botp/stable-diffusion-v1-5-inpainting"
LCM_LORA = "latent-consistency/lcm-lora-sdv1-5"


def main() -> int:
    require("torch")
    require("diffusers")
    require("transformers")
    require("peft")
    require("onnx")

    import torch
    from diffusers import StableDiffusionInpaintPipeline

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading base inpaint pipeline: {BASE_INPAINT}")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        BASE_INPAINT,
        torch_dtype=torch.float16,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )

    print(f"[2/6] Loading LCM-LoRA: {LCM_LORA}")
    pipe.load_lora_weights(LCM_LORA)

    print("[3/6] Fusing LoRA into UNet…")
    pipe.fuse_lora()
    pipe.unload_lora_weights()
    pipe = pipe.to("cpu")

    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet
    text_encoder.eval(); vae.eval(); unet.eval()

    print("[4/6] Exporting text_encoder…")
    sample_ids = torch.zeros(1, 77, dtype=torch.int32)
    class TextEncWrap(torch.nn.Module):
        def __init__(self, te): super().__init__(); self.te = te
        def forward(self, input_ids):
            return self.te(input_ids=input_ids.to(torch.long)).last_hidden_state
    torch.onnx.export(
        TextEncWrap(text_encoder), sample_ids, str(OUT_DIR / "text_encoder.onnx"),
        input_names=["input_ids"], output_names=["last_hidden_state"], opset_version=17,
        do_constant_folding=True,
    )

    print("[5/6] Exporting VAE encoder + decoder…")
    sample_image = torch.randn(1, 3, 512, 512, dtype=torch.float16)
    sample_latent = torch.randn(1, 4, 64, 64, dtype=torch.float16)

    class VaeEncWrap(torch.nn.Module):
        def __init__(self, vae): super().__init__(); self.vae = vae
        def forward(self, sample):
            # Diffusers' AutoencoderKL.encode returns AutoencoderKLOutput
            # whose .latent_dist.sample() is the conventional path. The
            # caller scales by VAE_SCALING_FACTOR (0.18215) — we mirror
            # that runtime-side, so the ONNX output here stays unscaled.
            posterior = self.vae.encode(sample).latent_dist
            return posterior.mode()
    torch.onnx.export(
        VaeEncWrap(vae), sample_image, str(OUT_DIR / "vae_encoder.onnx"),
        input_names=["sample"], output_names=["latent_sample"], opset_version=17,
        do_constant_folding=True,
    )

    class VaeDecWrap(torch.nn.Module):
        def __init__(self, vae): super().__init__(); self.vae = vae
        def forward(self, latent_sample):
            return self.vae.decode(latent_sample).sample
    torch.onnx.export(
        VaeDecWrap(vae), sample_latent, str(OUT_DIR / "vae_decoder.onnx"),
        input_names=["latent_sample"], output_names=["sample"], opset_version=17,
        do_constant_folding=True,
    )

    print("[6/6] Exporting UNet (~860M params; ~5 min on CPU)…")
    # SD 1.5 INPAINT UNet takes 9-channel input: 4 latent + 1 mask + 4 masked-image-latent.
    sample_unet = torch.randn(1, 9, 64, 64, dtype=torch.float16)
    timestep = torch.tensor([1], dtype=torch.float16)  # f16 for SD-15-fp16 export
    encoder_hidden_states = torch.randn(1, 77, 768, dtype=torch.float16)

    class UnetWrap(torch.nn.Module):
        def __init__(self, unet): super().__init__(); self.unet = unet
        def forward(self, sample, timestep, encoder_hidden_states):
            return self.unet(sample, timestep, encoder_hidden_states).sample
    torch.onnx.export(
        UnetWrap(unet), (sample_unet, timestep, encoder_hidden_states),
        str(OUT_DIR / "unet.onnx"),
        input_names=["sample", "timestep", "encoder_hidden_states"],
        output_names=["out_sample"], opset_version=17,
        do_constant_folding=True,
    )

    # Inline external-data weights so each .onnx is single-file (matches
    # our loader). torch's exporter writes weights as sidecar .onnx.data
    # files when models are large; we round-trip through onnx.save with
    # save_as_external_data=False to fold them in.
    print("\nInlining external data into single-file ONNX…")
    import onnx
    for part in ["text_encoder", "vae_encoder", "vae_decoder", "unet"]:
        p = OUT_DIR / f"{part}.onnx"
        m = onnx.load(str(p), load_external_data=True)
        onnx.save_model(m, str(p), save_as_external_data=False)
        data = p.with_suffix(".onnx.data")
        if data.exists(): data.unlink()

    print("\n=== SHA256 (paste into prunr-models registry) ===")
    for part in ["text_encoder", "vae_encoder", "vae_decoder", "unet"]:
        p = OUT_DIR / f"{part}.onnx"
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        size = p.stat().st_size
        print(f"  {part}.onnx  size={size}  sha256={h}")

    print(f"\nDone. Bundle at {OUT_DIR.resolve()}")
    return 0


def require(module: str) -> None:
    try:
        __import__(module)
    except ImportError:
        print(f"error: missing Python package '{module}'.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
