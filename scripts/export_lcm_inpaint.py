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
- diffusers, transformers, peft, torch, onnx, onnxconverter-common
- ~12 GB free disk during export (fp32 graphs before conversion)
- ~12 GB free RAM (the fp32 UNet proto and its fp16 copy overlap)

The LCM-LoRA (latent-consistency/lcm-lora-sdv1-5) is FUSED into the
base inpaint UNet before export so the runtime side has nothing
LoRA-aware to deal with.

Usage:
    pip install diffusers transformers peft torch onnx onnxconverter-common accelerate
    python scripts/export_lcm_inpaint.py
"""

import gc
import hashlib
import sys
from pathlib import Path

OUT_DIR = Path("out/sd-15-lcm-inpaint-fp16")
# Dynamic batch and spatial axes: the runtime batches cond+uncond into one
# UNet call and runs crops up to 512×768, so every part must accept them.
BATCH_HW = {0: "batch", 2: "height", 3: "width"}
# `dynamo=False`: the TorchScript exporter honours `dynamic_axes` and
# produced the SD 1.5 graph the runtime is tuned for; torch >= 2.9
# defaults to the dynamo exporter, which warns that dynamic_axes may
# violate its constraints and failed converting Pad to opset 17.
EXPORT_KW = dict(opset_version=17, do_constant_folding=True, dynamo=False)
PARTS = ["text_encoder", "vae_encoder", "vae_decoder", "unet"]
BASE_INPAINT = "botp/stable-diffusion-v1-5-inpainting"
LCM_LORA = "latent-consistency/lcm-lora-sdv1-5"


def repair_fp16_graph(g):
    """onnxconverter_common leaves three things inconsistent without shape
    inference: the model's own `Cast(to=FLOAT)` nodes (from `.float()`
    calls) keep producing fp32 into fp16 consumers, constants feeding the
    ops it keeps in fp32 (Resize scales, Range limits) were narrowed to
    fp16, and stale value_info still declares fp32. onnxruntime refuses
    all three at load."""
    import numpy as np
    from onnx import TensorProto, numpy_helper
    from onnxconverter_common.float16 import DEFAULT_OP_BLOCK_LIST
    block = set(DEFAULT_OP_BLOCK_LIST)
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    for n in g.node:
        if n.op_type != "Cast":
            continue
        feeds_blocked = any(c.op_type in block for o in n.output for c in consumers.get(o, []))
        want = TensorProto.FLOAT if feeds_blocked else TensorProto.FLOAT16
        for a in n.attribute:
            if a.name == "to" and a.i in (TensorProto.FLOAT, TensorProto.FLOAT16):
                a.i = want
    producers = {o: n for n in g.node for o in n.output}
    inits = {i.name: i for i in g.initializer}
    for n in g.node:
        if n.op_type not in block:
            continue
        for i in n.input:
            t = inits.get(i)
            if t is None and i in producers and producers[i].op_type == "Constant":
                t = next((a.t for a in producers[i].attribute if a.name == "value"), None)
            if t is not None and t.data_type == TensorProto.FLOAT16:
                t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float32), t.name))
    del g.value_info[:]


def main() -> int:
    require("torch")
    require("diffusers")
    require("transformers")
    require("peft")
    require("onnx")
    require("onnxconverter_common")

    import torch
    from diffusers import StableDiffusionInpaintPipeline

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading base inpaint pipeline: {BASE_INPAINT}")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        BASE_INPAINT,
        # Trace in fp32: PyTorch's CPU fp16 convolutions are so slow that
        # the VAE alone took over half an hour. The graphs are converted to
        # fp16 below, which is how the published SD 1.5 export was made.
        torch_dtype=torch.float32,
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
        input_names=["input_ids"], output_names=["last_hidden_state"], **EXPORT_KW,
        dynamic_axes={"input_ids": {0: "batch"}, "last_hidden_state": {0: "batch"}},
    )

    print("[5/6] Exporting VAE encoder + decoder…")
    sample_image = torch.randn(1, 3, 512, 512, dtype=torch.float32)
    sample_latent = torch.randn(1, 4, 64, 64, dtype=torch.float32)

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
        input_names=["sample"], output_names=["latent_sample"], **EXPORT_KW,
        dynamic_axes={"sample": BATCH_HW, "latent_sample": BATCH_HW},
    )

    class VaeDecWrap(torch.nn.Module):
        def __init__(self, vae): super().__init__(); self.vae = vae
        def forward(self, latent_sample):
            return self.vae.decode(latent_sample).sample
    torch.onnx.export(
        VaeDecWrap(vae), sample_latent, str(OUT_DIR / "vae_decoder.onnx"),
        input_names=["latent_sample"], output_names=["sample"], **EXPORT_KW,
        dynamic_axes={"latent_sample": BATCH_HW, "sample": BATCH_HW},
    )

    print("[6/6] Exporting UNet (~860M params; ~5 min on CPU)…")
    # SD 1.5 INPAINT UNet takes 9-channel input: 4 latent + 1 mask + 4 masked-image-latent.
    sample_unet = torch.randn(1, 9, 64, 64, dtype=torch.float32)
    timestep = torch.tensor([1], dtype=torch.float32)
    encoder_hidden_states = torch.randn(1, 77, 768, dtype=torch.float32)

    class UnetWrap(torch.nn.Module):
        def __init__(self, unet): super().__init__(); self.unet = unet
        def forward(self, sample, timestep, encoder_hidden_states):
            return self.unet(sample, timestep, encoder_hidden_states).sample
    torch.onnx.export(
        UnetWrap(unet), (sample_unet, timestep, encoder_hidden_states),
        str(OUT_DIR / "unet.onnx"),
        input_names=["sample", "timestep", "encoder_hidden_states"],
        output_names=["out_sample"], **EXPORT_KW,
        dynamic_axes={
            "sample": BATCH_HW, "timestep": {0: "batch"},
            "encoder_hidden_states": {0: "batch"}, "out_sample": BATCH_HW,
        },
    )

    # The conversion below overlaps a fp32 proto with its fp16 copy; the
    # torch weights would push that past what a 32 GB host has free.
    del pipe, unet, vae, text_encoder
    gc.collect()

    print("\nConverting to fp16 and inlining the weights…")
    import onnx
    from onnxconverter_common import float16
    for part in PARTS:
        p = OUT_DIR / f"{part}.onnx"
        m = onnx.load(str(p), load_external_data=True)
        # Shape inference is off: it cannot run on the >2 GB UNet proto.
        m = float16.convert_float_to_float16(m, keep_io_types=False, disable_shape_infer=True)
        repair_fp16_graph(m.graph)
        onnx.save_model(m, str(p), save_as_external_data=False)
        print(f"  {part}: converted")
    # torch wrote the large weights as sidecar files; the saves above
    # folded them in, so drop them only once every part has been read.
    for stale in [*OUT_DIR.glob("*.onnx.data"), *OUT_DIR.glob("onnx__*")]:
        stale.unlink()

    print("\n=== SHA256 (paste into prunr-models registry) ===")
    for part in PARTS:
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
