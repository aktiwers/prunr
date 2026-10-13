#!/usr/bin/env python3
"""
Export RORem-4S (SDXL inpaint fine-tuned for object removal, with its
4-step LCM LoRA fused in) to ONNX.

Output (OUT_DIR):
    unet.onnx + unet.onnx.data   fp16, prompt embeddings baked in
    vae_encoder.onnx             fp32 (the SDXL VAE overflows in fp16)
    vae_decoder.onnx             fp32

The removal prompt is fixed and guidance is off, so the UNet carries the
prompt's embeddings as constants: the runtime needs no text encoder (two
of them, ~1.6 GB, for SDXL). Its inputs are `sample` (9 channels: latent,
mask, masked-image latent), `timestep` and `time_ids` (SDXL's size
conditioning: original h/w, crop top/left, target h/w).

Weights:
    LetsThink/RORem (Apache-2.0; SDXL-derived, CreativeML OpenRAIL++-M)
    RORem LCM LoRA, from the authors' Google Drive (README of
    github.com/leeruibin/RORem), as pytorch_lora_weights.safetensors

Run on a host with ~40 GB free disk and ~28 GB free RAM:
    pip install diffusers transformers peft torch onnx onnxconverter-common accelerate
    python scripts/export_rorem.py <rorem pipeline dir> <lora dir> <out dir>
"""

import gc
import hashlib
import sys
from pathlib import Path

PROMPT = "4K, high quality, masterpiece, Highly detailed, Sharp focus, Professional, photorealistic, realistic"
EXPORT_KW = dict(opset_version=17, do_constant_folding=True, dynamo=False)
BATCH_HW = {0: "batch", 2: "height", 3: "width"}


def main() -> int:
    import torch
    from diffusers import AutoPipelineForInpainting, LCMScheduler

    pipe_dir, lora_dir, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading RORem and fusing the LCM LoRA")
    pipe = AutoPipelineForInpainting.from_pretrained(pipe_dir, torch_dtype=torch.float32)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(lora_dir)
    pipe.fuse_lora()
    pipe.unload_lora_weights()

    print("[2/5] Encoding the fixed prompt")
    with torch.no_grad():
        embeds, _, pooled, _ = pipe.encode_prompt(PROMPT, device="cpu", do_classifier_free_guidance=False)
    del pipe.text_encoder, pipe.text_encoder_2
    gc.collect()

    print("[3/5] Exporting the VAE (fp32)")
    vae = pipe.vae.eval()

    class VaeEnc(torch.nn.Module):
        def __init__(self, vae): super().__init__(); self.vae = vae
        def forward(self, sample):
            return self.vae.encode(sample).latent_dist.mode()

    class VaeDec(torch.nn.Module):
        def __init__(self, vae): super().__init__(); self.vae = vae
        def forward(self, latent_sample):
            return self.vae.decode(latent_sample).sample

    with torch.no_grad():
        torch.onnx.export(VaeEnc(vae), torch.randn(1, 3, 512, 512), str(out / "vae_encoder.onnx"),
                          input_names=["sample"], output_names=["latent_sample"], **EXPORT_KW,
                          dynamic_axes={"sample": BATCH_HW, "latent_sample": BATCH_HW})
        torch.onnx.export(VaeDec(vae), torch.randn(1, 4, 64, 64), str(out / "vae_decoder.onnx"),
                          input_names=["latent_sample"], output_names=["sample"], **EXPORT_KW,
                          dynamic_axes={"latent_sample": BATCH_HW, "sample": BATCH_HW})

    print("[4/5] Exporting the UNet (fp32 first; ~2.6B params)")
    unet = pipe.unet.eval()

    class Unet(torch.nn.Module):
        def __init__(self, unet, embeds, pooled):
            super().__init__()
            self.unet = unet
            self.register_buffer("embeds", embeds)
            self.register_buffer("pooled", pooled)

        def forward(self, sample, timestep, time_ids):
            b = sample.shape[0]
            cond = {"text_embeds": self.pooled.expand(b, -1), "time_ids": time_ids}
            return self.unet(sample, timestep, encoder_hidden_states=self.embeds.expand(b, -1, -1),
                             added_cond_kwargs=cond).sample

    raw = out / "raw"
    raw.mkdir(exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            Unet(unet, embeds, pooled),
            (torch.randn(1, 9, 64, 64), torch.tensor([999.0]), torch.tensor([[512.0, 512, 0, 0, 512, 512]])),
            str(raw / "unet.onnx"),
            input_names=["sample", "timestep", "time_ids"], output_names=["out_sample"], **EXPORT_KW,
            dynamic_axes={"sample": BATCH_HW, "timestep": {0: "batch"}, "time_ids": {0: "batch"},
                          "out_sample": BATCH_HW},
        )
    del pipe, unet, vae
    gc.collect()

    print("[5/5] Converting the UNet to fp16")
    import onnx
    from onnxconverter_common import float16
    sys.path.insert(0, str(Path(__file__).parent))
    from export_lcm_inpaint import repair_fp16_graph
    m = onnx.load(str(raw / "unet.onnx"), load_external_data=True)
    m = float16.convert_float_to_float16(m, keep_io_types=False, disable_shape_infer=True)
    repair_fp16_graph(m.graph)
    onnx.save_model(m, str(out / "unet.onnx"), save_as_external_data=True,
                    all_tensors_to_one_file=True, location="unet.onnx.data")

    print("\n=== SHA256 ===")
    for name in ["unet.onnx", "unet.onnx.data", "vae_encoder.onnx", "vae_decoder.onnx"]:
        p = out / name
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        print(f"  {name}  size={p.stat().st_size}  sha256={h}")
    print(f"\nDone. The fp32 UNet in {raw} is only an intermediate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
