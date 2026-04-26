#!/usr/bin/env python3
"""Export MI-GAN from PyTorch JIT to ONNX (compatible with ORT 2.0).

Pipeline:
  1. Download the pre-traced `migan_traced.pt` from Sanster/models on
     GitHub Releases (anonymous, ~27 MB). This is the same artefact
     iopaint uses, JIT-traced from the original Picsart MI-GAN repo.
  2. Wrap in a thin forward that matches our `(image, mask) →
     painted_image` contract — same shape we use for LaMa-family
     backends in `prunr-core::inpaint::process_inpaint`.
  3. Export to ONNX with fixed 512×512 input (matches our TILE).
  4. zstd-compress, print SHA256, print upload command.

CONTRACT (matches `prunr-core::inpaint::LamaSession::run_tile`):
  Inputs:
    image — [1, 3, 512, 512] f32 in [0, 1]
    mask  — [1, 1, 512, 512] f32 in {0, 1}, 1 = inpaint here
  Output:
    inpainted — [1, 3, 512, 512] f32 in [0, 1]

DEPENDENCIES:
    pip install torch onnx

LICENSE: MI-GAN is MIT-licensed (Picsart AI Research).
"""
import hashlib
import subprocess
import urllib.request
from pathlib import Path

# ── config ─────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
TMP = Path("/tmp/prunr-migan-export")
TMP.mkdir(parents=True, exist_ok=True)

JIT_URL = "https://github.com/Sanster/models/releases/download/migan/migan_traced.pt"
JIT_PATH = TMP / "migan_traced.pt"
OUT_ONNX = MODELS_DIR / "migan.onnx"
TILE = 512


def step(msg: str) -> None:
    print(f"\n=== {msg} ===")


def download() -> None:
    if JIT_PATH.exists():
        print(f"Cached: {JIT_PATH} ({JIT_PATH.stat().st_size / 1024 / 1024:.1f} MB)")
        return
    step(f"Downloading {JIT_URL}")
    urllib.request.urlretrieve(JIT_URL, str(JIT_PATH))
    print(f"Saved: {JIT_PATH}")


def build_wrapper():
    """Load the JIT model + wrap it in our (image, mask) contract.
    The traced model expects a 4-channel input where channel 0 is
    `0.5 - mask` and channels 1-3 are the masked image in [-1, 1]
    (per iopaint/model/mi_gan.py:forward). We re-wrap so the ONNX
    surface matches LaMa's (image, mask)."""
    import torch

    step("Loading JIT model")
    inner = torch.jit.load(str(JIT_PATH), map_location="cpu").eval()

    class MiganForward(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, image, mask):  # NCHW f32 [0, 1]
            # Match iopaint's preprocessing: image to [-1, 1], mask
            # threshold-binarised, concat with `0.5 - mask` channel 0.
            img_pm1 = image * 2.0 - 1.0
            erased = img_pm1 * (1.0 - mask)
            x = torch.cat([0.5 - mask, erased], dim=1)
            out_pm1 = self.m(x)  # [-1, 1]
            out = (out_pm1 * 0.5 + 0.5).clamp(0.0, 1.0)
            # Composite: keep source where mask=0, painted where mask=1.
            return out * mask + image * (1.0 - mask)

    return MiganForward(inner)


def export_onnx(model) -> None:
    import torch
    import onnx

    step(f"Exporting to {OUT_ONNX}")
    image = torch.zeros(1, 3, TILE, TILE)
    mask = torch.zeros(1, 1, TILE, TILE)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Re-trace the whole `(image, mask) → painted` pipeline as a single
    # ScriptModule. Direct-tracing the wrapper Module fails because the
    # inner JIT model isn't registered as a submodule for the active
    # trace; jit.trace returns a fresh ScriptModule that includes the
    # whole graph, which torch.onnx.export then handles cleanly.
    traced = torch.jit.trace(model, (image, mask), strict=False)

    # `dynamo=False` uses the legacy tracer-based exporter — needed
    # because traced/scripted modules carry weights as raw tensors
    # (not nn.Parameters), which the new dynamo exporter rejects.
    torch.onnx.export(
        traced,
        (image, mask),
        str(OUT_ONNX),
        input_names=["image", "mask"],
        output_names=["inpainted"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    sidecar = OUT_ONNX.with_suffix(".onnx.data")
    if sidecar.exists():
        step("Inlining external weights")
        m = onnx.load(str(OUT_ONNX), load_external_data=True)
        for tensor in m.graph.initializer:
            if tensor.HasField("data_location") and tensor.data_location == onnx.TensorProto.EXTERNAL:
                tensor.ClearField("external_data")
                tensor.data_location = onnx.TensorProto.DEFAULT
        onnx.save(m, str(OUT_ONNX))
        sidecar.unlink()

    size_mb = OUT_ONNX.stat().st_size / 1024 / 1024
    print(f"Exported: {OUT_ONNX.name} ({size_mb:.1f} MB)")


def verify_onnx() -> None:
    step("Verifying ONNX shape")
    import onnx
    m = onnx.load(str(OUT_ONNX))
    for inp in m.graph.input:
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"  Input {inp.name}: {dims}")
    for out in m.graph.output:
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  Output {out.name}: {dims}")


def compress_zst() -> None:
    step("Compressing to .zst")
    zst_path = OUT_ONNX.with_suffix(".onnx.zst")
    subprocess.run(
        ["zstd", "-19", "--force", "-q", str(OUT_ONNX), "-o", str(zst_path)],
        check=True,
    )
    print(f"Compressed: {zst_path.name} ({zst_path.stat().st_size / 1024 / 1024:.1f} MB)")


def print_sha256() -> None:
    step("SHA256")
    h = hashlib.sha256()
    with open(OUT_ONNX, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    digest = h.hexdigest()
    print(f"  {digest}")
    print()
    print("Hardcode this in:")
    print("  crates/prunr-models/src/lib.rs    (REGISTRY → MIGAN → sha256)")
    print()
    print("Then upload via:")
    print(f"  gh release upload models-v1 {OUT_ONNX} \\")
    print(f"    --repo aktiwers/prunr --clobber")


def main() -> None:
    download()
    model = build_wrapper()
    export_onnx(model)
    verify_onnx()
    compress_zst()
    print_sha256()


if __name__ == "__main__":
    main()
