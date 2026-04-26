#!/usr/bin/env python3
"""Export Big-LaMa from PyTorch to ONNX (compatible with ORT 2.0).

Pipeline:
  1. Download `big-lama.zip` from smartywu/big-lama on HuggingFace
     (the advimman/lama checkpoint, mirrored anonymously). 381 MB.
  2. Extract to `/tmp/big-lama/`. Contains `models/best.ckpt` + `config.yaml`.
  3. Clone advimman/lama for the model definitions (FFC/ResNet blocks).
  4. Instantiate the generator, load weights, wrap in an ONNX-friendly
     forward (RGBA image input gets split into image + mask the same way
     `LamaSession::run_tile` does in prunr-core::inpaint).
  5. JIT-trace with a 512x512 dummy, export to ONNX with fixed input
     shape (matches `prunr-core::inpaint::TILE = 512`).
  6. zstd-compress to `big_lama.onnx.zst`.
  7. Print SHA256 + the file path. Hardcode the SHA into
     `prunr-models::REGISTRY` and upload via `gh release upload models-v1`.

DEPENDENCIES (run in a fresh venv to avoid clashes):
    pip install torch==2.4.0 numpy pyyaml omegaconf hydra-core \\
                pytorch-lightning kornia einops onnx

CHECKPOINT URL is verified working as of 2026-04 (smartywu HF mirror,
381 MB zip). If it 404s in the future try:
  - https://huggingface.co/Sanster/models (gated; needs HF token)
  - Direct from advimman/lama README (Google Drive, manual)

The script is idempotent: re-running uses cached downloads + cloned repo.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

# ── config ─────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
TMP = Path("/tmp/prunr-big-lama-export")
TMP.mkdir(parents=True, exist_ok=True)

CHECKPOINT_URL = "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip"
CHECKPOINT_ZIP = TMP / "big-lama.zip"
CHECKPOINT_DIR = TMP / "big-lama"
LAMA_REPO_URL = "https://github.com/advimman/lama.git"
LAMA_REPO_DIR = TMP / "lama"

OUT_ONNX = MODELS_DIR / "big_lama.onnx"
TILE = 512  # matches prunr-core::inpaint::TILE


# ── steps ──────────────────────────────────────────────────────────────

def step(msg: str) -> None:
    print(f"\n=== {msg} ===")


def download_checkpoint() -> None:
    if CHECKPOINT_ZIP.exists():
        print(f"Cached: {CHECKPOINT_ZIP} ({CHECKPOINT_ZIP.stat().st_size / 1024 / 1024:.1f} MB)")
        return
    step(f"Downloading {CHECKPOINT_URL}")
    print("(~381 MB — first run only)")
    urllib.request.urlretrieve(CHECKPOINT_URL, str(CHECKPOINT_ZIP))
    print(f"Saved: {CHECKPOINT_ZIP}")


def extract_checkpoint() -> None:
    if (CHECKPOINT_DIR / "models" / "best.ckpt").exists():
        print(f"Cached extraction: {CHECKPOINT_DIR}")
        return
    step(f"Extracting {CHECKPOINT_ZIP}")
    with zipfile.ZipFile(CHECKPOINT_ZIP) as z:
        z.extractall(TMP)
    if not (CHECKPOINT_DIR / "models" / "best.ckpt").exists():
        raise SystemExit(
            f"Extraction did not produce expected layout under {CHECKPOINT_DIR}. "
            f"Got: {list(CHECKPOINT_DIR.glob('**/*.ckpt'))}"
        )


def clone_lama_repo() -> None:
    if (LAMA_REPO_DIR / "saicinpainting").exists():
        print(f"Cached repo: {LAMA_REPO_DIR}")
        return
    step(f"Cloning {LAMA_REPO_URL}")
    subprocess.run(["git", "clone", "--depth=1", LAMA_REPO_URL, str(LAMA_REPO_DIR)], check=True)


def load_model():
    """Instantiate the FFC ResNet generator directly + load weights from
    the checkpoint. Skips advimman/lama's training-module imports (which
    require sklearn / webdataset / scikit-image / etc that aren't needed
    just to *run* inference). Returns a torch.nn.Module that takes
    (image, mask) and returns the inpainted image — both as f32 NCHW
    tensors in [0, 1]."""
    sys.path.insert(0, str(LAMA_REPO_DIR))
    import torch
    import yaml
    from omegaconf import OmegaConf
    # Direct generator import — the only module path that doesn't pull
    # the entire trainer subgraph.
    from saicinpainting.training.modules.ffc import FFCResNetGenerator  # type: ignore

    config_path = CHECKPOINT_DIR / "config.yaml"
    ckpt_path = CHECKPOINT_DIR / "models" / "best.ckpt"

    step(f"Loading config: {config_path}")
    with open(config_path) as f:
        cfg = OmegaConf.create(yaml.safe_load(f))

    # The config's `generator:` block is the constructor kwargs for
    # FFCResNetGenerator. Drop the `kind` discriminator key.
    gen_kwargs = OmegaConf.to_container(cfg.generator, resolve=True)
    gen_kwargs.pop("kind", None)
    step(f"Generator config: {sorted(gen_kwargs.keys())}")
    generator = FFCResNetGenerator(**gen_kwargs)

    step(f"Loading checkpoint weights: {ckpt_path}")
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state)
    # Trainer prepends `generator.` to all weights — strip it.
    gen_state = {
        k[len("generator."):]: v
        for k, v in sd.items()
        if k.startswith("generator.")
    }
    missing, unexpected = generator.load_state_dict(gen_state, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:3]}…")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:3]}…")
    generator.eval()
    for p in generator.parameters():
        p.requires_grad_(False)

    class LamaForward(torch.nn.Module):
        """Thin wrapper presenting the (image, mask) → painted_image
        contract our prunr-core inpaint code expects. The generator
        takes a 4-channel input (image concat masked-image, see LaMa
        paper §3.3). We do that concat here so the ONNX signature is
        the natural (image, mask) we already use in `LamaSession`."""
        def __init__(self, gen):
            super().__init__()
            self.gen = gen

        def forward(self, image, mask):  # NCHW f32, [0, 1]
            # LaMa input: image with masked region zeroed, concat mask.
            # `(1 - mask) * image` zeroes the inpaint region; mask is the
            # 4th channel telling the network where to fill.
            masked = (1.0 - mask) * image
            x = torch.cat([masked, mask], dim=1)
            out = self.gen(x)
            # Composite: keep source where mask=0, use output where mask=1.
            return out * mask + image * (1.0 - mask)

    return LamaForward(generator)


def export_onnx(model) -> None:
    import torch
    import onnx

    step(f"Tracing + exporting to {OUT_ONNX}")
    image = torch.zeros(1, 3, TILE, TILE)
    mask = torch.zeros(1, 1, TILE, TILE)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # torch.onnx.export (dynamo path in torch 2.x) writes weights to a
    # sidecar `.data` file by default. We want a single self-contained
    # .onnx — re-load with onnx.load(load_external_data=True) and save
    # without the external-data flag to inline.
    torch.onnx.export(
        model,
        (image, mask),
        str(OUT_ONNX),
        input_names=["image", "mask"],
        output_names=["inpainted"],
        opset_version=17,
        do_constant_folding=True,
    )

    sidecar = OUT_ONNX.with_suffix(".onnx.data")
    if sidecar.exists():
        step("Inlining external weights")
        m = onnx.load(str(OUT_ONNX), load_external_data=True)
        # Strip external-data references; weights now live in the proto.
        for tensor in m.graph.initializer:
            if tensor.HasField("data_location") and tensor.data_location == onnx.TensorProto.EXTERNAL:
                tensor.ClearField("external_data")
                tensor.data_location = onnx.TensorProto.DEFAULT
        onnx.save(m, str(OUT_ONNX))
        sidecar.unlink()

    # FFC layers in advimman/lama use FFT/IFFT for spectral convolutions.
    # torch.onnx exports the inverse FFT with `onesided=1` inherited from
    # the forward FFT, which ONNX rejects at load time. Patcher lives in
    # `patch_big_lama_dft.py` so users with an already-exported file can
    # run it directly without re-tracing the model.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from patch_big_lama_dft import patch as patch_dft  # noqa: PLC0415
    step("Patching inverse-DFT onesided attribute")
    patch_dft(OUT_ONNX)

    size_mb = OUT_ONNX.stat().st_size / 1024 / 1024
    print(f"Exported: {OUT_ONNX.name} ({size_mb:.1f} MB)")


def compress_zst() -> None:
    step("Compressing to .zst")
    zst_path = OUT_ONNX.with_suffix(".onnx.zst")
    subprocess.run(
        ["zstd", "-19", "--force", "-q", str(OUT_ONNX), "-o", str(zst_path)],
        check=True,
    )
    zst_mb = zst_path.stat().st_size / 1024 / 1024
    print(f"Compressed: {zst_path.name} ({zst_mb:.1f} MB)")


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
    print("  crates/prunr-models/src/lib.rs    (REGISTRY → BigLaMa → sha256)")
    print()
    print("Then upload via:")
    print(f"  gh release upload models-v1 {OUT_ONNX} \\")
    print(f"    --repo aktiwers/prunr --clobber")


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


def main() -> None:
    download_checkpoint()
    extract_checkpoint()
    clone_lama_repo()
    model = load_model()
    export_onnx(model)
    verify_onnx()
    compress_zst()
    print_sha256()


if __name__ == "__main__":
    main()
