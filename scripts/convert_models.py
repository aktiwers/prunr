#!/usr/bin/env python3
"""Convert ONNX models to FP16 (GPU) and INT8 (CPU) variants.

Requirements:
    pip install onnx onnxconverter-common onnxruntime

Usage:
    python scripts/convert_models.py

Reads from models/*.onnx, writes *_fp16.onnx and *_int8.onnx alongside them,
then compresses with zstd.
"""

import subprocess
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS = ["silueta", "u2net", "birefnet_lite"]


def convert_fp16(src: Path, dst: Path):
    """Convert FP32 ONNX model to FP16, keeping I/O types as FP32."""
    import onnx
    from onnxconverter_common import float16

    print(f"  FP16: {src.name} -> {dst.name}")
    model = onnx.load(str(src))
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,       # inputs/outputs stay FP32 for compatibility
        min_positive_val=1e-7,
        max_finite_val=1e4,
    )
    onnx.save(model_fp16, str(dst))
    print(f"    {dst.stat().st_size / 1024 / 1024:.1f} MB")


def convert_int8(src: Path, dst: Path):
    """Dynamic INT8 quantization for CPU inference."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print(f"  INT8: {src.name} -> {dst.name}")
    quantize_dynamic(
        str(src),
        str(dst),
        weight_type=QuantType.QInt8,
    )
    print(f"    {dst.stat().st_size / 1024 / 1024:.1f} MB")


def compress_zstd(src: Path):
    """Compress with zstd level 19 (matches existing .zst files)."""
    dst = src.with_suffix(src.suffix + ".zst")
    print(f"  ZSTD: {src.name} -> {dst.name}")
    subprocess.run(
        ["zstd", "-19", "--force", "-q", str(src), "-o", str(dst)],
        check=True,
    )
    print(f"    {dst.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    missing = [m for m in MODELS if not (MODELS_DIR / f"{m}.onnx").exists()]
    if missing:
        print(f"Missing models: {missing}")
        print(f"Run: cargo xtask fetch-models")
        sys.exit(1)

    for name in MODELS:
        src = MODELS_DIR / f"{name}.onnx"
        print(f"\n=== {name} ({src.stat().st_size / 1024 / 1024:.1f} MB FP32) ===")

        fp16_dst = MODELS_DIR / f"{name}_fp16.onnx"
        int8_dst = MODELS_DIR / f"{name}_int8.onnx"

        convert_fp16(src, fp16_dst)
        convert_int8(src, int8_dst)

        compress_zstd(fp16_dst)
        compress_zstd(int8_dst)

    print("\nDone! New model files:")
    for f in sorted(MODELS_DIR.glob("*_fp16*")) + sorted(MODELS_DIR.glob("*_int8*")):
        print(f"  {f.name}: {f.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
