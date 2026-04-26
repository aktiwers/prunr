#!/usr/bin/env python3
"""Patch Big-LaMa ONNX to fix the inverse-DFT shape-inference error.

advimman/lama's FFC ResNet uses FFT/IFFT for spectral convolutions.
torch.onnx exports both the forward FFT (`inverse=0, onesided=1`) and
the inverse FFT — but it inherits `onesided=1` on the inverse, which
ONNX rejects:

    Node (node_DFT_NNN) Op (DFT) [ShapeInferenceError]
    is_onesided and inverse attributes cannot be enabled at the same time

Per the ONNX DFT op spec, the inverse should be `inverse=1, onesided=0`.
This script walks the graph, finds DFT nodes with `inverse=1`, and
clears `onesided` on each.

Usage:
    python scripts/patch_big_lama_dft.py <path-to-big_lama.onnx>

In-place patch. Writes a `<path>.bak` backup the first time it runs.
Idempotent — re-running on a patched file is a no-op.

DEPENDENCIES:
    pip install onnx
"""
import shutil
import sys
from pathlib import Path


def patch(path: Path) -> int:
    import onnx

    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        print(f"Backing up {path} → {backup}")
        shutil.copy2(path, backup)

    print(f"Loading {path}")
    model = onnx.load(str(path))

    fixed = 0
    for node in model.graph.node:
        if node.op_type != "DFT":
            continue
        attrs = {a.name: a for a in node.attribute}
        inverse = attrs.get("inverse")
        onesided = attrs.get("onesided")
        if inverse is None or inverse.i != 1:
            continue
        if onesided is None or onesided.i != 1:
            continue
        # The illegal combination — clear `onesided` on the inverse node.
        onesided.i = 0
        fixed += 1
        print(f"  Patched {node.name}: inverse=1, onesided 1 → 0")

    if fixed == 0:
        print("No illegal DFT nodes found — file already patched.")
        return 0

    print(f"Saving patched ONNX to {path} ({fixed} node(s) fixed)")
    onnx.save(model, str(path))
    return fixed


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python patch_big_lama_dft.py <path-to-big_lama.onnx>", file=sys.stderr)
        sys.exit(2)
    path = Path(sys.argv[1]).resolve()
    if not path.is_file():
        print(f"Not a file: {path}", file=sys.stderr)
        sys.exit(2)
    fixed = patch(path)
    sys.exit(0 if fixed >= 0 else 1)


if __name__ == "__main__":
    main()
