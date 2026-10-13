#!/usr/bin/env python3
"""
Re-save an ONNX model's external weights across several data files, each
under a size cap, so every file fits a GitHub release asset (2 GiB limit).
The graph references each tensor's own file; onnxruntime loads them from
the model's directory with nothing to reassemble.

Usage:
    python scripts/split_onnx_data.py <in.onnx> <out dir> [cap_mb=1900]
Writes <out dir>/<name>.onnx and <name>_0.data, <name>_1.data, ...
"""

import hashlib
import sys
from pathlib import Path


def main() -> int:
    import onnx
    from onnx import TensorProto

    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    cap = int(sys.argv[3] if len(sys.argv) > 3 else 1900) * 1024 * 1024
    out.mkdir(parents=True, exist_ok=True)
    stem = src.stem
    m = onnx.load(str(src), load_external_data=True)

    files, sizes, idx = [], [], -1
    for t in m.graph.initializer:
        raw = t.raw_data
        if len(raw) < 1024:
            continue
        if idx < 0 or sizes[idx] + len(raw) > cap:
            idx += 1
            name = f"{stem}_{idx}.data"
            files.append(open(out / name, "wb"))
            sizes.append(0)
        f = files[idx]
        offset = sizes[idx]
        f.write(raw)
        sizes[idx] += len(raw)
        t.ClearField("raw_data")
        t.data_location = TensorProto.EXTERNAL
        del t.external_data[:]
        for k, v in (("location", f"{stem}_{idx}.data"), ("offset", str(offset)), ("length", str(len(raw)))):
            e = t.external_data.add()
            e.key, e.value = k, v
    for f in files:
        f.close()
    onnx.save_model(m, str(out / f"{stem}.onnx"))

    print("=== SHA256 ===")
    for p in [out / f"{stem}.onnx", *(out / f"{stem}_{i}.data" for i in range(len(files)))]:
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        print(f"  {p.name}  size={p.stat().st_size}  sha256={h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
