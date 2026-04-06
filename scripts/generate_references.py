#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate rembg reference masks for BgPrunR pixel-accuracy testing.

Usage:
    pip install rembg[gpu]   # or: pip install rembg
    python scripts/generate_references.py

Output:
    tests/references/{stem}_u2net_mask.png  — one per input image in tests/test_images/

Settings used (MUST NOT be changed — these are the exact rembg defaults for comparison):
    model_name: "u2net"
    alpha_matting: False
    alpha_matting_foreground_threshold: 240  (unused, alpha_matting=False)
    alpha_matting_background_threshold: 10   (unused, alpha_matting=False)
    alpha_matting_erode_size: 10             (unused, alpha_matting=False)
    post_process_mask: False

rembg version used when generating current references:
    See: rembg --version  (record in tests/references/VERSIONS.txt after first run)

CRITICAL: Re-running this script with a different rembg version may change reference outputs.
If references are regenerated, all images must be re-run together and VERSIONS.txt updated.
"""

import sys
import os
from pathlib import Path


def check_dependencies():
    try:
        import rembg
        return rembg
    except ImportError:
        print("ERROR: rembg not installed. Run: pip install rembg", file=sys.stderr)
        sys.exit(1)


def generate_references():
    rembg = check_dependencies()
    from rembg import remove
    from rembg.session_factory import new_session
    from PIL import Image

    repo_root = Path(__file__).parent.parent
    input_dir = repo_root / "tests" / "test_images"
    output_dir = repo_root / "tests" / "references"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Use exact rembg defaults — DO NOT change these settings
    # alpha_matting=False, post_process_mask=False match rembg CLI defaults
    session = new_session("u2net")

    supported_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    input_files = sorted([
        f for f in input_dir.iterdir()
        if f.suffix.lower() in supported_extensions
    ])

    if not input_files:
        print(f"No images found in {input_dir}", file=sys.stderr)
        print("Download rembg test images from:", file=sys.stderr)
        print("  https://github.com/danielgatis/rembg/tree/main/tests/", file=sys.stderr)
        print("Save them to tests/test_images/", file=sys.stderr)
        sys.exit(1)

    print(f"Generating reference masks for {len(input_files)} image(s)...")

    for input_path in input_files:
        with open(input_path, "rb") as f:
            input_bytes = f.read()

        # rembg defaults: u2net, alpha_matting=False, post_process_mask=False
        output_bytes = remove(
            input_bytes,
            session=session,
            alpha_matting=False,
            post_process_mask=False,
        )

        # Extract alpha channel as the mask
        from io import BytesIO
        output_img = Image.open(BytesIO(output_bytes)).convert("RGBA")
        alpha_mask = output_img.split()[3]  # Extract alpha channel only

        output_name = f"{input_path.stem}_u2net_mask.png"
        output_path = output_dir / output_name
        alpha_mask.save(output_path)
        print(f"  {input_path.name} → {output_name}")

    # Record rembg version for reproducibility
    import subprocess
    try:
        version = subprocess.check_output(
            [sys.executable, "-m", "pip", "show", "rembg"],
            text=True
        )
        versions_path = output_dir / "VERSIONS.txt"
        with open(versions_path, "w") as f:
            f.write("# rembg version used to generate these references\n")
            f.write("# Re-run generate_references.py if rembg is updated\n\n")
            f.write(version)
        print(f"\nRecorded rembg version to {versions_path}")
    except Exception as e:
        print(f"Warning: Could not record rembg version: {e}", file=sys.stderr)

    print(f"\nDone. Reference masks saved to {output_dir}/")
    print("Next: commit tests/references/ to git, then run:")
    print("  cargo test -p bgprunr-core --features dev-models test_rembg_reference")


if __name__ == "__main__":
    generate_references()
