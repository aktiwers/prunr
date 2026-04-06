# Test Images

These images are used for the CORE-05 pixel-accuracy reference test.

## Required Images

Download the following images from the rembg repository:

1. `car-1.jpg` — https://github.com/danielgatis/rembg/raw/main/tests/car-1.jpg
2. `car-2.jpg` — https://github.com/danielgatis/rembg/raw/main/tests/car-2.jpg
3. `car-3.jpg` — https://github.com/danielgatis/rembg/raw/main/tests/car-3.jpg

Or download all at once:

```bash
cd tests/test_images
curl -LO https://github.com/danielgatis/rembg/raw/main/tests/car-1.jpg
curl -LO https://github.com/danielgatis/rembg/raw/main/tests/car-2.jpg
curl -LO https://github.com/danielgatis/rembg/raw/main/tests/car-3.jpg
```

These images are **NOT committed to git** (they are in .gitignore) because they are
copyrighted assets from the rembg repository. They are fetched on demand for testing.

## Generating Reference Masks

After downloading the test images, generate reference masks:

```bash
pip install rembg
python scripts/generate_references.py
```

The script saves masks to `tests/references/`. **Commit the generated masks** — they are the
ground truth for the CORE-05 pixel-accuracy test.

## Regenerating References

If rembg is updated to a new version, re-run `generate_references.py` on the same images and
commit the new masks. The Rust reference test will use whatever is committed as ground truth.
