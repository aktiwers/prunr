---
phase: 32
plan: "13"
slug: community-upscale-models
status: draft
created: 2026-05-16
revised: 2026-05-16
---

# Phase 32 Plan 13 — Preconditions: Community Upscale Models Export

## History — why this pair was chosen

The plan's original candidates were **4x-UltraSharp** (Kim2091) and
**4x_foolhardy_Remacri** (FoolhardyVEVO). License check against
openmodeldb confirmed BOTH are **CC-BY-NC-SA-4.0** — non-commercial,
not redistributable. Blocked. Replacement pair:

| Model | Author | License | Niche |
|-------|--------|---------|-------|
| **4x-NMKD-Siax-CX** | Nmkd | **WTFPL** | Clean photos / mild compression — UltraSharp role |
| **4x-NMKD-Superscale** | Nmkd | **WTFPL** | Noisy / heavily-compressed restoration — Remacri role |

Both ESRGAN/RRDB architecture (drop-in to `upscale_rgba_with_engine`).
WTFPL ("Do What The Fuck You Want To Public License") — most permissive
license possible. No attribution required (`attribution_required:
false`).

Out-of-codebase setup that MUST complete before Task 3 of Plan 32-13
(REGISTRY entries) can execute.

---

## Models

1. **4x-NMKD-Siax-CX** by `Nmkd` — https://openmodeldb.info/models/4x-NMKD-Siax-CX
   - "Universal upscaler for clean and slightly compressed images (JPEG quality 75 or better)"
   - Closest community-favourite analog to UltraSharp's clean-photo niche
2. **4x-NMKD-Superscale** by `Nmkd` — https://openmodeldb.info/models/4x-NMKD-Superscale
   - "Upscaling of realistic images/photos with noise and compression artifacts"
   - Closest community-favourite analog to Remacri's restoration/grain niche
   - Download (.pth, 63.9 MB): https://icedrive.net/1/43GNBihZyi

---

## Task P-1: 4x-NMKD-Siax-CX export

**Owner:** user (manual setup outside Claude's edit loop).

### License — already verified

- License: **WTFPL**
- Source page: https://openmodeldb.info/models/4x-NMKD-Siax-CX
- Redistribution: permitted. Commercial use: permitted. Attribution: not required.
- License URL: https://www.wtfpl.net/about/

No DECISION GATE — already passed.

### Step 1 — Download `.pth`

Working dir: `/tmp/realesrgan-export/` (existing venv).

```bash
cd /tmp/realesrgan-export
# Visit https://openmodeldb.info/models/4x-NMKD-Siax-CX and click the
# Download link. NMKD hosts on icedrive — direct curl-able URL is
# typically not exposed; manual browser download → save the .pth into
# /tmp/realesrgan-export/4x-NMKD-Siax-CX.pth
ls -la 4x-NMKD-Siax-CX.pth
```

### Step 2 — Export to ONNX

```bash
python export.py \
  --input ./4x-NMKD-Siax-CX.pth \
  --output ./4x-NMKD-Siax-CX.onnx \
  --opset 17 \
  --dynamic_axes 'data:{2:h,3:w}' \
  --scale 4
```

Note: NMKD models follow the same RRDBNet checkpoint shape as Real-ESRGAN
x4plus, so `export.py` should accept it as-is. If it complains about
missing keys (e.g. NMKD uses `params` instead of `params_ema`), add a
fallback to `export.py` and document the patch in this file's Notes.

### Step 3 — Verify tensor names

```bash
python -c "import onnx; m = onnx.load('4x-NMKD-Siax-CX.onnx'); print('input:', [i.name for i in m.graph.input], 'output:', [o.name for o in m.graph.output])"
```

Expect `input=["data"], output=["output"]`. If tensor names differ
(NMKD's training pipeline sometimes renames them), capture them — they
go into `UpscaleModelKnobs.input_name` directly.

### Step 4 — Compute sha256 + size

```bash
sha256sum 4x-NMKD-Siax-CX.onnx
stat -c '%s' 4x-NMKD-Siax-CX.onnx
```

Expect `~64 MB` (RRDB-23 = same param count as x4plus).

### Step 5 — Smoke test

Adapt `/tmp/realesrgan-export/smoke_x2plus.py` for single-pass scale=4
(Siax is native-4×, no two-pass chain):

```bash
# Copy and tweak:
cp smoke_x2plus.py smoke_siax.py
# Edit smoke_siax.py: model path → 4x-NMKD-Siax-CX.onnx, remove the
# second pass (or just run only pass 1 with scale=4 input dimensions).

/usr/bin/time -v python smoke_siax.py 2>&1 | grep -E "Maximum resident|Elapsed"
```

Capture peak RSS (MB).

### Step 6 — Upload to GitHub release

```bash
gh release upload models-v1 4x-NMKD-Siax-CX.onnx --repo aktiwers/prunr
curl -I "https://github.com/aktiwers/prunr/releases/download/models-v1/4x-NMKD-Siax-CX.onnx" | head -1
# Expect HTTP/2 302 → 200 after redirect.
```

### Step 7 — Capture values

- [x] URL: `https://github.com/aktiwers/prunr/releases/download/models-v1/4x-NMKD-Siax-CX.onnx`
- [x] License string: `WTFPL`
- [x] License URL: `https://www.wtfpl.net/about/`
- [x] Source URL: `https://openmodeldb.info/models/4x-NMKD-Siax-CX`
- [x] Author handle: `Nmkd`
- [x] `.pth` mirror used: `https://huggingface.co/gemasai/4x_NMKD-Siax_200k/resolve/main/4x_NMKD-Siax_200k.pth` (canonical NMKD model; "CX" / "_200k" tag both refer to the same checkpoint)
- [x] sha256: `d7db322397ae1041076a3f9fe9736a7bf9d3364e435eb5b0ece912e050a29f87`
- [x] Size (bytes): `67051621`
- [x] Size (MB, rounded up): `64`
- [x] Input tensor name: `data`
- [x] Output tensor name: `output`
- [x] Peak RSS during smoke test (MB, Python single-shot 256→1024): `1152`
- [x] Recommended `working_set_mb`: `600` (mirrors x4plus — same RRDB-23 arch, same param count = 16,697,987; Rust tiled dispatch reuses buffers so peak is much lower than Python single-shot)
- [x] `attribution_required`: `false` (WTFPL does not require attribution)

---

## Task P-2: 4x-NMKD-Superscale export

**Owner:** user (manual setup outside Claude's edit loop).

### License — already verified

- License: **WTFPL**
- Source page: https://openmodeldb.info/models/4x-NMKD-Superscale
- Redistribution: permitted. Commercial use: permitted. Attribution: not required.
- License URL: https://www.wtfpl.net/about/
- `.pth` download (icedrive): https://icedrive.net/1/43GNBihZyi (63.9 MB)

No DECISION GATE — already passed.

### Step 1 — Download `.pth`

Save the icedrive download into `/tmp/realesrgan-export/4x-NMKD-Superscale.pth`.

### Steps 2–6 — Mirror P-1

(Same export / smoke / upload sequence with `4x-NMKD-Superscale.pth` /
`.onnx`.)

### Step 7 — Capture values

- [x] URL: `https://github.com/aktiwers/prunr/releases/download/models-v1/4x-NMKD-Superscale.onnx`
- [x] License string: `WTFPL`
- [x] License URL: `https://www.wtfpl.net/about/`
- [x] Source URL: `https://openmodeldb.info/models/4x-NMKD-Superscale`
- [x] Author handle: `Nmkd`
- [x] `.pth` mirror used: `https://raw.githubusercontent.com/marduk191/nmkd-mirror/main/ESRGAN/Models/Realistic,%20Multipurpose/4x_NMKD-Superscale-SP_178000_G.pth` (the canonical SP_178000_G variant referenced as "NMKD-Superscale" community-wide)
- [x] sha256: `6b790a203b341d2db71735040f4f1b26c2bdc5440fdc850e217c2234cd996618`
- [x] Size (bytes): `67051621`
- [x] Size (MB, rounded up): `64`
- [x] Input tensor name: `data`
- [x] Output tensor name: `output`
- [x] Peak RSS during smoke test (MB, Python single-shot 256→1024): `1152`
- [x] Recommended `working_set_mb`: `600` (mirrors x4plus — same arch / param count; Rust tiled dispatch keeps RSS far below Python single-shot)
- [x] `attribution_required`: `false`

---

## Completion Gate

When both P-1 and P-2 are complete, fill in the captured values above
and signal the orchestrator with one of:

- `values captured` — proceed to Task 3 with both models
- `block siax-cx` / `block superscale` — drop that model; Task 3 ships
  the survivor only
- `block both` — close 32-13 as license-blocked again; we'll log a
  DEFERRED for user-import feature instead

---

## Notes from drafting

- Both models are RRDB-23 (same arch as x4plus). REGISTRY entries
  mirror x4plus exactly except for the `license` / `attribution_required`
  fields (WTFPL, false) and the model-specific names / URL / sha256.
- The export script `export.py` in `/tmp/realesrgan-export/` was written
  for xinntao's `params_ema` checkpoint key. NMKD's models may use
  `params` (no `_ema`). If so, the exporter needs a fallback — document
  the diff here.
- `is_user_visible()` returns `true` for both — they're full citizens
  of the Model Store, unlike the hidden x2plus.
