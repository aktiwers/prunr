---
phase: 02-core-inference-engine
plan: "05"
subsystem: testing

tags: [rembg, python, reference-masks, pixel-accuracy, u2net]

requires:
  - phase: 02-04
    provides: batch processing API — this plan adds test infrastructure for validating it

provides:
  - scripts/generate_references.py — runnable script that creates u2net alpha mask PNGs using exact rembg defaults
  - tests/references/ — git-tracked directory for committed ground truth masks
  - tests/test_images/README.md — instructions for fetching rembg test images on a fresh clone

affects:
  - 02-06 (reference test plan — depends on masks committed by running this script)

tech-stack:
  added: []
  patterns:
    - "Reference mask generation: rembg Python → alpha channel PNG saved to tests/references/{stem}_u2net_mask.png"
    - "Version pinning: VERSIONS.txt written next to masks to capture rembg version at generation time"

key-files:
  created:
    - scripts/generate_references.py
    - tests/references/.gitkeep
    - tests/test_images/README.md
  modified:
    - .gitignore

key-decisions:
  - "Test images (car-1.jpg, car-2.jpg, car-3.jpg) are not committed — excluded via .gitignore because they are copyrighted rembg assets fetched on demand"
  - "Reference masks (PNG alpha channels) ARE committed as ground truth for the Rust CORE-05 pixel-accuracy test"
  - "generate_references.py explicitly locks: u2net model, alpha_matting=False, post_process_mask=False — these must not be changed"

patterns-established:
  - "Reference generation: run script once, commit outputs, Rust test reads committed files as ground truth"
  - "Version traceability: VERSIONS.txt written alongside masks so regeneration with a different rembg version is detectable"

requirements-completed:
  - CORE-05

duration: 2min
completed: 2026-04-06
---

# Phase 02 Plan 05: Reference Generation Infrastructure Summary

**Python script generating u2net alpha mask PNGs from rembg with locked defaults (alpha_matting=False, post_process_mask=False) as committed ground truth for CORE-05 pixel-accuracy test**

## Performance

- **Duration:** 2 min
- **Started:** 2026-04-06T23:30:52Z
- **Completed:** 2026-04-06T23:32:04Z
- **Tasks:** 1
- **Files modified:** 4

## Accomplishments

- Created scripts/generate_references.py with self-documenting rembg settings, version recording, and clear error messages guiding users to download images
- Tracked tests/references/ directory via .gitkeep so the Rust test can always find the directory on a fresh clone
- Provided complete setup instructions in tests/test_images/README.md for fetching rembg test images and generating reference masks
- Updated .gitignore to exclude test image source files while keeping generated masks committable

## Task Commits

Each task was committed atomically:

1. **Task 1: Create scripts/generate_references.py and test image infrastructure** - `8eb552d` (chore)

**Plan metadata:** (see final commit)

## Files Created/Modified

- `scripts/generate_references.py` - Generates alpha mask PNGs using rembg defaults; records version to VERSIONS.txt; exits with instructions if images missing
- `tests/references/.gitkeep` - Empty file ensuring directory exists in git for Rust reference test
- `tests/test_images/README.md` - curl commands for fetching car-1.jpg, car-2.jpg, car-3.jpg from rembg repo; explains why images are not committed
- `.gitignore` - Added exclusion patterns for test image source files (jpg/jpeg/png/webp/bmp in tests/test_images/)

## Decisions Made

- Test images (car-1.jpg, car-2.jpg, car-3.jpg) are NOT committed — they are copyrighted rembg assets; .gitignore excludes them. Only the generated masks (tests/references/) are committed as ground truth.
- The script locks `alpha_matting=False` and `post_process_mask=False` explicitly, matching rembg CLI defaults exactly, so the Rust comparison is valid.
- VERSIONS.txt is written by the script to record which rembg version produced the current masks, making version drift detectable if references need regeneration.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

**Manual step required before Plan 06 (reference test) can pass:**

1. Download rembg test images:
   ```bash
   cd tests/test_images
   curl -LO https://github.com/danielgatis/rembg/raw/main/tests/car-1.jpg
   curl -LO https://github.com/danielgatis/rembg/raw/main/tests/car-2.jpg
   curl -LO https://github.com/danielgatis/rembg/raw/main/tests/car-3.jpg
   ```

2. Run the reference generator (requires Python + rembg):
   ```bash
   pip install rembg
   python scripts/generate_references.py
   ```

3. Commit the generated masks:
   ```bash
   git add tests/references/
   git commit -m "chore: add rembg reference masks for CORE-05 pixel-accuracy test"
   ```

## Next Phase Readiness

- Plan 06 (reference test in Rust) can be authored — it will read from tests/references/ and compare against prunr-core output
- The CORE-05 hard gate cannot be cleared until the user runs the steps above and commits the masks
- Once masks are committed, `cargo test -p prunr-core --features dev-models test_rembg_reference` becomes runnable

---
*Phase: 02-core-inference-engine*
*Completed: 2026-04-06*
