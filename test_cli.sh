#!/usr/bin/env bash
set -euo pipefail

# BgPrunR CLI test script
# Tests all CLI usage patterns against real images

BGPRUNR="cargo run -p bgprunr-app --features dev-models --"
TEST_DIR="$(cd "$(dirname "$0")" && pwd)/test_cli"
OUT_DIR="$TEST_DIR/output"
PASS=0
FAIL=0

red()   { printf "\033[31m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
bold()  { printf "\033[1m%s\033[0m\n" "$*"; }

check() {
    local desc="$1" file="$2"
    if [[ -f "$file" ]] && [[ -s "$file" ]]; then
        green "  PASS: $desc  ($file)"
        PASS=$((PASS + 1))
    else
        red "  FAIL: $desc  (expected $file)"
        FAIL=$((FAIL + 1))
    fi
}

check_not_exists() {
    local desc="$1" file="$2"
    if [[ ! -f "$file" ]]; then
        green "  PASS: $desc  ($file does not exist)"
        PASS=$((PASS + 1))
    else
        red "  FAIL: $desc  ($file should not exist)"
        FAIL=$((FAIL + 1))
    fi
}

# Clean up previous test outputs
rm -rf "$OUT_DIR"
rm -f "$TEST_DIR"/*_nobg.png "$TEST_DIR"/result.png
mkdir -p "$OUT_DIR"

bold "=== BgPrunR CLI Tests ==="
echo "Test images: $TEST_DIR"
echo "Output dir:  $OUT_DIR"
echo ""

# ── Test 1: Show help ────────────────────────────────────────────────
bold "Test 1: --help"
$BGPRUNR --help > /dev/null 2>&1
green "  PASS: --help exits cleanly"
PASS=$((PASS + 1))

# ── Test 2: Show version ─────────────────────────────────────────────
bold "Test 2: --version"
$BGPRUNR --version > /dev/null 2>&1
green "  PASS: --version exits cleanly"
PASS=$((PASS + 1))

# ── Test 3: Single image (shorthand, no flags) ───────────────────────
bold "Test 3: bgprunr photo.jpg (shorthand, default output)"
$BGPRUNR "$TEST_DIR/Firefox.png" -f -q
check "single image default output" "$TEST_DIR/Firefox_nobg.png"
rm -f "$TEST_DIR/Firefox_nobg.png"

# ── Test 4: Single image with -o ──────────────────────────────────────
bold "Test 4: bgprunr photo.jpg -o result.png"
$BGPRUNR "$TEST_DIR/Firefox.png" -o "$TEST_DIR/result.png" -f -q
check "single image with -o" "$TEST_DIR/result.png"
rm -f "$TEST_DIR/result.png"

# ── Test 5: Single image with u2net model ─────────────────────────────
bold "Test 5: bgprunr -m u2net photo.jpg"
$BGPRUNR -m u2net "$TEST_DIR/Firefox.png" -o "$OUT_DIR/firefox_u2net.png" -f -q
check "u2net model" "$OUT_DIR/firefox_u2net.png"

# ── Test 6: Batch to output directory ─────────────────────────────────
bold "Test 6: bgprunr *.jpg -o output/ (batch to folder)"
$BGPRUNR "$TEST_DIR/Bolli_Khajeh.jpg" "$TEST_DIR/20251120_013152.jpg" -o "$OUT_DIR" -f -q
check "batch output 1" "$OUT_DIR/Bolli_Khajeh_nobg.png"
check "batch output 2" "$OUT_DIR/20251120_013152_nobg.png"

# ── Test 7: Batch with parallel jobs ──────────────────────────────────
bold "Test 7: bgprunr -j 2 *.jpg -o output/ (parallel)"
rm -f "$OUT_DIR/Bolli_Khajeh_nobg.png" "$OUT_DIR/20251120_013152_nobg.png"
$BGPRUNR -j 2 "$TEST_DIR/Bolli_Khajeh.jpg" "$TEST_DIR/20251120_013152.jpg" -o "$OUT_DIR" -f -q
check "parallel batch 1" "$OUT_DIR/Bolli_Khajeh_nobg.png"
check "parallel batch 2" "$OUT_DIR/20251120_013152_nobg.png"

# ── Test 8: Backward compat — `remove` subcommand ────────────────────
bold "Test 8: bgprunr remove photo.jpg (backward compat)"
$BGPRUNR remove "$TEST_DIR/Firefox.png" -o "$OUT_DIR/firefox_compat.png" -f -q
check "remove subcommand" "$OUT_DIR/firefox_compat.png"

# ── Test 9: No overwrite without --force ──────────────────────────────
bold "Test 9: no overwrite without --force"
# Create a dummy file that would be overwritten
touch "$OUT_DIR/Firefox_nobg.png"
$BGPRUNR "$TEST_DIR/Firefox.png" -o "$OUT_DIR/Firefox_nobg.png" -q 2>/dev/null || true
# File should still be empty (0 bytes = the touch'd dummy)
if [[ ! -s "$OUT_DIR/Firefox_nobg.png" ]]; then
    green "  PASS: file not overwritten without --force"
    PASS=$((PASS + 1))
else
    red "  FAIL: file was overwritten without --force"
    FAIL=$((FAIL + 1))
fi

# ── Test 10: No-args launches GUI (skip in automated tests) ───────────
bold "Test 10: no args → GUI (skipped in automated test)"
green "  SKIP: no-args launches GUI, cannot test non-interactively"

# ── Test 11: Mix of all 3 images ──────────────────────────────────────
bold "Test 11: all 3 images batch"
rm -f "$OUT_DIR"/*_nobg.png
$BGPRUNR "$TEST_DIR"/*.{jpg,png} -o "$OUT_DIR" -f -q 2>/dev/null || true
check "3-image batch 1" "$OUT_DIR/Bolli_Khajeh_nobg.png"
check "3-image batch 2" "$OUT_DIR/20251120_013152_nobg.png"
check "3-image batch 3" "$OUT_DIR/Firefox_nobg.png"

# ── Summary ───────────────────────────────────────────────────────────
echo ""
bold "=== Results ==="
green "$PASS passed"
if [[ $FAIL -gt 0 ]]; then
    red "$FAIL failed"
    exit 1
else
    green "All tests passed!"
    exit 0
fi
