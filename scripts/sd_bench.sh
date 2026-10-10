#!/usr/bin/env bash
# Time the SD 1.5 Inpaint pipeline under its measurement switches.
# Needs the SD 1.5 Inpaint bundle installed (Model Store) and the app
# closed: each run uses as much RAM as one erase.
#
#   scripts/sd_bench.sh            # all variants
#   scripts/sd_bench.sh base gpu   # a subset
#
# Variants:
#   base    the shipped plan (CPU plugin, one 512×768 crop for the region)
#   tiles   tall crop off: two 512² tiles (PRUNR_SD_TALL_CROP=0)
#   gpu     OpenVINO on the iGPU (PRUNR_SD_OV_DEVICE=GPU)
#   keep    bundle kept loaded (session build shows only on the first run)
set -u
cd "$(dirname "$0")/.."
variants=("$@")
[ ${#variants[@]} -eq 0 ] && variants=(base tiles)
run() {
  local name=$1; shift
  echo "== $name"
  env "$@" RUST_LOG=info cargo test -q -p prunr-core --test sd_inpaint_smoke sd_bench -- --ignored --nocapture 2>&1 \
    | grep -E -A1 'SD: (tile done|inpaint done|session committed)|\[sd_bench\]|SKIP|panicked|WARN' \
    | sed -E 's/^.*(INFO|WARN) *//'
}
for v in "${variants[@]}"; do
  case $v in
    base)   run base ;;
    tiles)  run tiles PRUNR_SD_TALL_CROP=0 ;;
    gpu)    run gpu PRUNR_SD_OV_DEVICE=GPU ;;
    keep)   run keep PRUNR_SD_KEEP_LOADED=1 ;;
    *) echo "unknown variant: $v" ;;
  esac
done
