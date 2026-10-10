#!/usr/bin/env bash
# Time the SD 1.5 Inpaint pipeline under its measurement switches.
# Needs the SD 1.5 Inpaint bundle installed (Model Store) and the app
# closed: each run uses as much RAM as one erase.
#
#   scripts/sd_bench.sh            # all variants
#   scripts/sd_bench.sh base gpu   # a subset
#
# Variants:
#   base    current behaviour (OpenVINO on its default device, sequential CFG)
#   batch2  UNet session pinned to batch 2, one UNet call per step
#   gpu     OpenVINO on the iGPU (PRUNR_SD_OV_DEVICE=GPU)
#   gpu2    iGPU and batch 2
#   keep    bundle kept loaded (only shows in the second run of a session)
set -u
cd "$(dirname "$0")/.."
variants=("$@")
[ ${#variants[@]} -eq 0 ] && variants=(base batch2 gpu gpu2)
run() {
  local name=$1; shift
  echo "== $name"
  env "$@" RUST_LOG=info cargo test -q -p prunr-core --test sd_inpaint_smoke sd_bench -- --ignored --nocapture 2>&1 \
    | grep -E 'SD: (tile done|inpaint done|session committed)|\[sd_bench\]|SKIP|panicked' \
    | sed -E 's/^.*(INFO|WARN) *//'
}
for v in "${variants[@]}"; do
  case $v in
    base)   run base ;;
    batch2) run batch2 PRUNR_SD_BATCH2=1 ;;
    gpu)    run gpu PRUNR_SD_OV_DEVICE=GPU ;;
    gpu2)   run gpu2 PRUNR_SD_OV_DEVICE=GPU PRUNR_SD_BATCH2=1 ;;
    keep)   run keep PRUNR_SD_KEEP_LOADED=1 ;;
    *) echo "unknown variant: $v" ;;
  esac
done
