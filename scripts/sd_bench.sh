#!/usr/bin/env bash
# Time the SD 1.5 Inpaint pipeline under its measurement switches.
# Needs the SD 1.5 Inpaint bundle installed (Model Store) and the app
# closed: each run uses as much RAM as one erase.
#
#   scripts/sd_bench.sh            # all variants
#   scripts/sd_bench.sh base gpu   # a subset
#
# Variants:
#   base    current behaviour (OpenVINO on its default device, the CPU)
#   gpu     OpenVINO on the iGPU (PRUNR_SD_OV_DEVICE=GPU)
#   keep    bundle kept loaded (session build shows only on the first run)
#   dyn     OpenVINO with dynamic shapes (PRUNR_SD_OV_DYNAMIC=1)
#   t4      OpenVINO CPU plugin on 4 threads (PRUNR_SD_OV_THREADS=4)
#   tall    dynamic shapes and one 512×768 crop instead of two tiles
#   opt0    ONNX Runtime graph optimiser off (PRUNR_SD_ORT_OPT=0)
#   opt0tall  opt0 with the tall crop
set -u
cd "$(dirname "$0")/.."
variants=("$@")
[ ${#variants[@]} -eq 0 ] && variants=(base gpu)
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
    gpu)    run gpu PRUNR_SD_OV_DEVICE=GPU ;;
    keep)   run keep PRUNR_SD_KEEP_LOADED=1 ;;
    dyn)    run dyn PRUNR_SD_OV_DYNAMIC=1 ;;
    t4)     run t4 PRUNR_SD_OV_THREADS=4 ;;
    tall)   run tall PRUNR_SD_OV_DYNAMIC=1 PRUNR_SD_TALL_CROP=1 ;;
    opt0)   run opt0 PRUNR_SD_ORT_OPT=0 ;;
    opt0tall) run opt0tall PRUNR_SD_ORT_OPT=0 PRUNR_SD_OV_DYNAMIC=1 PRUNR_SD_TALL_CROP=1 ;;
    *) echo "unknown variant: $v" ;;
  esac
done
