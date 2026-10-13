//! End-to-end smoke test for SD 1.5 Inpaint. Auto-skips when the bundle
//! isn't installed so this won't fail CI on a fresh checkout. Once the
//! 4-part bundle finishes downloading, run with:
//!
//!   cargo test -p prunr-core --test sd_inpaint_smoke -- --nocapture
//!
//! Validates that:
//!   - The session bundle loads (4 ONNX files in the multi-part dir)
//!   - One full denoising loop runs without panicking
//!   - The painted region differs materially from the source
//!     (i.e. the pipeline is doing real work, not returning input)

mod test_common;

use image::{GrayImage, Luma, Rgba, RgbaImage};
use prunr_core::inpaint_blend;
use prunr_core::inpaint_sd::{self, SdInpaintRequest};
use test_common::skip_if_no_ort;

#[test]
fn sd_inpaint_modifies_painted_region() {
    let _ = tracing_subscriber::fmt::try_init();
    if skip_if_no_ort("sd_inpaint_smoke") { return; }
    let id = prunr_models::ModelId::SdV15InpaintFp16;
    if !prunr_models::is_available(id) {
        eprintln!("SKIP: SD 1.5 Inpaint bundle not installed at {:?}",
            prunr_models::on_demand_dir());
        return;
    }

    // 512×512 solid blue. Mask covers the centre 128×128 region.
    let w: u32 = 512;
    let h: u32 = 512;
    let mut image = RgbaImage::new(w, h);
    for p in image.pixels_mut() { *p = Rgba([40, 80, 200, 255]); }
    let mut mask = GrayImage::new(w, h);
    for y in 192..320 {
        for x in 192..320 {
            mask.put_pixel(x, y, Luma([255]));
        }
    }

    let req = SdInpaintRequest {
        num_inference_steps: 20,
        seed: Some(42),
        ..Default::default()
    };
    let progress = prunr_core::Progress::none();
    let result = match inpaint_sd::process_inpaint_with(&image, &mask, id, req, &progress) {
        Ok(r) => r.rgba,
        // The pre-flight RAM gate is an environmental refusal (SD wants
        // ~12 GB free), not a pipeline failure — skip like a missing bundle.
        Err(e) if e.to_string().contains("RAM free") => {
            eprintln!("[sd_inpaint_smoke] SKIP: {e}");
            return;
        }
        Err(e) => panic!("SD inpaint should succeed: {e}"),
    };

    assert_eq!(result.dimensions(), image.dimensions());

    let mut diffs = 0u32;
    for y in 192..320 {
        for x in 192..320 {
            let src = image.get_pixel(x, y);
            let dst = result.get_pixel(x, y);
            if src.0 != dst.0 {
                diffs += 1;
            }
        }
    }
    let total = 128 * 128;
    eprintln!("SD inpaint: {diffs}/{total} masked pixels differ from source");
    eprintln!("Sample painted pixel at (256,256): {:?} → {:?}",
        image.get_pixel(256, 256), result.get_pixel(256, 256));
    assert!(diffs > total / 2,
        "SD returned ≤ half-changed for {} of {} masked pixels", total - diffs, total);

    // Outside the mask: source must be preserved exactly (no scaling
    // drift from the VAE round-trip leaking out of the masked region).
    for (x, y) in [(0, 0), (10, 10), (500, 500), (100, 400)] {
        assert_eq!(
            image.get_pixel(x, y).0,
            result.get_pixel(x, y).0,
            "unmasked pixel at ({x},{y}) drifted",
        );
    }
}

/// Timing run for scripts/sd_bench.sh: the shape of a real erase (a
/// 328×607 painted region inside a larger photo, so it tiles today),
/// guided, 20 steps, fixed seed. Ignored by default; prints the stage
/// timings the pipeline logs at info level.
#[test]
#[ignore]
fn sd_bench() {
    let _ = tracing_subscriber::fmt::try_init();
    if skip_if_no_ort("sd_bench") { return; }
    // `PRUNR_SD_BENCH_MODEL=lcm` runs the LCM bundle and `=rorem` the RORem
    // eraser, each at the app's defaults for it, instead of SD 1.5.
    let (id, steps, guidance, scheduler) = match std::env::var("PRUNR_SD_BENCH_MODEL").as_deref() {
        Ok("lcm") => (prunr_models::ModelId::SdV15LcmInpaintFp16, 8, 1.5, inpaint_sd::SchedulerKind::Lcm),
        Ok("rorem") => (prunr_models::ModelId::RoremFp16, inpaint_sd::ROREM_STEPS, 1.0, inpaint_sd::SchedulerKind::Lcm),
        _ => (prunr_models::ModelId::SdV15InpaintFp16, 25, 4.0, inpaint_sd::SchedulerKind::DpmPp2MKarras),
    };
    if !prunr_models::is_available(id) {
        eprintln!("SKIP: {id:?} bundle not installed");
        return;
    }
    // `PRUNR_SD_BENCH_IMAGE=a.png,b.png PRUNR_SD_BENCH_MASK=a_mask.png,b_mask.png`
    // run real photos with their painted regions (grown and blended as
    // the app does) instead of the synthetic pair.
    let list = |name: &str| -> Vec<String> {
        std::env::var(name).map(|v| v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()).unwrap_or_default()
    };
    let (photos, painted) = (list("PRUNR_SD_BENCH_IMAGE"), list("PRUNR_SD_BENCH_MASK"));
    assert_eq!(photos.len(), painted.len(), "one PRUNR_SD_BENCH_MASK per PRUNR_SD_BENCH_IMAGE");
    let mut pairs: Vec<(String, RgbaImage, GrayImage)> = photos.iter().zip(&painted).map(|(photo, painted)| {
        let image = image::open(photo).expect("bench photo").into_rgba8();
        let painted = image::open(painted).expect("bench painted region").into_luma8();
        let mask = inpaint_blend::fill_region(&painted, inpaint_blend::DEFAULT_GROW_PX, inpaint_blend::DEFAULT_EDGE_BLEND_PX);
        let stem = std::path::Path::new(photo).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        (stem, image, mask)
    }).collect();
    if pairs.is_empty() {
        let (w, h) = (1200u32, 900u32);
        let image = RgbaImage::from_fn(w, h, |x, y| {
            Rgba([(x / 5 % 256) as u8, (y / 4 % 256) as u8, ((x + y) / 9 % 256) as u8, 255])
        });
        // `PRUNR_SD_BENCH_REGION=WxH` sizes the painted region (default 328×607).
        let (rw, rh) = std::env::var("PRUNR_SD_BENCH_REGION").ok()
            .and_then(|v| { let (a, b) = v.split_once('x')?; Some((a.parse::<u32>().ok()?, b.parse::<u32>().ok()?)) })
            .unwrap_or((328, 607));
        let mut mask = GrayImage::new(w, h);
        let (x0, y0) = ((w - rw) / 2, (h - rh) / 2);
        for y in y0..y0 + rh {
            for x in x0..x0 + rw {
                mask.put_pixel(x, y, Luma([255]));
            }
        }
        pairs.push(("synthetic".into(), image, mask));
    }
    // `PRUNR_SD_BENCH_STEPS=2,4,8`, `PRUNR_SD_BENCH_SEED=7,8` and
    // `PRUNR_SD_BENCH_FILL=auto,full` run every combination; each
    // defaults to one value.
    let nums = |name: &str| -> Vec<u64> { list(name).iter().map(|v| v.parse().expect(name)).collect() };
    let steps_list = { let v = nums("PRUNR_SD_BENCH_STEPS"); if v.is_empty() { vec![steps as u64] } else { v } };
    let seeds = { let v = nums("PRUNR_SD_BENCH_SEED"); if v.is_empty() { vec![42] } else { v } };
    let fills: Vec<prunr_core::inpaint::FillScale> = {
        let v: Vec<_> = list("PRUNR_SD_BENCH_FILL").iter().map(|v| v.parse().expect("PRUNR_SD_BENCH_FILL")).collect();
        if v.is_empty() { vec![Default::default()] } else { v }
    };
    // The shipped plan, with the same overrides the app honours.
    let flag = |name: &str| match std::env::var(name).ok()?.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    };
    let planned = inpaint_sd::plan_tuning(false, inpaint_sd::SD_DEFAULT_MARGIN_MB);
    let tuning = inpaint_sd::SdTuning {
        keep_loaded: flag("PRUNR_SD_KEEP_LOADED").unwrap_or(planned.keep_loaded),
        tall_crop: flag("PRUNR_SD_TALL_CROP").unwrap_or(planned.tall_crop),
        ov_device: std::env::var("PRUNR_SD_OV_DEVICE").ok().filter(|s| !s.is_empty()),
        // `PRUNR_SD_MEM_PATTERN=0`, `PRUNR_SD_DEV_INIT=1`, `PRUNR_SD_THREADS=n`
        // try the session memory knobs.
        mem: inpaint_sd::SdMemPlan {
            no_memory_pattern: flag("PRUNR_SD_MEM_PATTERN").map_or(planned.mem.no_memory_pattern, |on| !on),
            device_initializers: flag("PRUNR_SD_DEV_INIT").unwrap_or(planned.mem.device_initializers),
            intra_threads: std::env::var("PRUNR_SD_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(planned.mem.intra_threads),
        },
        ..planned
    };
    // `PRUNR_SD_BENCH_OUT=dir` saves each fill blended into its photo.
    let out_dir = std::env::var("PRUNR_SD_BENCH_OUT").ok();
    let progress = prunr_core::Progress::none();
    for (stem, image, mask) in &pairs {
        for &steps in &steps_list {
            for &seed in &seeds {
                for &fill_scale in &fills {
                    let req = SdInpaintRequest {
                        prompt: "clean background".to_string(),
                        num_inference_steps: steps as u32,
                        guidance_scale: guidance,
                        scheduler,
                        seed: Some(seed),
                        fill_scale,
                        tuning: tuning.clone(),
                        ..Default::default()
                    };
                    let started = std::time::Instant::now();
                    let result = match inpaint_sd::process_inpaint_with(image, mask, id, req, &progress) {
                        Ok(r) => r.rgba,
                        Err(e) if e.to_string().contains("RAM free") => {
                            eprintln!("[sd_bench] SKIP: {e}");
                            return;
                        }
                        Err(e) => panic!("SD inpaint should succeed: {e}"),
                    };
                    assert_eq!(result.dimensions(), image.dimensions());
                    eprintln!("[sd_bench] {stem} steps={steps} seed={seed} fill={fill_scale} total {} ms", started.elapsed().as_millis());
                    if let Some(dir) = &out_dir {
                        let done = inpaint_blend::finalize_inpaint(&result, image, mask, inpaint_blend::DEFAULT_EDGE_BLEND_PX, 0.0);
                        done.save(format!("{dir}/{stem}_s{steps}_seed{seed}_{fill_scale}.png")).expect("save the bench output");
                    }
                }
            }
        }
    }
}
