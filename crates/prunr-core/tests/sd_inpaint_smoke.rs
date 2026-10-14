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

/// `PRUNR_SD_BENCH_MODEL=lcm` runs the LCM bundle and `=rorem` the RORem
/// eraser, each at the app's defaults for it, instead of SD 1.5:
/// (model, steps, guidance, scheduler).
fn bench_model() -> (prunr_models::ModelId, u32, f32, inpaint_sd::SchedulerKind) {
    match std::env::var("PRUNR_SD_BENCH_MODEL").as_deref() {
        Ok("lcm") => (prunr_models::ModelId::SdV15LcmInpaintFp16, 8, 1.5, inpaint_sd::SchedulerKind::Lcm),
        Ok("rorem") => (prunr_models::ModelId::RoremFp16, inpaint_sd::ROREM_STEPS, 1.0, inpaint_sd::SchedulerKind::Lcm),
        _ => (prunr_models::ModelId::SdV15InpaintFp16, 25, 4.0, inpaint_sd::SchedulerKind::DpmPp2MKarras),
    }
}

/// Step previews on the real model: one per denoising step, placed on
/// the photo's preview over the erased region, and the fill the same
/// bytes as a run without them. Ignored by default (it runs the model
/// twice); the model is picked as for `sd_bench`, `PRUNR_SD_BENCH_STEPS`
/// sets the step count, and `PRUNR_SD_PREVIEW_OUT=dir` saves each step's
/// preview pasted on the photo's, and the fill, as PNGs.
#[test]
#[ignore]
fn sd_step_previews() {
    use prunr_core::progress::{check_contract, ProgressUpdate, RecordingSink};
    let _ = tracing_subscriber::fmt::try_init();
    if skip_if_no_ort("sd_step_previews") { return; }
    let (id, default_steps, guidance, scheduler) = bench_model();
    if !prunr_models::is_available(id) {
        eprintln!("SKIP: {id:?} bundle not installed");
        return;
    }
    let steps = std::env::var("PRUNR_SD_BENCH_STEPS").ok().map_or(default_steps, |v| v.parse().expect("PRUNR_SD_BENCH_STEPS"));
    let (w, h) = (900u32, 700u32);
    let image = RgbaImage::from_fn(w, h, |x, y| {
        let stripe = if (x / 40 + y / 40) % 2 == 0 { 40 } else { 0 };
        Rgba([(60 + x / 6) as u8, (90 + y / 8) as u8 + stripe, 120, 255])
    });
    let mut mask = GrayImage::new(w, h);
    for y in 260..440 {
        for x in 380..520 {
            mask.put_pixel(x, y, Luma([255]));
        }
    }
    let req = SdInpaintRequest {
        prompt: "clean background".to_string(),
        num_inference_steps: steps,
        guidance_scale: guidance,
        scheduler,
        seed: Some(42),
        tuning: inpaint_sd::plan_tuning(false, inpaint_sd::SD_DEFAULT_MARGIN_MB).with_env_overrides(),
        ..Default::default()
    };
    const EDGE: u32 = 600;
    let sink = std::sync::Arc::new(RecordingSink::default());
    let previewed = match inpaint_sd::process_inpaint_with(&image, &mask, id, req.clone(), &prunr_core::Progress::new(sink.clone()).with_tile_previews(EDGE)) {
        Ok(r) => r.rgba,
        Err(e) if e.to_string().contains("RAM free") => {
            eprintln!("[sd_step_previews] SKIP: {e}");
            return;
        }
        Err(e) => panic!("SD inpaint should succeed: {e}"),
    };
    let plain = inpaint_sd::process_inpaint_with(&image, &mask, id, req, &prunr_core::Progress::none()).expect("the run without previews");
    assert!(previewed == plain.rgba, "previews change the fill");

    let updates = sink.updates();
    assert_eq!(check_contract(&updates), Ok(()));
    let previews: Vec<_> = updates.iter().filter_map(|u| match u {
        ProgressUpdate::TilePreview(p) => Some(p),
        _ => None,
    }).collect();
    assert_eq!(previews.len(), steps as usize, "one preview per step of the one crop");
    let scale = EDGE as f32 / w as f32;
    let backdrop = image::imageops::resize(&image, EDGE, (h as f32 * scale).round() as u32, image::imageops::FilterType::Triangle);
    for p in &previews {
        assert_eq!((p.full_width, p.full_height), backdrop.dimensions());
        let (hx, hy) = ((380.0 * scale) as u32, (260.0 * scale) as u32);
        assert!(p.x <= hx && p.y <= hy && p.x + p.width >= (520.0 * scale) as u32 && p.y + p.height >= (440.0 * scale) as u32,
            "the preview covers the erased region: {:?}", (p.x, p.y, p.width, p.height));
    }
    if let Ok(dir) = std::env::var("PRUNR_SD_PREVIEW_OUT") {
        for (i, p) in previews.iter().enumerate() {
            let tile = RgbaImage::from_raw(p.width, p.height, p.rgba.clone()).expect("the preview's size");
            let mut frame = backdrop.clone();
            image::imageops::replace(&mut frame, &tile, p.x as i64, p.y as i64);
            frame.save(format!("{dir}/{id:?}_step{:02}.png", i + 1)).expect("save a preview");
        }
        previewed.save(format!("{dir}/{id:?}_fill.png")).expect("save the fill");
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
    let (id, steps, guidance, scheduler) = bench_model();
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
    // The shipped plan, with the same overrides the app honours
    // (`PRUNR_SD_KEEP_LOADED`, `_TALL_CROP`, `_OV_DEVICE`, `_MEM_PATTERN`,
    // `_DEV_INIT`, `_THREADS`).
    let tuning = inpaint_sd::plan_tuning(false, inpaint_sd::SD_DEFAULT_MARGIN_MB).with_env_overrides();
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
