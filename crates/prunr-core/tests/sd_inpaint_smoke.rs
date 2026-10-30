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

use image::{GrayImage, Rgba, RgbaImage};
use prunr_core::inpaint_blend;
use prunr_core::inpaint_sd::{self, SdInpaintRequest};
use test_common::{changed_in, rect_mask, skip_if_model_missing, skip_if_no_ort};

#[test]
fn sd_inpaint_modifies_painted_region() {
    let _ = tracing_subscriber::fmt::try_init();
    if skip_if_no_ort("sd_inpaint_smoke") { return; }
    let id = prunr_models::ModelId::SdV15InpaintFp16;
    if skip_if_model_missing("sd_inpaint_smoke", id) { return; }

    // 512×512 solid blue. Mask covers the centre 128×128 region.
    let w: u32 = 512;
    let h: u32 = 512;
    let mut image = RgbaImage::new(w, h);
    for p in image.pixels_mut() { *p = Rgba([40, 80, 200, 255]); }
    let mask = rect_mask(w, h, 192, 192, 128, 128);

    let req = SdInpaintRequest {
        num_inference_steps: 20,
        seed: Some(42),
        ..Default::default()
    };
    let progress = prunr_core::Progress::none();
    let run = inpaint_sd::process_inpaint_with(&image, &mask, id, req, &progress).map(|r| r.rgba);
    let Some(result) = fill_or_skip("sd_inpaint_smoke", run) else { return; };

    assert_eq!(result.dimensions(), image.dimensions());

    let diffs = changed_in(&image, &result, 192, 192, 128, 128);
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

/// The run's fill, or `None` after a SKIP line when the pre-flight RAM
/// gate refused the load: not enough free memory on this machine right
/// now, so the test skips like a missing bundle.
fn fill_or_skip(label: &str, run: Result<RgbaImage, prunr_core::CoreError>) -> Option<RgbaImage> {
    match run {
        Ok(rgba) => Some(rgba),
        Err(e) if e.to_string().contains("of free memory to load") => {
            eprintln!("[{label}] SKIP: {e}");
            None
        }
        Err(e) => panic!("SD inpaint should succeed: {e}"),
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
/// two or three times on one kept session); the model is picked as for
/// `sd_bench`, `PRUNR_SD_BENCH_STEPS` sets the step count, and
/// `PRUNR_SD_PREVIEW_OUT=dir` saves each step's preview pasted on the
/// photo's, and the fills, as PNGs.
#[test]
#[ignore]
fn sd_step_previews() {
    use prunr_core::progress::{check_contract, ProgressUpdate, RecordingSink};
    let _ = tracing_subscriber::fmt::try_init();
    if skip_if_no_ort("sd_step_previews") { return; }
    let (id, default_steps, guidance, scheduler) = bench_model();
    if skip_if_model_missing("sd_step_previews", id) { return; }
    let steps = std::env::var("PRUNR_SD_BENCH_STEPS").ok().map_or(default_steps, |v| v.parse().expect("PRUNR_SD_BENCH_STEPS"));
    let (w, h) = (900u32, 700u32);
    let image = RgbaImage::from_fn(w, h, |x, y| {
        let stripe = if (x / 40 + y / 40) % 2 == 0 { 40 } else { 0 };
        Rgba([(60 + x / 6) as u8, (90 + y / 8) as u8 + stripe, 120, 255])
    });
    let mask = rect_mask(w, h, 380, 260, 140, 180);
    // One kept session for every run, so the runs differ only in their
    // previews.
    let req = SdInpaintRequest {
        prompt: "clean background".to_string(),
        num_inference_steps: steps,
        guidance_scale: guidance,
        scheduler,
        seed: Some(42),
        tuning: inpaint_sd::plan_tuning(true, inpaint_sd::SD_DEFAULT_MARGIN_MB).with_env_overrides(),
        ..Default::default()
    };
    let fill = |progress: &prunr_core::Progress| inpaint_sd::process_inpaint_with(&image, &mask, id, req.clone(), progress).map(|f| f.rgba);
    const EDGE: u32 = 600;
    let sink = std::sync::Arc::new(RecordingSink::default());
    let run = fill(&prunr_core::Progress::new(sink.clone()).with_tile_previews(EDGE));
    let Some(previewed) = fill_or_skip("sd_step_previews", run) else { return; };
    let plain = fill(&prunr_core::Progress::none()).expect("the run without previews");
    let out_dir = std::env::var("PRUNR_SD_PREVIEW_OUT").ok();
    if let Some(dir) = &out_dir {
        previewed.save(format!("{dir}/{id:?}_fill.png")).expect("save the fill");
        plain.save(format!("{dir}/{id:?}_fill_without_previews.png")).expect("save the fill");
    }

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
    if let Some(dir) = &out_dir {
        for (i, p) in previews.iter().enumerate() {
            let tile = RgbaImage::from_raw(p.width, p.height, p.rgba.clone()).expect("the preview's size");
            let mut frame = backdrop.clone();
            image::imageops::replace(&mut frame, &tile, p.x as i64, p.y as i64);
            frame.save(format!("{dir}/{id:?}_step{:02}.png", i + 1)).expect("save a preview");
        }
    }

    let differing = |a: &RgbaImage, b: &RgbaImage| a.pixels().zip(b.pixels()).filter(|(p, q)| p != q).count();
    if previewed != plain {
        // A second plain run tells previews apart from a model that
        // differs run to run on this machine's provider.
        let again = fill(&prunr_core::Progress::none()).expect("the second run without previews");
        assert!(again == plain, "the model differs run to run here ({} px), so previews cannot be told apart", differing(&again, &plain));
        panic!("previews change the fill: {} px differ", differing(&previewed, &plain));
    }
}

/// The soft edge on the real model follows each part's own band: two
/// holes far apart, one with a 4 px band and one with 24 px, fill as each
/// does with that band for the whole region, and one band for every pixel
/// fills as that band does. Ignored by default (five runs on one kept
/// session); the model is picked as for `sd_bench`, and
/// `PRUNR_SD_PREVIEW_OUT=dir` saves the fills.
#[test]
#[ignore]
fn sd_soft_edge_follows_each_parts_band() {
    let _ = tracing_subscriber::fmt::try_init();
    if skip_if_no_ort("sd_soft_edge") { return; }
    let (id, steps, guidance, scheduler) = bench_model();
    if skip_if_model_missing("sd_soft_edge", id) { return; }
    let (w, h) = (1500u32, 700u32);
    let image = RgbaImage::from_fn(w, h, |x, y| {
        let stripe = if (x / 40 + y / 40) % 2 == 0 { 40 } else { 0 };
        Rgba([(60 + x / 9) as u8, (90 + y / 8) as u8 + stripe, 120, 255])
    });
    let (left, right) = ((100u32, 260u32, 160u32, 180u32), (1200u32, 260u32, 160u32, 180u32));
    let mut mask = rect_mask(w, h, left.0, left.1, left.2, left.3);
    let r = rect_mask(w, h, right.0, right.1, right.2, right.3);
    mask.pixels_mut().zip(r.pixels()).for_each(|(m, r)| m.0[0] = m.0[0].max(r.0[0]));
    let req = SdInpaintRequest {
        prompt: "clean background".to_string(),
        num_inference_steps: steps,
        guidance_scale: guidance,
        scheduler,
        seed: Some(42),
        fill_scale: prunr_core::inpaint::FillScale::Full,
        tuning: inpaint_sd::plan_tuning(true, inpaint_sd::SD_DEFAULT_MARGIN_MB).with_env_overrides(),
        ..Default::default()
    };
    let fill = |soft_edge_px: f32, band: Option<GrayImage>| {
        let req = SdInpaintRequest { soft_edge_px, soft_edge_band: band.map(std::sync::Arc::new), ..req.clone() };
        inpaint_sd::process_inpaint_with(&image, &mask, id, req, &prunr_core::Progress::none()).map(|f| f.rgba)
    };
    let Some(narrow) = fill_or_skip("sd_soft_edge", fill(4.0, None)) else { return; };
    let wide = fill(24.0, None).expect("a 24 px band");
    let wide_again = fill(24.0, None).expect("a 24 px band again");
    let mixed = fill(24.0, Some(GrayImage::from_fn(w, h, |x, _| image::Luma([if x < w / 2 { 4 } else { 24 }])))).expect("a band per part");
    let same = fill(24.0, Some(GrayImage::from_pixel(w, h, image::Luma([24])))).expect("one band per pixel");
    if let Ok(dir) = std::env::var("PRUNR_SD_PREVIEW_OUT") {
        for (name, img) in [("narrow", &narrow), ("wide", &wide), ("mixed", &mixed), ("same", &same)] {
            img.save(format!("{dir}/{id:?}_soft_{name}.png")).expect("save a fill");
        }
    }
    // The mean colour difference over a hole and the band around it.
    let apart = |a: &RgbaImage, b: &RgbaImage, (x0, y0, rw, rh): (u32, u32, u32, u32)| {
        let pad = 40;
        let (x0, y0) = (x0.saturating_sub(pad), y0.saturating_sub(pad));
        let (x1, y1) = ((x0 + rw + 2 * pad).min(w), (y0 + rh + 2 * pad).min(h));
        let sum: u64 = (y0..y1).flat_map(|y| (x0..x1).map(move |x| (x, y)))
            .map(|(x, y)| (0..3).map(|c| a.get_pixel(x, y)[c].abs_diff(b.get_pixel(x, y)[c]) as u64).sum::<u64>())
            .sum();
        sum as f64 / ((x1 - x0) * (y1 - y0) * 3) as f64
    };
    // A card may differ run to run; that difference is the floor.
    let noise = apart(&wide, &wide_again, left).max(apart(&wide, &wide_again, right));
    let (own_left, own_right) = (apart(&mixed, &narrow, left), apart(&mixed, &wide, right));
    let (other_left, other_right) = (apart(&mixed, &wide, left), apart(&mixed, &narrow, right));
    let same_vs_wide = apart(&same, &wide, left).max(apart(&same, &wide, right));
    eprintln!("[sd_soft_edge] {id:?} mean difference: run to run {noise:.3}; each part against its own band {own_left:.3} / {own_right:.3}, against the other band {other_left:.3} / {other_right:.3}; one band per pixel against one band {same_vs_wide:.3}");
    let floor = noise.max(0.01) * 2.0;
    assert!(other_left > floor && other_right > floor, "the band changes the fill, or the test shows nothing");
    assert!(own_left <= floor && own_right <= floor, "each part fills as its own band does");
    assert!(same_vs_wide <= floor, "one band per pixel fills as one band");
}

/// A session loaded and dropped for each fill, as the app runs one
/// without Keep loaded, gives its memory back: after five fills the
/// process holds about what it held after the first. Ignored by default
/// (five model loads); the model is picked as for `sd_bench`.
#[test]
#[ignore]
fn sd_reloads_give_their_memory_back() {
    let _ = tracing_subscriber::fmt::try_init();
    if skip_if_no_ort("sd_reloads") { return; }
    let (id, _, guidance, scheduler) = bench_model();
    if skip_if_model_missing("sd_reloads", id) { return; }
    let (w, h) = (768u32, 768u32);
    let image = RgbaImage::from_fn(w, h, |x, y| Rgba([(x / 3 % 256) as u8, (y / 3 % 256) as u8, 128, 255]));
    let mask = rect_mask(w, h, 256, 256, 256, 256);
    let progress = prunr_core::Progress::none();
    let mut held_mb = Vec::new();
    for seed in 0..5 {
        let req = SdInpaintRequest {
            num_inference_steps: 4,
            guidance_scale: guidance,
            scheduler,
            seed: Some(seed),
            tuning: inpaint_sd::plan_tuning(false, inpaint_sd::SD_DEFAULT_MARGIN_MB),
            ..Default::default()
        };
        let run = inpaint_sd::process_inpaint_with(&image, &mask, id, req, &progress).map(|r| r.rgba);
        // Only the first load may skip for a busy machine; a later refusal
        // is what this test is for.
        let filled = if seed == 0 { fill_or_skip("sd_reloads", run) } else { Some(run.unwrap_or_else(|e| panic!("fill {seed}: {e}"))) };
        if filled.is_none() { return; }
        held_mb.push(prunr_core::sysmem::process_rss_mb().expect("this process's memory"));
    }
    eprintln!("[sd_reloads] {id:?} process memory after each fill (MB): {held_mb:?}");
    let grown = held_mb[4].saturating_sub(held_mb[0]);
    assert!(grown < 1000, "the process kept {grown} MB more after five fills than after one: {held_mb:?}");
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
    if skip_if_model_missing("sd_bench", id) { return; }
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
        let mask = rect_mask(w, h, (w - rw) / 2, (h - rh) / 2, rw, rh);
        pairs.push(("synthetic".into(), image, mask));
    }
    // `PRUNR_SD_BENCH_STEPS=2,4,8`, `PRUNR_SD_BENCH_SEED=7,8`,
    // `PRUNR_SD_BENCH_FILL=auto,full`, `PRUNR_SD_BENCH_STRENGTH=1,0.6` and
    // `PRUNR_SD_BENCH_PROMPT=a|b` (split on `|`) run every combination;
    // each defaults to one value.
    let nums = |name: &str| -> Vec<u64> { list(name).iter().map(|v| v.parse().expect(name)).collect() };
    let steps_list = { let v = nums("PRUNR_SD_BENCH_STEPS"); if v.is_empty() { vec![steps as u64] } else { v } };
    let seeds = { let v = nums("PRUNR_SD_BENCH_SEED"); if v.is_empty() { vec![42] } else { v } };
    let strengths: Vec<f32> = {
        let v: Vec<f32> = list("PRUNR_SD_BENCH_STRENGTH").iter().map(|v| v.parse().expect("PRUNR_SD_BENCH_STRENGTH")).collect();
        if v.is_empty() { vec![inpaint_sd::FULL_STRENGTH] } else { v }
    };
    let prompts: Vec<String> = std::env::var("PRUNR_SD_BENCH_PROMPT")
        .map(|v| v.split('|').map(str::to_string).collect())
        .unwrap_or_else(|_| vec!["clean background".to_string()]);
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
                for (&fill_scale, &strength, (p, prompt)) in fills.iter()
                    .flat_map(|f| strengths.iter().map(move |s| (f, s)))
                    .flat_map(|(f, s)| prompts.iter().enumerate().map(move |p| (f, s, p)))
                {
                    let req = SdInpaintRequest {
                        prompt: prompt.clone(),
                        num_inference_steps: steps as u32,
                        guidance_scale: guidance,
                        scheduler,
                        strength,
                        seed: Some(seed),
                        fill_scale,
                        tuning: tuning.clone(),
                        ..Default::default()
                    };
                    let started = std::time::Instant::now();
                    let run = inpaint_sd::process_inpaint_with(image, mask, id, req, &progress).map(|r| r.rgba);
                    let Some(result) = fill_or_skip("sd_bench", run) else { return; };
                    assert_eq!(result.dimensions(), image.dimensions());
                    eprintln!("[sd_bench] {stem} steps={steps} seed={seed} fill={fill_scale} strength={strength} prompt={prompt:?} total {} ms", started.elapsed().as_millis());
                    if let Some(dir) = &out_dir {
                        let done = inpaint_blend::finalize_inpaint(result, image, mask, inpaint_blend::Band::Uniform(inpaint_blend::DEFAULT_EDGE_BLEND_PX), 0.0, inpaint_blend::EdgeColour::Off);
                        done.save(format!("{dir}/{stem}_s{steps}_seed{seed}_{fill_scale}_str{strength}_p{p}.png")).expect("save the bench output");
                    }
                }
            }
        }
    }
}
