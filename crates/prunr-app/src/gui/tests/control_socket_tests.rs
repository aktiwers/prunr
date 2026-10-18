//! The control socket's commands, each sent through the same inbox and
//! `pump` the socket thread feeds, on a harness window.

use std::sync::mpsc;

use egui_kittest::Harness;
use serde_json::Value;

use super::harness::{loaded, settle};
use crate::gui::app::PrunrApp;
use crate::gui::automation::{Automation, Pending, Request};

/// The loaded window with automation listening on a channel.
fn socket() -> (Harness<'static, PrunrApp>, mpsc::Sender<Pending>) {
    let mut h = loaded();
    let (auto, tx) = Automation::for_test(&h.ctx);
    h.state_mut().automation = Some(auto);
    settle(&mut h);
    (h, tx)
}

/// Sends `json` and runs frames until the reply: `Ok(result)` or `Err(error)`.
fn ask(h: &mut Harness<'_, PrunrApp>, tx: &mpsc::Sender<Pending>, json: &str) -> Result<Value, String> {
    let request: Request = serde_json::from_str(json).expect("a request");
    let (reply, answer) = mpsc::channel();
    tx.send(Pending { request, reply }).expect("the inbox is open");
    for _ in 0..200 {
        h.step();
        if let Ok(r) = answer.try_recv() {
            return match r.ok {
                true => Ok(r.result.unwrap_or(Value::Null)),
                false => Err(r.error.unwrap_or_default()),
            };
        }
    }
    panic!("no reply to {json}");
}

#[test]
fn catalog_lists_every_visit_and_model_by_picker_name() {
    let (mut h, tx) = socket();
    let catalog = ask(&mut h, &tx, r#"{"cmd":"catalog"}"#).unwrap();
    let visits = catalog["visits"].as_array().unwrap();
    assert_eq!(visits.len(), crate::gui::surface_catalog::VISITS.len());
    let models = catalog["models"].as_array().unwrap();
    assert!(models.iter().any(|m| m["name"] == "No model" && m["id"] == "None"), "{models:?}");
    let picked = crate::gui::automation::state::dump(h.state()).model;
    assert!(models.iter().any(|m| m["id"] == picked.as_str()), "state names models by their catalog id");
    let mask = visits.iter().find(|v| v["id"] == "mask").unwrap();
    assert_eq!(mask["open"][0]["click"], "Mask");
    assert!(mask["models"].as_array().unwrap().iter().any(|m| m == "Silueta"), "{mask}");
    let settings = visits.iter().find(|v| v["id"] == "settings").unwrap();
    assert_eq!(settings["kind"]["modal"], "settings");
    assert_eq!(settings["dismiss"]["key"], "Escape");
}

#[test]
fn check_reports_missing_names_and_nothing_on_a_clean_window() {
    let (mut h, tx) = socket();
    assert_eq!(ask(&mut h, &tx, r#"{"cmd":"check","expect":["Process"]}"#).unwrap(), Value::Array(Vec::new()));
    let found = ask(&mut h, &tx, r#"{"cmd":"check","expect":["Process","No such control"]}"#).unwrap();
    let found = found.as_array().unwrap();
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(found[0]["rule"], "missing");
}

#[test]
fn tooltips_lists_what_the_toolbar_s_controls_carry() {
    let (mut h, tx) = socket();
    let tips = ask(&mut h, &tx, r#"{"cmd":"tooltips"}"#).unwrap();
    let titles: Vec<&str> = tips.as_array().unwrap().iter().filter_map(|t| t["title"].as_str()).collect();
    assert!(titles.contains(&"Model"), "{titles:?}");
}

#[test]
fn clips_is_empty_on_the_loaded_window() {
    let (mut h, tx) = socket();
    assert_eq!(ask(&mut h, &tx, r#"{"cmd":"clips"}"#).unwrap(), Value::Array(Vec::new()));
}

/// The harness feeds no input hook, so the queued input is read back:
/// one pointer move onto the control, no press.
#[test]
fn hover_moves_the_pointer_onto_the_control_without_a_click() {
    let (mut h, tx) = socket();
    let target = ask(&mut h, &tx, r#"{"cmd":"hover","name":"Model"}"#).unwrap();
    let r: Vec<f32> = target["rect"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
    let auto = h.state_mut().automation.as_mut().unwrap();
    let frame = auto.take_frame().expect("one input frame");
    let [egui::Event::PointerMoved(pos)] = frame.events.as_slice() else { panic!("{:?}", frame.events) };
    assert!(pos.x >= r[0] && pos.x <= r[2] && pos.y >= r[1] && pos.y <= r[3], "{pos:?} in {r:?}");
    assert!(auto.take_frame().is_none(), "no press follows");
    assert!(ask(&mut h, &tx, r#"{"cmd":"hover","name":"No such control"}"#).is_err());
}

/// A name a quick strip and its open panel share is clicked on the
/// panel's (the later) only when the step asks for the last.
#[test]
fn click_takes_the_last_of_a_shared_name_only_when_asked() {
    let (mut h, tx) = socket();
    super::harness::click(&mut h, "Effect");
    super::harness::click(&mut h, "All effects");
    let shared = super::harness::names(&h).into_iter().filter(|n| n == "Sepia").count();
    assert_eq!(shared, 2, "the strip and the panel both show it");
    assert!(ask(&mut h, &tx, r#"{"cmd":"click","name":"Sepia"}"#).unwrap_err().contains("2 controls named"));
    let clicked = ask(&mut h, &tx, r#"{"cmd":"click","name":"Sepia","last":true}"#).unwrap();
    assert_eq!(clicked["name"], "Sepia");
}

/// A row below a list's fold is scrolled into view by its own node,
/// the AccessKit action egui answers.
#[test]
fn reveal_asks_egui_to_scroll_the_control_into_view() {
    let (mut h, tx) = socket();
    let target = ask(&mut h, &tx, r#"{"cmd":"reveal","name":"Model"}"#).unwrap();
    let frame = h.state_mut().automation.as_mut().unwrap().take_frame().expect("one input frame");
    let [egui::Event::AccessKitActionRequest(request)] = frame.events.as_slice() else { panic!("{:?}", frame.events) };
    assert_eq!(request.action, accesskit::Action::ScrollIntoView);
    assert_eq!(request.target_node, accesskit::NodeId(target["id"].as_u64().unwrap()));
}

#[test]
fn resize_asks_the_window_for_the_size() {
    let (mut h, tx) = socket();
    assert_eq!(ask(&mut h, &tx, r#"{"cmd":"resize","w":900,"h":540}"#).unwrap(), Value::Null);
}

#[test]
fn screenshot_replies_with_the_file_it_writes() {
    let (mut h, tx) = socket();
    let path = ask(&mut h, &tx, r#"{"cmd":"screenshot","name":"the-window"}"#).unwrap();
    let path = std::path::PathBuf::from(path.as_str().unwrap());
    assert_eq!(path, crate::gui::app::screenshot_dir().join("the-window.png"));
    assert!(ask(&mut h, &tx, r#"{"cmd":"screenshot","name":"../escape"}"#).is_err());
}

#[test]
fn wait_idle_waits_out_a_job_or_says_it_timed_out() {
    use crate::gui::item::Engine;
    use crate::gui::timeline::StepKind;
    let (mut h, tx) = socket();
    assert_eq!(ask(&mut h, &tx, r#"{"cmd":"wait_idle"}"#).unwrap()["idle"], true);
    h.state_mut().batch.items[0].start_job(Some(StepKind::Erase), Engine::Erase);
    assert_eq!(ask(&mut h, &tx, r#"{"cmd":"wait_idle","timeout_ms":50}"#).unwrap()["idle"], false);
    h.state_mut().batch.items[0].job = None;
    assert_eq!(ask(&mut h, &tx, r#"{"cmd":"wait_idle"}"#).unwrap()["idle"], true);
}

#[test]
fn state_has_the_viewport_and_the_toasts_on_screen() {
    let (mut h, tx) = socket();
    h.state_mut().toasts.info("Saved one image");
    let state = ask(&mut h, &tx, r#"{"cmd":"state"}"#).unwrap();
    assert_eq!(state["viewport"], serde_json::json!([1400.0, 900.0]));
    assert_eq!(state["toasts"], serde_json::json!(["Saved one image"]));
}
