use super::super::app::PrunrApp;

#[test]
fn handle_cancel_raises_global_flag_during_processing() {
    let mut app = PrunrApp::new_for_test();
    assert!(!app.processor.cancels.is_cancelled(0), "cancel registry should start clean");

    app.handle_cancel();

    assert!(app.processor.cancels.is_cancelled(0),
        "handle_cancel must raise the global flag (every id reports cancelled)");
}

#[test]
fn handle_cancel_selected_raises_only_selected_item_flags() {
    use super::super::item::BatchStatus;
    use super::fixtures::push_test_item;

    let mut app = PrunrApp::new_for_test();
    // Seed batch: id 1 ticked+processing, id 2 ticked+done, id 3 unticked+processing.
    for (id, ticked, status) in [(1u64, true, BatchStatus::Processing), (2, true, BatchStatus::Done), (3, false, BatchStatus::Processing)] {
        let item = push_test_item(&mut app, id);
        item.ticked = ticked;
        crate::gui::tests::fixtures::put_in(item, status);
    }

    app.handle_cancel_selected();

    // Only id 1 matches selected + Processing.
    assert!(app.processor.cancels.is_cancelled(1), "selected processing item must be cancelled");
    assert!(!app.processor.cancels.is_cancelled(2), "selected-but-done item must not be cancelled");
    assert!(!app.processor.cancels.is_cancelled(3), "unselected item must not be cancelled");

    // Cancelled item flipped immediately → thumbnail / canvas spinners stop
    // before the subprocess acknowledges the cancel.
    let item1 = app.batch.find_by_id(1).expect("id 1 still present");
    assert_eq!(item1.status(), BatchStatus::Pending);
    let item3 = app.batch.find_by_id(3).expect("id 3 still present");
    assert_eq!(item3.status(), BatchStatus::Processing, "unselected item keeps running");
}

/// F1 shows the shortcuts where they can also be changed: Settings on
/// the Shortcuts tab. A second press closes it.
#[test]
fn f1_opens_settings_on_shortcuts_and_closes_it_again() {
    use crate::gui::views::settings::SettingsTab;
    use crate::gui::views::shortcuts::Action;
    let ctx = egui::Context::default();
    let mut app = PrunrApp::new_for_test();
    app.perform(Action::Shortcuts, &ctx);
    assert!(app.show_settings);
    assert_eq!(app.settings_tab, SettingsTab::Shortcuts);
    app.perform(Action::Shortcuts, &ctx);
    assert!(!app.show_settings);
    app.settings_tab = SettingsTab::Performance;
    app.show_settings = true;
    app.perform(Action::Shortcuts, &ctx);
    assert!(app.show_settings && app.settings_tab == SettingsTab::Shortcuts, "from another tab it switches, not closes");
}

/// A modal owns the keyboard: under it only what closes or switches it
/// acts; the arrow keys, Undo, Delete and the rest wait for it.
#[test]
fn shortcuts_that_act_on_the_image_wait_under_a_modal() {
    use crate::gui::views::shortcuts::Action;
    let ctx = egui::Context::default();
    let mut app = PrunrApp::new_for_test();
    super::fixtures::push_test_item(&mut app, 1);
    super::fixtures::push_test_item(&mut app, 2);
    let radius = app.settings.brush.radius;
    for (modal, open) in [("settings", (|a: &mut PrunrApp| a.show_settings = true) as fn(&mut PrunrApp)), ("cli_help", |a| a.show_cli_help = true)] {
        open(&mut app);
        assert_eq!(app.open_modals(), [modal]);
        for action in [Action::NextImage, Action::ToggleQueue, Action::ToggleAdjustments, Action::BrushLarger, Action::FitToWindow] {
            app.perform(action, &ctx);
        }
        assert_eq!(app.batch.selected_index, 0, "{modal}: the arrow key waits");
        assert!(!app.sidebar_hidden && !app.adjustments_hidden && !app.zoom_state.pending_fit_zoom, "{modal}: the panels stay");
        assert_eq!(app.settings.brush.radius, radius, "{modal}: the brush stays");
        app.perform(Action::Cancel, &ctx);
        assert!(app.open_modals().is_empty(), "{modal}: Escape closes it");
    }
    app.perform(Action::NextImage, &ctx);
    assert_eq!(app.batch.selected_index, 1, "with no modal up the arrow key acts");
}

/// Escape under a modal that Escape does not close (a file dialog, the
/// reset confirm) leaves the work behind it alone: a running cut goes on.
#[test]
fn escape_under_a_modal_it_cannot_close_cancels_nothing_behind_it() {
    use crate::gui::views::shortcuts::Action;
    let ctx = egui::Context::default();
    let mut app = PrunrApp::new_for_test();
    let item = super::fixtures::push_test_item(&mut app, 1);
    item.ticked = true;
    super::fixtures::put_in(item, super::super::item::BatchStatus::Processing);
    app.pending_reset_confirm = true;
    app.perform(Action::Cancel, &ctx);
    assert!(!app.processor.cancels.is_cancelled(1), "the cut goes on under the modal");
    app.pending_reset_confirm = false;
    app.perform(Action::Cancel, &ctx);
    assert!(app.processor.cancels.is_cancelled(1), "with no modal up Escape cancels it");
}
