pub const READY: &str = "Ready";

pub struct StatusState {
    pub stage: String,
    pub pct: f32,
    pub text: String,
    /// When a temporary message was set; `None` once it clears.
    set_at: Option<std::time::Instant>,
}

impl Default for StatusState {
    fn default() -> Self {
        Self {
            stage: String::new(),
            pct: 0.0,
            text: READY.to_string(),
            set_at: None,
        }
    }
}

impl StatusState {
    /// Show a temporary status message that auto-clears after 3 seconds.
    pub fn set_temporary(&mut self, text: &str) {
        self.text = text.to_string();
        self.set_at = Some(std::time::Instant::now());
    }

    pub fn is_temporary(&self) -> bool {
        self.set_at.is_some()
    }

    /// Clear temporary status if it has expired (called once per frame).
    pub fn tick(&mut self) {
        if self.set_at.is_some_and(|at| at.elapsed().as_secs_f32() > 3.0) {
            self.text = READY.to_string();
            self.set_at = None;
        }
    }
}
