/// Progress and status text state, displayed in the status bar.
pub struct StatusState {
    pub stage: String,
    pub pct: f32,
    pub text: String,
    pub is_temporary: bool,
    set_at: Option<std::time::Instant>,
}

impl Default for StatusState {
    fn default() -> Self {
        Self {
            stage: String::new(),
            pct: 0.0,
            text: "Ready".to_string(),
            is_temporary: false,
            set_at: None,
        }
    }
}

impl StatusState {
    /// Show a temporary status message that auto-clears after 3 seconds.
    pub fn set_temporary(&mut self, text: &str) {
        self.text = text.to_string();
        self.is_temporary = true;
        self.set_at = Some(std::time::Instant::now());
    }

    /// Clear temporary status if it has expired (called once per frame).
    pub fn tick(&mut self) {
        if self.is_temporary {
            if let Some(set_at) = self.set_at {
                if set_at.elapsed().as_secs_f32() > 3.0 {
                    self.text = "Ready".to_string();
                    self.is_temporary = false;
                    self.set_at = None;
                }
            }
        }
    }
}
