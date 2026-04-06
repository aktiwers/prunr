use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Model error: {0}")]
    Model(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_error_model_variant() {
        let err = CoreError::Model("test error".to_string());
        assert_eq!(err.to_string(), "Model error: test error");
    }
}
