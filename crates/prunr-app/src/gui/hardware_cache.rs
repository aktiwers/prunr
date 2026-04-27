use crate::runtime_install::RuntimeId;

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct HardwareInstallCache {
    pub openvino: bool,
}

impl HardwareInstallCache {
    pub fn refresh() -> Self {
        Self {
            openvino: RuntimeId::OpenVino.is_installed(),
        }
    }
}
