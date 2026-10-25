//! The registry's `free_dims` name each model's free input sizes, at the
//! axes they size: an engine that compiles each size fixes them by these
//! names, and ORT ignores a name the model does not have. Checks every
//! installed model; skips without an ONNX Runtime.

mod test_common;

use test_common::skip_if_no_ort;

#[test]
fn free_dims_name_each_free_input_size() {
    if skip_if_no_ort("model inputs") {
        return;
    }
    for descriptor in prunr_models::REGISTRY.iter().filter(|d| !d.free_dims.is_empty()) {
        let Some(bytes) = prunr_models::resolve_bytes(descriptor.id) else {
            eprintln!("[model inputs] SKIP {:?}: not installed", descriptor.id);
            continue;
        };
        // Only the input is read: no graph optimizations, which some
        // models need tuned (Nomos8k).
        let session = prunr_core::ort_runtime::session_builder()
            .and_then(|b| b.with_optimization_level(ort::session::builder::GraphOptimizationLevel::Disable).map_err(|e| prunr_core::CoreError::Inference(e.to_string())))
            .expect("a session builder")
            .commit_from_memory(&bytes)
            .unwrap_or_else(|e| panic!("{:?} loads: {e}", descriptor.id));
        let ort::value::ValueType::Tensor { shape, dimension_symbols, .. } = session.inputs()[0].dtype() else {
            panic!("{:?}: its input is not a tensor", descriptor.id);
        };
        let free: Vec<(&str, usize)> = dimension_symbols.iter().enumerate()
            .filter(|&(axis, _)| shape[axis] < 0)
            .map(|(axis, name)| (name.as_str(), axis))
            .collect();
        assert_eq!(free, descriptor.free_dims, "{:?}: the input's free sizes", descriptor.id);
    }
}
