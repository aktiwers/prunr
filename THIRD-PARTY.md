# Third-party notices

Prunr is built from the open-source Rust crates below (every platform's build counted) and ships with the ONNX Runtime. The AI models are credited in the app under Settings → Credits and in `crates/prunr-models/src/lib.rs`, each with its own licence.

## Runtimes

| Component | Licence | Source |
|---|---|---|
| ONNX Runtime (the bundled builds: CPU on Linux, CoreML on macOS, Windows ML with DirectML on Windows; and the OpenVINO, CUDA and MIGraphX builds the app can download) | MIT | https://github.com/microsoft/onnxruntime |
| OpenVINO Toolkit (inside the OpenVINO build of ONNX Runtime) | Apache-2.0 | https://github.com/openvinotoolkit/openvino |
| NVIDIA CUDA and cuDNN libraries (downloaded with the CUDA build of ONNX Runtime) | NVIDIA's licence terms, shown before the download | https://pypi.org/user/nvidia/ |
| Material Symbols (icon font, through egui_material_icons) | Apache-2.0 | https://github.com/google/material-design-icons |

## Rust crates

Generated from `cargo metadata`; the licence column is the crate's own declaration. Regenerate after changing dependencies.

### MIT OR Apache-2.0 (314)

| Crate | Version | Repository |
|---|---|---|
| accesskit | 0.24.0 | https://github.com/AccessKit/accesskit |
| accesskit_atspi_common | 0.18.0 | https://github.com/AccessKit/accesskit |
| accesskit_consumer | 0.35.0 | https://github.com/AccessKit/accesskit |
| accesskit_macos | 0.26.0 | https://github.com/AccessKit/accesskit |
| accesskit_unix | 0.21.0 | https://github.com/AccessKit/accesskit |
| accesskit_windows | 0.32.1 | https://github.com/AccessKit/accesskit |
| ahash | 0.8.12 | https://github.com/tkaitchuck/ahash |
| aligned | 0.4.3 | https://github.com/rust-embedded-community/aligned |
| allocator-api2 | 0.2.21 | https://github.com/zakarumych/allocator-api2 |
| android-activity | 0.6.1 | https://github.com/rust-mobile/android-activity |
| anstream | 1.0.0 | https://github.com/rust-cli/anstyle.git |
| anstyle | 1.0.14 | https://github.com/rust-cli/anstyle.git |
| anstyle-parse | 1.0.0 | https://github.com/rust-cli/anstyle.git |
| anstyle-query | 1.1.5 | https://github.com/rust-cli/anstyle.git |
| anstyle-wincon | 3.0.11 | https://github.com/rust-cli/anstyle.git |
| anyhow | 1.0.102 | https://github.com/dtolnay/anyhow |
| arbitrary | 1.4.2 | https://github.com/rust-fuzz/arbitrary/ |
| arboard | 3.6.1 | https://github.com/1Password/arboard |
| arrayvec | 0.7.6 | https://github.com/bluss/arrayvec |
| as-raw-xcb-connection | 1.0.1 | https://github.com/psychon/as-raw-xcb-connection |
| as-slice | 0.2.1 | https://github.com/japaric/as-slice |
| ash | 0.38.0+1.3.281 | https://github.com/ash-rs/ash |
| async-broadcast | 0.7.2 | https://github.com/smol-rs/async-broadcast |
| async-recursion | 1.1.1 | https://github.com/dcchut/async-recursion |
| async-trait | 0.1.89 | https://github.com/dtolnay/async-trait |
| base64 | 0.22.1 | https://github.com/marshallpierce/rust-base64 |
| bitflags | 2.11.0 | https://github.com/bitflags/bitflags |
| block-buffer | 0.10.4 | https://github.com/RustCrypto/utils |
| bumpalo | 3.20.2 | https://github.com/fitzgen/bumpalo |
| cc | 1.2.59 | https://github.com/rust-lang/cc-rs |
| cfg-expr | 0.15.8 | https://github.com/EmbarkStudios/cfg-expr |
| cfg-if | 1.0.4 | https://github.com/rust-lang/cfg-if |
| clap | 4.6.0 | https://github.com/clap-rs/clap |
| clap_builder | 4.6.0 | https://github.com/clap-rs/clap |
| clap_derive | 4.6.0 | https://github.com/clap-rs/clap |
| clap_lex | 1.1.0 | https://github.com/clap-rs/clap |
| cocoa | 0.26.1 | https://github.com/servo/core-foundation-rs |
| cocoa-foundation | 0.2.1 | https://github.com/servo/core-foundation-rs |
| colorchoice | 1.0.5 | https://github.com/rust-cli/anstyle.git |
| core-foundation | 0.10.1 | https://github.com/servo/core-foundation-rs |
| core-foundation | 0.9.4 | https://github.com/servo/core-foundation-rs |
| core-foundation-sys | 0.8.7 | https://github.com/servo/core-foundation-rs |
| core-graphics | 0.23.2 | https://github.com/servo/core-foundation-rs |
| core-graphics | 0.24.0 | https://github.com/servo/core-foundation-rs |
| core-graphics-types | 0.1.3 | https://github.com/servo/core-foundation-rs |
| core-graphics-types | 0.2.0 | https://github.com/servo/core-foundation-rs |
| cpufeatures | 0.2.17 | https://github.com/RustCrypto/utils |
| crc32fast | 1.5.0 | https://github.com/srijs/rust-crc32fast |
| crossbeam-deque | 0.8.6 | https://github.com/crossbeam-rs/crossbeam |
| crossbeam-epoch | 0.9.18 | https://github.com/crossbeam-rs/crossbeam |
| crossbeam-utils | 0.8.21 | https://github.com/crossbeam-rs/crossbeam |
| crypto-common | 0.1.7 | https://github.com/RustCrypto/traits |
| dary_heap | 0.3.8 | https://github.com/hanmertens/dary_heap |
| data-url | 0.3.2 | https://github.com/servo/rust-url |
| derive_arbitrary | 1.4.2 | https://github.com/rust-fuzz/arbitrary |
| digest | 0.10.7 | https://github.com/RustCrypto/traits |
| dirs | 6.0.0 | https://github.com/soc/dirs-rs |
| dirs-sys | 0.5.0 | https://github.com/dirs-dev/dirs-sys-rs |
| displaydoc | 0.2.5 | https://github.com/yaahc/displaydoc |
| document-features | 0.2.12 | https://github.com/slint-ui/document-features |
| ecolor | 0.34.1 | https://github.com/emilk/egui |
| eframe | 0.34.1 | https://github.com/emilk/egui/tree/main/crates/eframe |
| egui | 0.34.1 | https://github.com/emilk/egui |
| egui-wgpu | 0.34.1 | https://github.com/emilk/egui/tree/main/crates/egui-wgpu |
| egui-winit | 0.34.1 | https://github.com/emilk/egui/tree/main/crates/egui-winit |
| egui_extras | 0.34.1 | https://github.com/emilk/egui |
| egui_glow | 0.34.1 | https://github.com/emilk/egui/tree/main/crates/egui_glow |
| either | 1.15.0 | https://github.com/rayon-rs/either |
| emath | 0.34.1 | https://github.com/emilk/egui/tree/main/crates/emath |
| enum-map | 2.7.3 | https://codeberg.org/xfix/enum-map |
| enum-map-derive | 0.17.0 | https://codeberg.org/xfix/enum-map |
| enumflags2 | 0.7.12 | https://github.com/meithecatte/enumflags2 |
| enumflags2_derive | 0.7.12 | https://github.com/meithecatte/enumflags2 |
| epaint | 0.34.1 | https://github.com/emilk/egui/tree/main/crates/epaint |
| errno | 0.3.14 | https://github.com/lambda-fairy/rust-errno |
| euclid | 0.22.14 | https://github.com/servo/euclid |
| fast_image_resize | 6.0.0 | https://github.com/cykooz/fast_image_resize |
| fdeflate | 0.3.7 | https://github.com/image-rs/fdeflate |
| field-offset | 0.3.6 | https://github.com/Diggsey/rust-field-offset |
| find-msvc-tools | 0.1.9 | https://github.com/rust-lang/cc-rs |
| fixedbitset | 0.5.7 | https://github.com/petgraph/fixedbitset |
| flate2 | 1.1.9 | https://github.com/rust-lang/flate2-rs |
| font-types | 0.11.2 | https://github.com/googlefonts/fontations |
| form_urlencoded | 1.2.2 | https://github.com/servo/rust-url |
| futures-channel | 0.3.32 | https://github.com/rust-lang/futures-rs |
| futures-core | 0.3.32 | https://github.com/rust-lang/futures-rs |
| futures-executor | 0.3.32 | https://github.com/rust-lang/futures-rs |
| futures-io | 0.3.32 | https://github.com/rust-lang/futures-rs |
| futures-macro | 0.3.32 | https://github.com/rust-lang/futures-rs |
| futures-sink | 0.3.32 | https://github.com/rust-lang/futures-rs |
| futures-task | 0.3.32 | https://github.com/rust-lang/futures-rs |
| futures-util | 0.3.32 | https://github.com/rust-lang/futures-rs |
| getrandom | 0.2.17 | https://github.com/rust-random/getrandom |
| getrandom | 0.3.4 | https://github.com/rust-random/getrandom |
| gif | 0.13.3 | https://github.com/image-rs/image-gif |
| gif | 0.14.1 | https://github.com/image-rs/image-gif |
| gpu-allocator | 0.28.0 | https://github.com/Traverse-Research/gpu-allocator |
| gpu-descriptor | 0.3.2 | https://github.com/zakarumych/gpu-descriptor |
| gpu-descriptor-types | 0.2.0 | https://github.com/zakarumych/gpu-descriptor |
| half | 2.7.1 | https://github.com/VoidStarKat/half-rs |
| hashbrown | 0.15.5 | https://github.com/rust-lang/hashbrown |
| hashbrown | 0.16.1 | https://github.com/rust-lang/hashbrown |
| heck | 0.4.1 | https://github.com/withoutboats/heck |
| heck | 0.5.0 | https://github.com/withoutboats/heck |
| hermit-abi | 0.5.2 | https://github.com/hermit-os/hermit-rs |
| hex | 0.4.3 | https://github.com/KokaKiwi/rust-hex |
| http | 1.4.0 | https://github.com/hyperium/http |
| httparse | 1.10.1 | https://github.com/seanmonstar/httparse |
| idna | 1.1.0 | https://github.com/servo/rust-url/ |
| image | 0.25.10 | https://github.com/image-rs/image |
| image-webp | 0.2.4 | https://github.com/image-rs/image-webp |
| ipnet | 2.12.0 | https://github.com/krisprice/ipnet |
| iri-string | 0.7.12 | https://github.com/lo48576/iri-string |
| is_terminal_polyfill | 1.70.2 | https://github.com/polyfill-rs/is_terminal_polyfill |
| itertools | 0.14.0 | https://github.com/rust-itertools/itertools |
| itoa | 1.0.18 | https://github.com/dtolnay/itoa |
| jni | 0.22.4 | https://github.com/jni-rs/jni-rs |
| jni-macros | 0.22.4 | https://github.com/jni-rs/jni-rs |
| jni-sys | 0.3.1 | https://github.com/jni-rs/jni-sys |
| jni-sys | 0.4.1 | https://github.com/jni-rs/jni-sys |
| jni-sys-macros | 0.4.1 | https://github.com/jni-rs/jni-sys |
| jobserver | 0.1.34 | https://github.com/rust-lang/jobserver-rs |
| js-sys | 0.3.94 | https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/js-sys |
| lazy_static | 1.5.0 | https://github.com/rust-lang-nursery/lazy-static.rs |
| libc | 0.2.184 | https://github.com/rust-lang/libc |
| litrs | 1.0.0 | https://github.com/LukasKalbertodt/litrs |
| lock_api | 0.4.14 | https://github.com/Amanieu/parking_lot |
| log | 0.4.29 | https://github.com/rust-lang/log |
| memmap2 | 0.9.10 | https://github.com/RazrFalcon/memmap2-rs |
| memory-stats | 1.2.0 | https://github.com/Arc-blroth/memory-stats |
| mime | 0.3.17 | https://github.com/hyperium/mime |
| naga | 29.0.1 | https://github.com/gfx-rs/wgpu |
| ndarray | 0.17.2 | https://github.com/rust-ndarray/ndarray |
| ndk | 0.9.0 | https://github.com/rust-mobile/ndk |
| ndk-context | 0.1.1 | https://github.com/rust-windowing/android-ndk-rs |
| ndk-sys | 0.6.0+11769913 | https://github.com/rust-mobile/ndk |
| num-bigint | 0.4.6 | https://github.com/rust-num/num-bigint |
| num-complex | 0.4.6 | https://github.com/rust-num/num-complex |
| num-derive | 0.4.2 | https://github.com/rust-num/num-derive |
| num-integer | 0.1.46 | https://github.com/rust-num/num-integer |
| num-rational | 0.4.2 | https://github.com/rust-num/num-rational |
| num-traits | 0.2.19 | https://github.com/rust-num/num-traits |
| num_cpus | 1.17.0 | https://github.com/seanmonstar/num_cpus |
| once_cell | 1.21.4 | https://github.com/matklad/once_cell |
| once_cell_polyfill | 1.70.2 | https://github.com/polyfill-rs/once_cell_polyfill |
| ordered-stream | 0.2.0 | https://github.com/danieldg/ordered-stream |
| ort | 2.0.0-rc.13 | https://github.com/pykeio/ort |
| ort-sys | 2.0.0-rc.13 | https://github.com/pykeio/ort |
| parking_lot | 0.12.5 | https://github.com/Amanieu/parking_lot |
| parking_lot_core | 0.9.12 | https://github.com/Amanieu/parking_lot |
| paste | 1.0.15 | https://github.com/dtolnay/paste |
| pastey | 0.1.1 | https://github.com/as1100k/pastey |
| percent-encoding | 2.3.2 | https://github.com/servo/rust-url/ |
| petgraph | 0.8.3 | https://github.com/petgraph/petgraph |
| piper | 0.2.5 | https://github.com/smol-rs/piper |
| pkg-config | 0.3.32 | https://github.com/rust-lang/pkg-config-rs |
| png | 0.17.16 | https://github.com/image-rs/image-png |
| png | 0.18.1 | https://github.com/image-rs/image-png |
| ppv-lite86 | 0.2.21 | https://github.com/cryptocorrosion/cryptocorrosion |
| presser | 0.3.1 | https://github.com/EmbarkStudios/presser |
| proc-macro-crate | 1.3.1 | https://github.com/bkchr/proc-macro-crate |
| proc-macro-crate | 2.0.2 | https://github.com/bkchr/proc-macro-crate |
| proc-macro-crate | 3.5.0 | https://github.com/bkchr/proc-macro-crate |
| proc-macro-error | 1.0.4 | https://gitlab.com/CreepySkeleton/proc-macro-error |
| proc-macro-error-attr | 1.0.4 | https://gitlab.com/CreepySkeleton/proc-macro-error |
| proc-macro-error-attr2 | 2.0.0 | https://github.com/GnomedDev/proc-macro-error-2 |
| proc-macro-error2 | 2.0.1 | https://github.com/GnomedDev/proc-macro-error-2 |
| proc-macro2 | 1.0.106 | https://github.com/dtolnay/proc-macro2 |
| profiling | 1.0.17 | https://github.com/aclysma/profiling |
| profiling-procmacros | 1.0.17 | https://github.com/aclysma/profiling |
| quinn | 0.11.9 | https://github.com/quinn-rs/quinn |
| quinn-proto | 0.11.14 | https://github.com/quinn-rs/quinn |
| quinn-udp | 0.5.14 | https://github.com/quinn-rs/quinn |
| quote | 1.0.45 | https://github.com/dtolnay/quote |
| rand | 0.8.5 | https://github.com/rust-random/rand |
| rand | 0.9.2 | https://github.com/rust-random/rand |
| rand_chacha | 0.9.0 | https://github.com/rust-random/rand |
| rand_core | 0.6.4 | https://github.com/rust-random/rand |
| rand_core | 0.9.5 | https://github.com/rust-random/rand |
| rand_distr | 0.5.1 | https://github.com/rust-random/rand_distr |
| range-alloc | 0.1.5 | https://github.com/gfx-rs/range-alloc |
| raw-window-metal | 1.1.0 | https://github.com/rust-windowing/raw-window-metal |
| rayon | 1.11.0 | https://github.com/rayon-rs/rayon |
| rayon-core | 1.13.0 | https://github.com/rayon-rs/rayon |
| read-fonts | 0.37.0 | https://github.com/googlefonts/fontations |
| regex | 1.12.3 | https://github.com/rust-lang/regex |
| regex-automata | 0.4.14 | https://github.com/rust-lang/regex |
| regex-syntax | 0.8.10 | https://github.com/rust-lang/regex |
| renderdoc-sys | 1.1.0 | https://github.com/ebkalderon/renderdoc-rs |
| reqwest | 0.12.28 | https://github.com/seanmonstar/reqwest |
| rle-decode-fast | 1.0.3 | https://github.com/WanzenBug/rle-decode-helper |
| roxmltree | 0.20.0 | https://github.com/RazrFalcon/roxmltree |
| rustc_version | 0.4.1 | https://github.com/djc/rustc-version-rs |
| rustls-pki-types | 1.14.0 | https://github.com/rustls/pki-types |
| rustversion | 1.0.22 | https://github.com/dtolnay/rustversion |
| scopeguard | 1.2.0 | https://github.com/bluss/scopeguard |
| semver | 1.0.28 | https://github.com/dtolnay/semver |
| serde | 1.0.228 | https://github.com/serde-rs/serde |
| serde_core | 1.0.228 | https://github.com/serde-rs/serde |
| serde_derive | 1.0.228 | https://github.com/serde-rs/serde |
| serde_json | 1.0.149 | https://github.com/serde-rs/json |
| serde_repr | 0.1.20 | https://github.com/dtolnay/serde-repr |
| serde_spanned | 0.6.9 | https://github.com/toml-rs/toml |
| serde_spanned | 1.1.1 | https://github.com/toml-rs/toml |
| sha2 | 0.10.9 | https://github.com/RustCrypto/hashes |
| shlex | 1.3.0 | https://github.com/comex/rust-shlex |
| signal-hook-registry | 1.4.8 | https://github.com/vorner/signal-hook |
| simdutf8 | 0.1.5 | https://github.com/rusticstuff/simdutf8 |
| skrifa | 0.40.0 | https://github.com/googlefonts/fontations |
| smallvec | 1.15.1 | https://github.com/servo/rust-smallvec |
| smol_str | 0.2.2 | https://github.com/rust-analyzer/smol_str |
| socket2 | 0.6.3 | https://github.com/rust-lang/socket2 |
| stable_deref_trait | 1.2.1 | https://github.com/storyyeller/stable_deref_trait |
| static_assertions | 1.1.0 | https://github.com/nvzqz/static-assertions-rs |
| syn | 1.0.109 | https://github.com/dtolnay/syn |
| syn | 2.0.117 | https://github.com/dtolnay/syn |
| system-deps | 6.2.2 | https://github.com/gdesmott/system-deps |
| tempfile | 3.27.0 | https://github.com/Stebalien/tempfile |
| thiserror | 1.0.69 | https://github.com/dtolnay/thiserror |
| thiserror | 2.0.18 | https://github.com/dtolnay/thiserror |
| thiserror-impl | 1.0.69 | https://github.com/dtolnay/thiserror |
| thiserror-impl | 2.0.18 | https://github.com/dtolnay/thiserror |
| thread_local | 1.1.9 | https://github.com/Amanieu/thread_local-rs |
| tokio-rustls | 0.26.4 | https://github.com/rustls/tokio-rustls |
| toml | 0.8.2 | https://github.com/toml-rs/toml |
| toml | 1.1.2+spec-1.1.0 | https://github.com/toml-rs/toml |
| toml_datetime | 0.6.3 | https://github.com/toml-rs/toml |
| toml_datetime | 1.1.1+spec-1.1.0 | https://github.com/toml-rs/toml |
| toml_edit | 0.19.15 | https://github.com/toml-rs/toml |
| toml_edit | 0.20.2 | https://github.com/toml-rs/toml |
| toml_edit | 0.25.10+spec-1.1.0 | https://github.com/toml-rs/toml |
| toml_parser | 1.1.2+spec-1.1.0 | https://github.com/toml-rs/toml |
| toml_writer | 1.1.1+spec-1.1.0 | https://github.com/toml-rs/toml |
| ttf-parser | 0.25.1 | https://github.com/harfbuzz/ttf-parser |
| typenum | 1.19.0 | https://github.com/paholg/typenum |
| unicase | 2.9.0 | https://github.com/seanmonstar/unicase |
| unicode-bidi | 0.3.18 | https://github.com/servo/unicode-bidi |
| unicode-script | 0.5.8 | https://github.com/unicode-rs/unicode-script |
| unicode-segmentation | 1.13.2 | https://github.com/unicode-rs/unicode-segmentation |
| unicode-width | 0.2.2 | https://github.com/unicode-rs/unicode-width |
| unty | 0.0.4 | https://github.com/bincode-org/unty |
| url | 2.5.8 | https://github.com/servo/rust-url |
| wasm-bindgen | 0.2.117 | https://github.com/wasm-bindgen/wasm-bindgen |
| wasm-bindgen-futures | 0.4.67 | https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/futures |
| wasm-bindgen-macro | 0.2.117 | https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/macro |
| wasm-bindgen-macro-support | 0.2.117 | https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/macro-support |
| wasm-bindgen-shared | 0.2.117 | https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/shared |
| web-sys | 0.3.94 | https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/web-sys |
| web-time | 1.1.0 | https://github.com/daxpedda/web-time |
| webbrowser | 1.2.0 | https://github.com/amodm/webbrowser-rs |
| weezl | 0.1.12 | https://github.com/image-rs/weezl |
| wgpu | 29.0.1 | https://github.com/gfx-rs/wgpu |
| wgpu-core | 29.0.1 | https://github.com/gfx-rs/wgpu |
| wgpu-core-deps-apple | 29.0.0 | https://github.com/gfx-rs/wgpu |
| wgpu-core-deps-emscripten | 29.0.0 | https://github.com/gfx-rs/wgpu |
| wgpu-core-deps-wasm | 29.0.0 | https://github.com/gfx-rs/wgpu |
| wgpu-core-deps-windows-linux-android | 29.0.0 | https://github.com/gfx-rs/wgpu |
| wgpu-hal | 29.0.1 | https://github.com/gfx-rs/wgpu |
| wgpu-naga-bridge | 29.0.1 | https://github.com/gfx-rs/wgpu |
| wgpu-types | 29.0.1 | https://github.com/gfx-rs/wgpu |
| windows | 0.52.0 | https://github.com/microsoft/windows-rs |
| windows | 0.61.3 | https://github.com/microsoft/windows-rs |
| windows | 0.62.2 | https://github.com/microsoft/windows-rs |
| windows-collections | 0.2.0 | https://github.com/microsoft/windows-rs |
| windows-collections | 0.3.2 | https://github.com/microsoft/windows-rs |
| windows-core | 0.52.0 | https://github.com/microsoft/windows-rs |
| windows-core | 0.58.0 | https://github.com/microsoft/windows-rs |
| windows-core | 0.61.2 | https://github.com/microsoft/windows-rs |
| windows-core | 0.62.2 | https://github.com/microsoft/windows-rs |
| windows-future | 0.2.1 | https://github.com/microsoft/windows-rs |
| windows-future | 0.3.2 | https://github.com/microsoft/windows-rs |
| windows-implement | 0.52.0 | https://github.com/microsoft/windows-rs |
| windows-implement | 0.58.0 | https://github.com/microsoft/windows-rs |
| windows-implement | 0.60.2 | https://github.com/microsoft/windows-rs |
| windows-interface | 0.52.0 | https://github.com/microsoft/windows-rs |
| windows-interface | 0.58.0 | https://github.com/microsoft/windows-rs |
| windows-interface | 0.59.3 | https://github.com/microsoft/windows-rs |
| windows-link | 0.1.3 | https://github.com/microsoft/windows-rs |
| windows-link | 0.2.1 | https://github.com/microsoft/windows-rs |
| windows-numerics | 0.2.0 | https://github.com/microsoft/windows-rs |
| windows-numerics | 0.3.1 | https://github.com/microsoft/windows-rs |
| windows-result | 0.2.0 | https://github.com/microsoft/windows-rs |
| windows-result | 0.3.4 | https://github.com/microsoft/windows-rs |
| windows-result | 0.4.1 | https://github.com/microsoft/windows-rs |
| windows-strings | 0.1.0 | https://github.com/microsoft/windows-rs |
| windows-strings | 0.4.2 | https://github.com/microsoft/windows-rs |
| windows-strings | 0.5.1 | https://github.com/microsoft/windows-rs |
| windows-sys | 0.52.0 | https://github.com/microsoft/windows-rs |
| windows-sys | 0.59.0 | https://github.com/microsoft/windows-rs |
| windows-sys | 0.60.2 | https://github.com/microsoft/windows-rs |
| windows-sys | 0.61.2 | https://github.com/microsoft/windows-rs |
| windows-targets | 0.52.6 | https://github.com/microsoft/windows-rs |
| windows-targets | 0.53.5 | https://github.com/microsoft/windows-rs |
| windows-threading | 0.1.0 | https://github.com/microsoft/windows-rs |
| windows-threading | 0.2.1 | https://github.com/microsoft/windows-rs |
| windows_aarch64_gnullvm | 0.52.6 | https://github.com/microsoft/windows-rs |
| windows_aarch64_gnullvm | 0.53.1 | https://github.com/microsoft/windows-rs |
| windows_aarch64_msvc | 0.52.6 | https://github.com/microsoft/windows-rs |
| windows_aarch64_msvc | 0.53.1 | https://github.com/microsoft/windows-rs |
| windows_i686_gnu | 0.52.6 | https://github.com/microsoft/windows-rs |
| windows_i686_gnu | 0.53.1 | https://github.com/microsoft/windows-rs |
| windows_i686_gnullvm | 0.52.6 | https://github.com/microsoft/windows-rs |
| windows_i686_gnullvm | 0.53.1 | https://github.com/microsoft/windows-rs |
| windows_i686_msvc | 0.52.6 | https://github.com/microsoft/windows-rs |
| windows_i686_msvc | 0.53.1 | https://github.com/microsoft/windows-rs |
| windows_x86_64_gnu | 0.52.6 | https://github.com/microsoft/windows-rs |
| windows_x86_64_gnu | 0.53.1 | https://github.com/microsoft/windows-rs |
| windows_x86_64_gnullvm | 0.52.6 | https://github.com/microsoft/windows-rs |
| windows_x86_64_gnullvm | 0.53.1 | https://github.com/microsoft/windows-rs |
| windows_x86_64_msvc | 0.52.6 | https://github.com/microsoft/windows-rs |
| windows_x86_64_msvc | 0.53.1 | https://github.com/microsoft/windows-rs |
| x11rb | 0.13.2 | https://github.com/psychon/x11rb |
| x11rb-protocol | 0.13.2 | https://github.com/psychon/x11rb |
| zstd-safe | 7.2.4 | https://github.com/gyscos/zstd-rs |

### MIT (187)

| Crate | Version | Repository |
|---|---|---|
| aligned-vec | 0.6.4 | https://github.com/sarah-ek/aligned-vec/ |
| android-properties | 0.2.2 | https://github.com/miklelappo/android-properties |
| arg_enum_proc_macro | 0.3.4 | https://github.com/lu-zero/arg_enum_proc_macro |
| ashpd | 0.11.1 | https://github.com/bilelmoussaoui/ashpd |
| atk | 0.18.2 | https://github.com/gtk-rs/gtk3-rs |
| atk-sys | 0.18.2 | https://github.com/gtk-rs/gtk3-rs |
| av-scenechange | 0.14.1 | https://github.com/rust-av/av-scenechange |
| bincode | 2.0.1 | https://github.com/bincode-org/bincode |
| bincode_derive | 2.0.1 | https://github.com/bincode-org/bincode |
| block | 0.1.6 | http://github.com/SSheldon/rust-block |
| block2 | 0.5.1 | https://github.com/madsmtm/objc2 |
| block2 | 0.6.2 | https://github.com/madsmtm/objc2 |
| built | 0.8.0 | https://github.com/lukaslueg/built |
| bytes | 1.11.1 | https://github.com/tokio-rs/bytes |
| cairo-rs | 0.18.5 | https://github.com/gtk-rs/gtk-rs-core |
| cairo-sys-rs | 0.18.2 | https://github.com/gtk-rs/gtk-rs-core |
| calloop | 0.13.0 | https://github.com/Smithay/calloop |
| calloop | 0.14.4 | https://github.com/Smithay/calloop |
| calloop-wayland-source | 0.3.0 | https://github.com/smithay/calloop-wayland-source |
| calloop-wayland-source | 0.4.1 | https://github.com/smithay/calloop-wayland-source |
| cfg_aliases | 0.2.1 | https://github.com/katharostech/cfg_aliases |
| color_quant | 1.1.0 | https://github.com/image-rs/color_quant.git |
| combine | 4.6.7 | https://github.com/Marwes/combine |
| console | 0.15.11 | https://github.com/console-rs/console |
| core_maths | 0.1.1 | https://github.com/robertbastian/core_maths |
| crunchy | 0.2.4 | https://github.com/eira-fransham/crunchy |
| dispatch | 0.2.0 | http://github.com/SSheldon/rust-dispatch |
| dlib | 0.5.3 | https://github.com/elinorbgr/dlib |
| egui-notify | 0.22.0 | https://github.com/ItsEthra/egui-notify |
| egui_material_icons | 0.6.0 | https://github.com/lucasmerlin/hello_egui/tree/main/crates/egui_material_icons |
| endi | 1.1.1 | https://github.com/zeenix/endi |
| equator | 0.4.2 | https://github.com/sarah-ek/equator/ |
| equator-macro | 0.4.2 | https://github.com/sarah-ek/equator/ |
| fax | 0.2.6 | https://github.com/pdf-rs/fax |
| fax_derive | 0.2.0 | https://github.com/pdf-rs/fax |
| float-cmp | 0.9.0 | https://github.com/mikedilger/float-cmp |
| fontconfig-parser | 0.5.8 | https://github.com/Riey/fontconfig-parser |
| fontdb | 0.23.0 | https://github.com/RazrFalcon/fontdb |
| gdk | 0.18.2 | https://github.com/gtk-rs/gtk3-rs |
| gdk-pixbuf | 0.18.5 | https://github.com/gtk-rs/gtk-rs-core |
| gdk-pixbuf-sys | 0.18.0 | https://github.com/gtk-rs/gtk-rs-core |
| gdk-sys | 0.18.2 | https://github.com/gtk-rs/gtk3-rs |
| gdkx11 | 0.18.2 | https://github.com/gtk-rs/gtk3-rs |
| gdkx11-sys | 0.18.2 | https://github.com/gtk-rs/gtk3-rs |
| generic-array | 0.14.7 | https://github.com/fizyk20/generic-array.git |
| gio | 0.18.4 | https://github.com/gtk-rs/gtk-rs-core |
| gio-sys | 0.18.1 | https://github.com/gtk-rs/gtk-rs-core |
| glib | 0.18.5 | https://github.com/gtk-rs/gtk-rs-core |
| glib-macros | 0.18.5 | https://github.com/gtk-rs/gtk-rs-core |
| glib-sys | 0.18.1 | https://github.com/gtk-rs/gtk-rs-core |
| glutin-winit | 0.5.0 | https://github.com/rust-windowing/glutin |
| gobject-sys | 0.18.0 | https://github.com/gtk-rs/gtk-rs-core |
| gtk | 0.18.2 | https://github.com/gtk-rs/gtk3-rs |
| gtk-sys | 0.18.2 | https://github.com/gtk-rs/gtk3-rs |
| gtk3-macros | 0.18.2 | https://github.com/gtk-rs/gtk3-rs |
| http-body | 1.0.1 | https://github.com/hyperium/http-body |
| http-body-util | 0.1.3 | https://github.com/hyperium/http-body |
| hyper | 1.9.0 | https://github.com/hyperium/hyper |
| hyper-util | 0.1.20 | https://github.com/hyperium/hyper-util |
| imagesize | 0.13.0 | https://github.com/Roughsketch/imagesize |
| indicatif | 0.17.11 | https://github.com/console-rs/indicatif |
| instant-clip-tokenizer | 0.1.0 | https://github.com/instant-labs/instant-clip-tokenizer |
| interpolate_name | 0.2.4 | https://github.com/lu-zero/interpolate_name |
| libflate | 2.2.1 | https://github.com/sile/libflate |
| libflate_lz77 | 2.2.0 | https://github.com/sile/libflate |
| libm | 0.2.16 | https://github.com/rust-lang/compiler-builtins |
| libredox | 0.1.15 | https://gitlab.redox-os.org/redox-os/libredox.git |
| loop9 | 0.1.5 | https://gitlab.com/kornelski/loop9.git |
| malloc_buf | 0.0.6 | https://github.com/SSheldon/malloc_buf |
| matchers | 0.2.0 | https://github.com/hawkw/matchers |
| maybe-rayon | 0.1.1 | https://github.com/shssoichiro/maybe-rayon |
| memoffset | 0.9.1 | https://github.com/Gilnaa/memoffset |
| mime_guess2 | 2.3.1 | https://github.com/ttys3/mime_guess2 |
| mio | 1.2.0 | https://github.com/tokio-rs/mio |
| new_debug_unreachable | 1.0.6 | https://github.com/mbrubeck/rust-debug-unreachable |
| nom | 8.0.0 | https://github.com/rust-bakery/nom |
| noop_proc_macro | 0.3.0 | https://github.com/lu-zero/noop_proc_macro |
| nu-ansi-term | 0.50.3 | https://github.com/nushell/nu-ansi-term |
| number_prefix | 0.4.0 | https://github.com/ogham/rust-number-prefix |
| objc | 0.2.7 | http://github.com/SSheldon/rust-objc |
| objc-sys | 0.3.5 | https://github.com/madsmtm/objc2 |
| objc2 | 0.5.2 | https://github.com/madsmtm/objc2 |
| objc2 | 0.6.4 | https://github.com/madsmtm/objc2 |
| objc2-app-kit | 0.2.2 | https://github.com/madsmtm/objc2 |
| objc2-cloud-kit | 0.2.2 | https://github.com/madsmtm/objc2 |
| objc2-contacts | 0.2.2 | https://github.com/madsmtm/objc2 |
| objc2-core-data | 0.2.2 | https://github.com/madsmtm/objc2 |
| objc2-core-image | 0.2.2 | https://github.com/madsmtm/objc2 |
| objc2-core-location | 0.2.2 | https://github.com/madsmtm/objc2 |
| objc2-encode | 4.1.0 | https://github.com/madsmtm/objc2 |
| objc2-foundation | 0.2.2 | https://github.com/madsmtm/objc2 |
| objc2-foundation | 0.3.2 | https://github.com/madsmtm/objc2 |
| objc2-link-presentation | 0.2.2 | https://github.com/madsmtm/objc2 |
| objc2-metal | 0.2.2 | https://github.com/madsmtm/objc2 |
| objc2-quartz-core | 0.2.2 | https://github.com/madsmtm/objc2 |
| objc2-symbols | 0.2.2 | https://github.com/madsmtm/objc2 |
| objc2-ui-kit | 0.2.2 | https://github.com/madsmtm/objc2 |
| objc2-uniform-type-identifiers | 0.2.2 | https://github.com/madsmtm/objc2 |
| objc2-user-notifications | 0.2.2 | https://github.com/madsmtm/objc2 |
| orbclient | 0.3.51 | https://gitlab.redox-os.org/redox-os/orbclient |
| ordered-float | 5.3.0 | https://github.com/reem/rust-ordered-float |
| os_pipe | 1.2.3 | https://github.com/oconnor663/os_pipe.rs |
| pango | 0.18.3 | https://github.com/gtk-rs/gtk-rs-core |
| pango-sys | 0.18.0 | https://github.com/gtk-rs/gtk-rs-core |
| phf | 0.11.3 | https://github.com/rust-phf/rust-phf |
| phf | 0.13.1 | https://github.com/rust-phf/rust-phf |
| phf_generator | 0.11.3 | https://github.com/rust-phf/rust-phf |
| phf_generator | 0.13.1 | https://github.com/rust-phf/rust-phf |
| phf_macros | 0.11.3 | https://github.com/rust-phf/rust-phf |
| phf_macros | 0.13.1 | https://github.com/rust-phf/rust-phf |
| phf_shared | 0.11.3 | https://github.com/rust-phf/rust-phf |
| phf_shared | 0.13.1 | https://github.com/rust-phf/rust-phf |
| pico-args | 0.5.0 | https://github.com/RazrFalcon/pico-args |
| quick-xml | 0.38.4 | https://github.com/tafia/quick-xml |
| quick-xml | 0.39.2 | https://github.com/tafia/quick-xml |
| redox_syscall | 0.4.1 | https://gitlab.redox-os.org/redox-os/syscall |
| redox_syscall | 0.5.18 | https://gitlab.redox-os.org/redox-os/syscall |
| redox_syscall | 0.7.3 | https://gitlab.redox-os.org/redox-os/syscall |
| redox_users | 0.5.2 | https://gitlab.redox-os.org/redox-os/users |
| rfd | 0.15.4 | https://github.com/PolyMeilex/rfd |
| rgb | 0.8.53 | https://github.com/kornelski/rust-rgb |
| rustybuzz | 0.20.1 | https://github.com/harfbuzz/rustybuzz |
| sctk-adwaita | 0.10.1 | https://github.com/PolyMeilex/sctk-adwaita |
| sharded-slab | 0.1.7 | https://github.com/hawkw/sharded-slab |
| simd-adler32 | 0.3.9 | https://github.com/mcountryman/simd-adler32 |
| simd_helpers | 0.1.0 | https://github.com/lu-zero/simd_helpers |
| slab | 0.4.12 | https://github.com/tokio-rs/slab |
| smithay-client-toolkit | 0.19.2 | https://github.com/smithay/client-toolkit |
| smithay-client-toolkit | 0.20.0 | https://github.com/smithay/client-toolkit |
| smithay-clipboard | 0.7.3 | https://github.com/smithay/smithay-clipboard |
| strict-num | 0.1.1 | https://github.com/RazrFalcon/strict-num |
| strsim | 0.11.1 | https://github.com/rapidfuzz/strsim-rs |
| synstructure | 0.13.2 | https://github.com/mystor/synstructure |
| sysinfo | 0.37.2 | https://github.com/GuillaumeGomez/sysinfo |
| tiff | 0.11.3 | https://github.com/image-rs/image-tiff |
| tokio | 1.51.0 | https://github.com/tokio-rs/tokio |
| tower | 0.5.3 | https://github.com/tower-rs/tower |
| tower-http | 0.6.8 | https://github.com/tower-rs/tower-http |
| tower-layer | 0.3.3 | https://github.com/tower-rs/tower |
| tower-service | 0.3.3 | https://github.com/tower-rs/tower |
| tracing | 0.1.44 | https://github.com/tokio-rs/tracing |
| tracing-attributes | 0.1.31 | https://github.com/tokio-rs/tracing |
| tracing-core | 0.1.36 | https://github.com/tokio-rs/tracing |
| tracing-log | 0.2.0 | https://github.com/tokio-rs/tracing |
| tracing-subscriber | 0.3.23 | https://github.com/tokio-rs/tracing |
| tree_magic_mini | 3.2.2 | https://github.com/mbrubeck/tree_magic/ |
| try-lock | 0.2.5 | https://github.com/seanmonstar/try-lock |
| uds_windows | 1.2.1 | https://github.com/haraldh/rust_uds_windows |
| urlencoding | 2.1.3 | https://github.com/kornelski/rust_urlencoding |
| valuable | 0.1.1 | https://github.com/tokio-rs/valuable |
| version-compare | 0.2.1 | https://gitlab.com/timvisee/version-compare |
| virtue | 0.0.18 | https://github.com/bincode-org/virtue |
| want | 0.3.1 | https://github.com/seanmonstar/want |
| wayland-backend | 0.3.15 | https://github.com/smithay/wayland-rs |
| wayland-client | 0.31.14 | https://github.com/smithay/wayland-rs |
| wayland-csd-frame | 0.3.0 | https://github.com/rust-windowing/wayland-csd-frame |
| wayland-cursor | 0.31.14 | https://github.com/smithay/wayland-rs |
| wayland-protocols | 0.32.12 | https://github.com/smithay/wayland-rs |
| wayland-protocols-experimental | 20250721.0.1 | https://github.com/smithay/wayland-rs |
| wayland-protocols-misc | 0.3.12 | https://github.com/smithay/wayland-rs |
| wayland-protocols-plasma | 0.3.12 | https://github.com/smithay/wayland-rs |
| wayland-protocols-wlr | 0.3.12 | https://github.com/smithay/wayland-rs |
| wayland-scanner | 0.31.10 | https://github.com/smithay/wayland-rs |
| wayland-sys | 0.31.11 | https://github.com/smithay/wayland-rs |
| winnow | 0.5.40 | https://github.com/winnow-rs/winnow |
| winnow | 0.7.15 | https://github.com/winnow-rs/winnow |
| winnow | 1.0.1 | https://github.com/winnow-rs/winnow |
| winresource | 0.1.31 | https://github.com/BenjaminRi/winresource |
| x11 | 2.21.0 | https://github.com/AltF02/x11-rs.git |
| x11-dl | 2.21.0 | https://github.com/AltF02/x11-rs.git |
| xcursor | 0.3.10 | https://github.com/esposm03/xcursor-rs |
| xkbcommon-dl | 0.4.2 | https://github.com/rust-windowing/xkbcommon-dl |
| xml-rs | 0.8.28 | https://github.com/kornelski/xml-rs |
| xmlwriter | 0.1.0 | https://github.com/RazrFalcon/xmlwriter |
| y4m | 0.8.0 | https://github.com/image-rs/y4m.git |
| zbus | 5.14.0 | https://github.com/z-galaxy/zbus/ |
| zbus-lockstep | 0.5.2 | https://github.com/luukvanderduim/zbus-lockstep |
| zbus-lockstep-macros | 0.5.2 | https://github.com/luukvanderduim/zbus-lockstep |
| zbus_macros | 5.14.0 | https://github.com/z-galaxy/zbus/ |
| zbus_names | 4.3.1 | https://github.com/z-galaxy/zbus/ |
| zbus_xml | 5.1.0 | https://github.com/z-galaxy/zbus/ |
| zip | 2.4.2 | https://github.com/zip-rs/zip2.git |
| zmij | 1.0.21 | https://github.com/dtolnay/zmij |
| zstd | 0.13.3 | https://github.com/gyscos/zstd-rs |
| zvariant | 5.10.0 | https://github.com/z-galaxy/zbus/ |
| zvariant_derive | 5.10.0 | https://github.com/z-galaxy/zbus/ |
| zvariant_utils | 3.3.0 | https://github.com/z-galaxy/zbus/ |

### Apache-2.0 OR MIT (55)

| Crate | Version | Repository |
|---|---|---|
| async-channel | 2.5.0 | https://github.com/smol-rs/async-channel |
| async-executor | 1.14.0 | https://github.com/smol-rs/async-executor |
| async-fs | 2.2.0 | https://github.com/smol-rs/async-fs |
| async-io | 2.6.0 | https://github.com/smol-rs/async-io |
| async-lock | 3.4.2 | https://github.com/smol-rs/async-lock |
| async-net | 2.0.0 | https://github.com/smol-rs/async-net |
| async-process | 2.5.0 | https://github.com/smol-rs/async-process |
| async-signal | 0.2.13 | https://github.com/smol-rs/async-signal |
| async-task | 4.7.1 | https://github.com/smol-rs/async-task |
| atomic-waker | 1.1.2 | https://github.com/smol-rs/atomic-waker |
| atspi | 0.29.0 | https://github.com/odilia-app/atspi |
| atspi-common | 0.13.0 | https://github.com/odilia-app/atspi |
| atspi-proxies | 0.13.0 | https://github.com/odilia-app/atspi |
| autocfg | 1.5.0 | https://github.com/cuviper/autocfg |
| bit-set | 0.9.1 | https://github.com/contain-rs/bit-set |
| bit-vec | 0.9.1 | https://github.com/contain-rs/bit-vec |
| blocking | 1.6.2 | https://github.com/smol-rs/blocking |
| color | 0.3.2 | https://github.com/linebender/color |
| concurrent-queue | 2.5.0 | https://github.com/smol-rs/concurrent-queue |
| core2 | 0.4.0 | https://github.com/bbqsrc/core2 |
| drag | 2.1.0 |  |
| encode_unicode | 1.0.0 | https://github.com/tormol/encode_unicode |
| equivalent | 1.0.2 | https://github.com/indexmap-rs/equivalent |
| event-listener | 5.4.1 | https://github.com/smol-rs/event-listener |
| event-listener-strategy | 0.5.4 | https://github.com/smol-rs/event-listener-strategy |
| fastrand | 2.4.0 | https://github.com/smol-rs/fastrand |
| fearless_simd | 0.3.0 | https://github.com/linebender/fearless_simd |
| futures-lite | 2.6.1 | https://github.com/smol-rs/futures-lite |
| idna_adapter | 1.2.1 | https://github.com/hsivonen/idna_adapter |
| indexmap | 2.13.1 | https://github.com/indexmap-rs/indexmap |
| kurbo | 0.11.3 | https://github.com/linebender/kurbo |
| kurbo | 0.13.0 | https://github.com/linebender/kurbo |
| linebender_resource_handle | 0.1.1 | https://github.com/linebender/raw_resource_handle |
| nohash-hasher | 0.2.0 | https://github.com/paritytech/nohash-hasher |
| ntapi | 0.4.3 | https://github.com/MSxDOS/ntapi |
| parking | 2.2.1 | https://github.com/smol-rs/parking |
| peniko | 0.6.0 | https://github.com/linebender/peniko |
| pin-project | 1.1.11 | https://github.com/taiki-e/pin-project |
| pin-project-internal | 1.1.11 | https://github.com/taiki-e/pin-project |
| pin-project-lite | 0.2.17 | https://github.com/taiki-e/pin-project-lite |
| polling | 3.11.0 | https://github.com/smol-rs/polling |
| portable-atomic | 1.13.1 | https://github.com/taiki-e/portable-atomic |
| portable-atomic-util | 0.2.6 | https://github.com/taiki-e/portable-atomic-util |
| resvg | 0.45.1 | https://github.com/linebender/resvg |
| rustc-hash | 2.1.2 | https://github.com/rust-lang/rustc-hash |
| simd_cesu8 | 1.1.1 | https://github.com/seancroach/simd_cesu8 |
| simplecss | 0.2.2 | https://github.com/linebender/simplecss |
| svgtypes | 0.15.3 | https://github.com/linebender/svgtypes |
| usvg | 0.45.1 | https://github.com/linebender/resvg |
| utf8_iter | 1.0.4 | https://github.com/hsivonen/utf8_iter |
| utf8parse | 0.2.2 | https://github.com/alacritty/vte |
| uuid | 1.23.0 | https://github.com/uuid-rs/uuid |
| vello_common | 0.0.6 | https://github.com/linebender/vello |
| vello_cpu | 0.0.6 | https://github.com/linebender/vello |
| zeroize | 1.8.2 | https://github.com/RustCrypto/utils |

### MIT/Apache-2.0 (27)

| Crate | Version | Repository |
|---|---|---|
| android_system_properties | 0.1.5 | https://github.com/nical/android_system_properties |
| bitflags | 1.3.2 | https://github.com/bitflags/bitflags |
| bitstream-io | 4.9.0 | https://github.com/tuffy/bitstream-io |
| downcast-rs | 1.2.1 | https://github.com/marcianx/downcast-rs |
| foreign-types | 0.5.0 | https://github.com/sfackler/foreign-types |
| foreign-types-macros | 0.2.3 | https://github.com/sfackler/foreign-types |
| foreign-types-shared | 0.3.1 | https://github.com/sfackler/foreign-types |
| khronos-egl | 6.0.0 | https://github.com/timothee-haudebourg/khronos-egl |
| matrixmultiply | 0.3.10 | https://github.com/bluss/matrixmultiply/ |
| plain | 0.2.3 | https://github.com/randomites/plain |
| qoi | 0.4.1 | https://github.com/aldanor/qoi-rust |
| quick-error | 2.0.1 | http://github.com/tailhook/quick-error |
| rawpointer | 0.2.1 | https://github.com/bluss/rawpointer/ |
| scoped-tls | 1.0.1 | https://github.com/alexcrichton/scoped-tls |
| serde_urlencoded | 0.7.1 | https://github.com/nox/serde_urlencoded |
| siphasher | 1.0.2 | https://github.com/jedisct1/rust-siphash |
| type-map | 0.5.1 | https://github.com/kardeiz/type-map |
| unicode-bidi-mirroring | 0.4.0 | https://github.com/RazrFalcon/unicode-bidi-mirroring |
| unicode-ccc | 0.4.0 | https://github.com/RazrFalcon/unicode-ccc |
| unicode-properties | 0.1.4 | https://github.com/unicode-rs/unicode-properties |
| unicode-vo | 0.1.0 | https://github.com/RazrFalcon/unicode-vo |
| version_check | 0.9.5 | https://github.com/SergioBenitez/version_check |
| winapi | 0.3.9 | https://github.com/retep998/winapi-rs |
| winapi-i686-pc-windows-gnu | 0.4.0 | https://github.com/retep998/winapi-rs |
| winapi-x86_64-pc-windows-gnu | 0.4.0 | https://github.com/retep998/winapi-rs |
| wl-clipboard-rs | 0.9.3 | https://github.com/YaLTeR/wl-clipboard-rs |
| zstd-sys | 2.0.16+zstd.1.5.7 | https://github.com/gyscos/zstd-rs |

### Apache-2.0 (19)

| Crate | Version | Repository |
|---|---|---|
| ab_glyph | 0.2.32 | https://github.com/alexheretic/ab-glyph |
| ab_glyph_rasterizer | 0.1.10 | https://github.com/alexheretic/ab-glyph |
| accesskit_winit | 0.32.2 | https://github.com/AccessKit/accesskit |
| codespan-reporting | 0.13.1 | https://github.com/brendanzab/codespan |
| gethostname | 1.1.0 | https://codeberg.org/swsnr/gethostname.rs.git |
| gl_generator | 0.14.0 | https://github.com/brendanzab/gl-rs/ |
| glutin | 0.32.3 | https://github.com/rust-windowing/glutin |
| glutin_egl_sys | 0.7.1 | https://github.com/rust-windowing/glutin |
| glutin_glx_sys | 0.6.1 | https://github.com/rust-windowing/glutin |
| glutin_wgl_sys | 0.6.1 | https://github.com/rust-windowing/glutin |
| include-flate | 0.3.2 | https://github.com/SOF3/include-flate.git |
| include-flate-codegen | 0.3.2 | https://github.com/SOF3/include-flate.git |
| include-flate-compress | 0.3.2 | https://github.com/SOF3/include-flate.git |
| khronos_api | 3.1.0 | https://github.com/brendanzab/gl-rs/ |
| owned_ttf_parser | 0.25.1 | https://github.com/alexheretic/owned-ttf-parser |
| spirv | 0.4.0+sdk-1.4.341.0 | https://github.com/gfx-rs/rspirv |
| sync_wrapper | 1.0.2 | https://github.com/Actyx/sync_wrapper |
| winit | 0.30.13 | https://github.com/rust-windowing/winit |
| zopfli | 0.8.3 | https://github.com/zopfli-rs/zopfli |

### Unicode-3.0 (18)

| Crate | Version | Repository |
|---|---|---|
| icu_collections | 2.2.0 | https://github.com/unicode-org/icu4x |
| icu_locale_core | 2.2.0 | https://github.com/unicode-org/icu4x |
| icu_normalizer | 2.2.0 | https://github.com/unicode-org/icu4x |
| icu_normalizer_data | 2.2.0 | https://github.com/unicode-org/icu4x |
| icu_properties | 2.2.0 | https://github.com/unicode-org/icu4x |
| icu_properties_data | 2.2.0 | https://github.com/unicode-org/icu4x |
| icu_provider | 2.2.0 | https://github.com/unicode-org/icu4x |
| litemap | 0.8.2 | https://github.com/unicode-org/icu4x |
| potential_utf | 0.1.5 | https://github.com/unicode-org/icu4x |
| tinystr | 0.8.3 | https://github.com/unicode-org/icu4x |
| writeable | 0.6.3 | https://github.com/unicode-org/icu4x |
| yoke | 0.8.2 | https://github.com/unicode-org/icu4x |
| yoke-derive | 0.8.2 | https://github.com/unicode-org/icu4x |
| zerofrom | 0.1.7 | https://github.com/unicode-org/icu4x |
| zerofrom-derive | 0.1.7 | https://github.com/unicode-org/icu4x |
| zerotrie | 0.2.4 | https://github.com/unicode-org/icu4x |
| zerovec | 0.11.6 | https://github.com/unicode-org/icu4x |
| zerovec-derive | 0.11.3 | https://github.com/unicode-org/icu4x |

### Zlib OR Apache-2.0 OR MIT (12)

| Crate | Version | Repository |
|---|---|---|
| bytemuck | 1.25.0 | https://github.com/Lokathor/bytemuck |
| bytemuck_derive | 1.10.2 | https://github.com/Lokathor/bytemuck |
| dispatch2 | 0.3.1 | https://github.com/madsmtm/objc2 |
| objc2-app-kit | 0.3.2 | https://github.com/madsmtm/objc2 |
| objc2-core-foundation | 0.3.2 | https://github.com/madsmtm/objc2 |
| objc2-core-graphics | 0.3.2 | https://github.com/madsmtm/objc2 |
| objc2-io-kit | 0.3.2 | https://github.com/madsmtm/objc2 |
| objc2-io-surface | 0.3.2 | https://github.com/madsmtm/objc2 |
| objc2-metal | 0.3.2 | https://github.com/madsmtm/objc2 |
| objc2-quartz-core | 0.3.2 | https://github.com/madsmtm/objc2 |
| objc2-ui-kit | 0.3.2 | https://github.com/madsmtm/objc2 |
| tinyvec | 1.11.0 | https://github.com/Lokathor/tinyvec |

### MIT OR Apache-2.0 OR Zlib (11)

| Crate | Version | Repository |
|---|---|---|
| cursor-icon | 1.2.0 | https://github.com/rust-windowing/cursor-icon |
| glow | 0.17.0 | https://github.com/grovesNL/glow |
| lru-slab | 0.1.2 | https://github.com/Ralith/lru-slab |
| raw-window-handle | 0.6.2 | https://github.com/rust-windowing/raw-window-handle |
| tinyvec_macros | 0.1.1 | https://github.com/Soveu/tinyvec_macros |
| xkeysym | 0.2.1 | https://github.com/notgull/xkeysym |
| zune-core | 0.4.12 |  |
| zune-core | 0.5.1 | https://github.com/etemesi254/zune-image |
| zune-inflate | 0.2.54 |  |
| zune-jpeg | 0.4.21 | https://github.com/etemesi254/zune-image/tree/dev/crates/zune-jpeg |
| zune-jpeg | 0.5.15 | https://github.com/etemesi254/zune-image/tree/dev/crates/zune-jpeg |

### Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT (7)

| Crate | Version | Repository |
|---|---|---|
| linux-raw-sys | 0.12.1 | https://github.com/sunfishcode/linux-raw-sys |
| linux-raw-sys | 0.4.15 | https://github.com/sunfishcode/linux-raw-sys |
| rustix | 0.38.44 | https://github.com/bytecodealliance/rustix |
| rustix | 1.1.4 | https://github.com/bytecodealliance/rustix |
| wasi | 0.11.1+wasi-snapshot-preview1 | https://github.com/bytecodealliance/wasi |
| wasip2 | 1.0.2+wasi-0.2.9 | https://github.com/bytecodealliance/wasi-rs |
| wit-bindgen | 0.51.0 | https://github.com/bytecodealliance/wit-bindgen |

### BSD-3-Clause (7)

| Crate | Version | Repository |
|---|---|---|
| avif-serialize | 0.8.8 | https://github.com/kornelski/avif-serialize |
| exr | 1.74.0 | https://github.com/johannesvollmer/exrs |
| lebe | 0.5.3 | https://github.com/johannesvollmer/lebe |
| ravif | 0.13.0 | https://github.com/kornelski/cavif-rs |
| subtle | 2.6.1 | https://github.com/dalek-cryptography/subtle |
| tiny-skia | 0.11.4 | https://github.com/RazrFalcon/tiny-skia |
| tiny-skia-path | 0.11.4 | https://github.com/RazrFalcon/tiny-skia/tree/master/path |

### Unlicense OR MIT (5)

| Crate | Version | Repository |
|---|---|---|
| aho-corasick | 1.1.4 | https://github.com/BurntSushi/aho-corasick |
| byteorder-lite | 0.1.0 | https://github.com/image-rs/byteorder-lite |
| memchr | 2.8.0 | https://github.com/BurntSushi/memchr |
| termcolor | 1.4.1 | https://github.com/BurntSushi/termcolor |
| winapi-util | 0.1.11 | https://github.com/BurntSushi/winapi-util |

### BSD-2-Clause (4)

| Crate | Version | Repository |
|---|---|---|
| arrayref | 0.3.9 | https://github.com/droundy/arrayref |
| av1-grain | 0.2.5 | https://github.com/rust-av/av1-grain |
| rav1e | 0.8.1 | https://github.com/xiph/rav1e/ |
| v_frame | 0.3.9 | https://github.com/rust-av/v_frame |

### ISC (4)

| Crate | Version | Repository |
|---|---|---|
| libloading | 0.8.9 | https://github.com/nagisa/rust_libloading/ |
| libloading | 0.9.0 | https://github.com/nagisa/rust_libloading/ |
| rustls-webpki | 0.103.10 | https://github.com/rustls/webpki |
| untrusted | 0.9.0 | https://github.com/briansmith/untrusted |

### Zlib (4)

| Crate | Version | Repository |
|---|---|---|
| adler32 | 1.2.0 | https://github.com/remram44/adler32-rs |
| foldhash | 0.1.5 | https://github.com/orlp/foldhash |
| foldhash | 0.2.0 | https://github.com/orlp/foldhash |
| slotmap | 1.1.1 | https://github.com/orlp/slotmap |

### Apache-2.0/MIT (3)

| Crate | Version | Repository |
|---|---|---|
| bit_field | 0.10.3 | https://github.com/phil-opp/rust-bit-field |
| pollster | 0.4.0 | https://github.com/zesterer/pollster |
| rustc-hash | 1.1.0 | https://github.com/rust-lang-nursery/rustc-hash |

### Apache-2.0 OR ISC OR MIT (2)

| Crate | Version | Repository |
|---|---|---|
| hyper-rustls | 0.27.7 | https://github.com/rustls/hyper-rustls |
| rustls | 0.23.37 | https://github.com/rustls/rustls |

### BSD-2-Clause OR Apache-2.0 OR MIT (2)

| Crate | Version | Repository |
|---|---|---|
| zerocopy | 0.8.48 | https://github.com/google/zerocopy |
| zerocopy-derive | 0.8.48 | https://github.com/google/zerocopy |

### BSD-3-Clause OR Apache-2.0 (2)

| Crate | Version | Repository |
|---|---|---|
| moxcms | 0.8.1 | https://github.com/awxkee/moxcms.git |
| pxfm | 0.1.28 | https://github.com/awxkee/pxfm |

### BSD-3-Clause OR MIT OR Apache-2.0 (2)

| Crate | Version | Repository |
|---|---|---|
| num_enum | 0.7.6 | https://github.com/illicitonion/num_enum |
| num_enum_derive | 0.7.6 | https://github.com/illicitonion/num_enum |

### BSL-1.0 (2)

| Crate | Version | Repository |
|---|---|---|
| clipboard-win | 5.4.1 | https://github.com/DoumanAsh/clipboard-win |
| error-code | 3.3.2 | https://github.com/DoumanAsh/error-code |

### Unlicense/MIT (2)

| Crate | Version | Repository |
|---|---|---|
| same-file | 1.0.6 | https://github.com/BurntSushi/same-file |
| walkdir | 2.5.0 | https://github.com/BurntSushi/walkdir |

### (MIT OR Apache-2.0) AND NCSA (1)

| Crate | Version | Repository |
|---|---|---|
| libfuzzer-sys | 0.4.12 | https://github.com/rust-fuzz/libfuzzer |

### (MIT OR Apache-2.0) AND OFL-1.1 AND Ubuntu-font-1.0 (1)

| Crate | Version | Repository |
|---|---|---|
| epaint_default_fonts | 0.34.1 | https://github.com/emilk/egui/tree/main/crates/epaint_default_fonts |

### (MIT OR Apache-2.0) AND Unicode-3.0 (1)

| Crate | Version | Repository |
|---|---|---|
| unicode-ident | 1.0.24 | https://github.com/dtolnay/unicode-ident |

### 0BSD OR MIT OR Apache-2.0 (1)

| Crate | Version | Repository |
|---|---|---|
| adler2 | 2.0.1 | https://github.com/oyvindln/adler2 |

### Apache-2.0 AND ISC (1)

| Crate | Version | Repository |
|---|---|---|
| ring | 0.17.14 | https://github.com/briansmith/ring |

### Apache-2.0 AND MIT (1)

| Crate | Version | Repository |
|---|---|---|
| dpi | 0.1.2 | https://github.com/rust-windowing/winit |

### Apache-2.0 OR BSL-1.0 (1)

| Crate | Version | Repository |
|---|---|---|
| ryu | 1.0.23 | https://github.com/dtolnay/ryu |

### Apache-2.0 OR GPL-2.0-only (1)

| Crate | Version | Repository |
|---|---|---|
| self_cell | 1.2.2 | https://github.com/Voultapher/self_cell |

### Apache-2.0 WITH LLVM-exception (1)

| Crate | Version | Repository |
|---|---|---|
| target-lexicon | 0.12.16 | https://github.com/bytecodealliance/target-lexicon |

### CC0-1.0 (1)

| Crate | Version | Repository |
|---|---|---|
| hexf-parse | 0.2.1 | https://github.com/lifthrasiir/hexf |

### CC0-1.0 OR Apache-2.0 (1)

| Crate | Version | Repository |
|---|---|---|
| imgref | 1.12.0 | https://github.com/kornelski/imgref |

### CC0-1.0 OR MIT-0 OR Apache-2.0 (1)

| Crate | Version | Repository |
|---|---|---|
| dunce | 1.0.5 | https://gitlab.com/kornelski/dunce |

### CDLA-Permissive-2.0 (1)

| Crate | Version | Repository |
|---|---|---|
| webpki-roots | 1.0.6 | https://github.com/rustls/webpki-roots |

### MIT / Apache-2.0 (1)

| Crate | Version | Repository |
|---|---|---|
| cgl | 0.3.2 | https://github.com/servo/cgl-rs |

### MIT OR Apache-2.0 OR LGPL-2.1-or-later (1)

| Crate | Version | Repository |
|---|---|---|
| r-efi | 5.3.0 | https://github.com/r-efi/r-efi |

### MIT OR Zlib OR Apache-2.0 (1)

| Crate | Version | Repository |
|---|---|---|
| miniz_oxide | 0.8.9 | https://github.com/Frommi/miniz_oxide/tree/master/miniz_oxide |

### MPL-2.0 (1)

| Crate | Version | Repository |
|---|---|---|
| option-ext | 0.2.0 | https://github.com/soc/option-ext.git |
