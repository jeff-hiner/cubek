//! Standalone kernels for attention operations.
//!
//! These kernels are used for pre-processing steps like quantization
//! that run before the main attention kernel.

mod quantize;

pub use quantize::*;
