//! SageAttention tile implementation using INT8 quantization and DP4a
//!
//! This module implements the core SageAttention algorithm:
//! 1. Quantize Q and K to INT8 with per-block scaling
//! 2. Compute Q·K^T using DP4a (packed INT8 dot product)
//! 3. Dequantize and apply softmax
//! 4. Compute attention·V in FP16

mod attention;
pub mod setup;

pub use attention::*;
