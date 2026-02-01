//! INT8 CMMA tile attention implementation.
//!
//! This module provides hardware-accelerated INT8 attention using CMMA (tensor cores):
//! - Q·K^T uses i8×i8→i32 CMMA with per-tile quantization
//! - P×V uses f16×f16→f32 CMMA (V stays float, not quantized)
//! - Scores are dequantized to f32 for softmax computation

mod attention;
pub mod setup;

pub use attention::*;
