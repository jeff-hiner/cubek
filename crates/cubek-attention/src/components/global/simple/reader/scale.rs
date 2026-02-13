//! Scale reader for INT8 CMMA attention with per-block quantization scales.
//!
//! Following SageAttention's approach and the MaskReader pattern, ScaleReader
//! provides a mechanism to thread quantization scales through the attention hierarchy.
//!
//! For INT8 CMMA attention:
//! - Q and K are pre-quantized to i8 with per-block scales
//! - Q scales shape: `[batch * heads * num_q_blocks]` flattened
//! - K scales shape: `[batch * heads * num_k_blocks]` flattened
//! - Block size typically equals tile/partition size (e.g., 64)
//! - The combined scale (q_scale * k_scale) is applied after CMMA to dequantize i32 results
//!
//! For non-INT8 attention, `ScaleReader::Uniform(1.0)` is used with no overhead.

use cubecl;
use cubecl::prelude::*;

/// Iterator for per-block scale values.
///
/// Stores the base offset and current block index for iteration.
#[derive(CubeType)]
pub struct PerBlockScaleIterator {
    /// Slice of scale values for this batch*head.
    scales: Slice<f32>,
    /// Current block index within this batch*head.
    current_block: RuntimeCell<u32>,
}

#[cube]
impl PerBlockScaleIterator {
    /// Create a new per-block scale iterator.
    pub fn new(scales: Slice<f32>, initial_block: u32) -> Self {
        PerBlockScaleIterator {
            scales,
            current_block: RuntimeCell::new(initial_block),
        }
    }

    /// Read the current scale value.
    pub fn read(&self) -> f32 {
        let idx = self.current_block.read();
        self.scales[idx as usize]
    }

    /// Advance to the next block's scale.
    pub fn advance(&mut self) {
        self.current_block.store(self.current_block.read() + 1);
    }
}

/// Reader for quantization scales in INT8 CMMA attention.
///
/// Follows the MaskReader pattern to provide scale data to the attention hierarchy.
/// Supports per-block scales that can advance during K/V iteration.
#[derive(CubeType)]
pub enum ScaleReader {
    /// Per-block scales from a pre-computed scale tensor.
    PerBlock(PerBlockScaleIterator),
    /// Uniform scale (1.0 for non-quantized attention).
    Uniform {
        /// The uniform scale value.
        scale: f32,
    },
}

#[cube]
impl ScaleReader {
    /// Create a uniform scale reader (for non-INT8 attention).
    pub fn new_uniform(scale: f32) -> Self {
        ScaleReader::new_Uniform(scale)
    }

    /// Create a per-block scale reader from a scale tensor slice.
    ///
    /// # Arguments
    /// * `scales` - Slice of scales for this batch*head (num_blocks elements)
    /// * `initial_block` - Initial block index (usually 0 for K, or based on stage_q_offset for Q)
    pub fn new_per_block(scales: Slice<f32>, initial_block: u32) -> Self {
        ScaleReader::new_PerBlock(PerBlockScaleIterator::new(scales, initial_block))
    }

    /// Read the current scale value.
    pub fn read(&self) -> f32 {
        match self {
            ScaleReader::PerBlock(iter) => iter.read(),
            ScaleReader::Uniform { scale } => *scale,
        }
    }

    /// Advance to the next block's scale.
    /// Called each K/V iteration to move to the next K block's scale.
    pub fn advance(&mut self) {
        match self {
            ScaleReader::PerBlock(iter) => iter.advance(),
            ScaleReader::Uniform { .. } => {
                // No-op for uniform scales
            }
        }
    }
}

/// Combined scale reader for both Q and K scales.
///
/// Q scale is read once (Q is loaded once per workgroup).
/// K scale advances each K/V iteration.
#[derive(CubeType)]
pub struct CombinedScaleReader {
    /// Query scale (read once, does not advance).
    q_scale: f32,
    /// Key scale reader (advances each K/V iteration).
    k_scale_reader: ScaleReader,
}

#[cube]
impl CombinedScaleReader {
    /// Create a new combined scale reader.
    ///
    /// # Arguments
    /// * `q_scale_reader` - Reader for Q scales (will be read once immediately)
    /// * `k_scale_reader` - Reader for K scales (will advance each iteration)
    pub fn new(q_scale_reader: ScaleReader, k_scale_reader: ScaleReader) -> Self {
        // Read Q scale once since Q doesn't change during iteration
        let q_scale = q_scale_reader.read();
        CombinedScaleReader { q_scale, k_scale_reader }
    }

    /// Create a uniform combined scale reader (both scales = 1.0).
    pub fn new_uniform() -> Self {
        CombinedScaleReader {
            q_scale: 1.0f32,
            k_scale_reader: ScaleReader::new_uniform(1.0f32),
        }
    }

    /// Read the combined scale (q_scale * current_k_scale).
    pub fn read_combined(&self) -> f32 {
        self.q_scale * self.k_scale_reader.read()
    }

    /// Advance the K scale reader to the next block.
    pub fn advance_k(&mut self) {
        self.k_scale_reader.advance();
    }
}
