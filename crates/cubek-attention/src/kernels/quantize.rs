//! INT8 quantization kernels for SageAttention-style pre-quantization.
//!
//! Following SageAttention's approach, Q and K are quantized to INT8 before the
//! attention kernel. This allows proper scale computation without the
//! complications of on-the-fly quantization during strided tile loading.
//!
//! Two quantization modes are provided:
//! - Per-row: Scale layout `[batch, heads, seq]` - one scale per row
//! - Per-head: Scale layout `[batch, heads]` - one scale per entire head (recommended)

use cubecl::ir::{ElemType, FloatKind, StorageType};
use cubecl::prelude::*;
use cubecl::{CubeCount, CubeDim, Runtime, client::ComputeClient};

/// Minimum scale value to avoid division by zero during quantization.
const MIN_QUANT_SCALE: f32 = 1e-6;

/// Per-block INT8 quantization kernel.
///
/// Quantizes a 4D tensor `[batch, heads, seq, dim]` to INT8 with per-block scales.
/// Each block spans one row of the last dimension (i.e., block_size = dim).
///
/// # Arguments
/// * `input` - Input tensor in f16/f32, shape `[batch, heads, seq, dim]`
/// * `output` - Output tensor in i8, same shape as input
/// * `scales` - Output scales tensor in f32, shape `[batch, heads, seq]`
/// * `dim` - The innermost dimension size (head_dim)
#[cube(launch, launch_unchecked)]
pub fn quantize_per_row<EI: Numeric>(
    input: &Tensor<Line<EI>>,
    output: &mut Tensor<Line<i8>>,
    scales: &mut Tensor<f32>,
    #[comptime] dim: u32,
    #[comptime] line_size: u32,
) {
    // Each workgroup handles one row (one block = one row of dim elements)
    // Workgroup ID encodes [batch, head, seq_idx]
    let row_idx = ABSOLUTE_POS;
    let total_rows = scales.len();

    if row_idx < total_rows {
        // Elements per row
        let elements_per_row = comptime!(dim as usize);
        let lines_per_row = comptime!(dim / line_size);

        // Base offset for this row in input/output tensors
        let row_offset = row_idx * elements_per_row / line_size as usize;

        // Phase 1: Find max absolute value in this row
        #[expect(clippy::manual_div_ceil, reason = "CubeCL macro doesn't support div_ceil")]
        let elements_per_unit = (lines_per_row + CUBE_DIM_X - 1) / CUBE_DIM_X;
        let mut local_max = 0.0f32;

        for i in 0..elements_per_unit {
            let line_idx = UNIT_POS_X + i * CUBE_DIM_X;
            if line_idx < lines_per_row {
                let line = input[row_offset + line_idx as usize];
                for j in 0..line_size {
                    let val = f32::cast_from(line[j as usize]);
                    local_max = f32::max(local_max, f32::abs(val));
                }
            }
        }

        // Reduce across workgroup to get row max
        let row_max = plane_max(local_max);

        // Compute scale: scale = max / 127 (for dequantization: original = quantized * scale)
        let scale = f32::max(row_max / 127.0f32, MIN_QUANT_SCALE);
        let inv_scale = 127.0f32 / f32::max(row_max, MIN_QUANT_SCALE);

        // One thread writes the scale
        if UNIT_POS_X == 0 {
            scales[row_idx] = scale;
        }

        // Phase 2: Quantize all elements using the computed scale
        for i in 0..elements_per_unit {
            let line_idx = UNIT_POS_X + i * CUBE_DIM_X;
            if line_idx < lines_per_row {
                let in_line = input[row_offset + line_idx as usize];
                let mut out_line = Line::<i8>::empty(line_size as usize);
                for j in 0..line_size {
                    let val = f32::cast_from(in_line[j as usize]);
                    let quantized = f32::round(val * inv_scale);
                    let clamped = f32::clamp(quantized, -127.0f32, 127.0f32);
                    out_line[j as usize] = i8::cast_from(clamped);
                }
                output[row_offset + line_idx as usize] = out_line;
            }
        }
    }
}

/// Per-head INT8 quantization kernel - Phase 1: Find max absolute value per head.
///
/// Each workgroup processes one head and computes the max abs across all seq×dim elements.
/// Uses shared memory reduction for efficiency.
///
/// # Arguments
/// * `input` - Input tensor in f16/f32, shape `[batch, heads, seq, dim]`
/// * `max_vals` - Output max absolute values per head, shape `[batch, heads]`
/// * `seq` - Sequence length
/// * `dim` - Head dimension
#[cube(launch, launch_unchecked)]
pub fn quantize_per_head_find_max<EI: Numeric>(
    input: &Tensor<Line<EI>>,
    max_vals: &mut Tensor<f32>,
    #[comptime] seq: u32,
    #[comptime] dim: u32,
    #[comptime] line_size: u32,
) {
    // Each workgroup handles one head
    // CUBE_POS gives us the head index (flattened batch*heads)
    let head_idx = CUBE_POS_X;
    let total_heads = max_vals.len() as u32;

    if head_idx < total_heads {
        // Elements per head = seq * dim
        let lines_per_head = comptime!(seq * dim / line_size);

        // Base offset for this head in input tensor
        let head_offset = head_idx as usize * lines_per_head as usize;

        // Each thread processes multiple lines
        #[expect(clippy::manual_div_ceil, reason = "CubeCL macro doesn't support div_ceil")]
        let lines_per_thread = (lines_per_head + CUBE_DIM_X - 1) / CUBE_DIM_X;
        let mut local_max = 0.0f32;

        for i in 0..lines_per_thread {
            let line_idx = UNIT_POS_X + i * CUBE_DIM_X;
            if line_idx < lines_per_head {
                let line = input[head_offset + line_idx as usize];
                for j in 0..line_size {
                    let val = f32::cast_from(line[j as usize]);
                    local_max = f32::max(local_max, f32::abs(val));
                }
            }
        }

        // Reduce across workgroup to get head max
        let head_max = plane_max(local_max);

        // One thread writes the max
        if UNIT_POS_X == 0 {
            max_vals[head_idx as usize] = head_max;
        }
    }
}

/// Per-head INT8 quantization kernel - Phase 2: Quantize using per-head scales.
///
/// Uses the pre-computed max values to quantize the entire tensor.
///
/// # Arguments
/// * `input` - Input tensor in f16/f32, shape `[batch, heads, seq, dim]`
/// * `output` - Output tensor in i8, same shape as input
/// * `max_vals` - Max absolute values per head, shape `[batch, heads]`
/// * `scales` - Output scales per head, shape `[batch, heads]`
/// * `seq` - Sequence length
/// * `dim` - Head dimension
#[cube(launch, launch_unchecked)]
pub fn quantize_per_head_apply<EI: Numeric>(
    input: &Tensor<Line<EI>>,
    output: &mut Tensor<Line<i8>>,
    max_vals: &Tensor<f32>,
    scales: &mut Tensor<f32>,
    #[comptime] seq: u32,
    #[comptime] dim: u32,
    #[comptime] line_size: u32,
) {
    // Each workgroup handles one head
    let head_idx = CUBE_POS_X;
    let total_heads = max_vals.len() as u32;

    if head_idx < total_heads {
        // Get the max for this head and compute scale
        let head_max = max_vals[head_idx as usize];
        let scale = f32::max(head_max / 127.0f32, MIN_QUANT_SCALE);
        let inv_scale = 127.0f32 / f32::max(head_max, MIN_QUANT_SCALE);

        // One thread writes the scale
        if UNIT_POS_X == 0 {
            scales[head_idx as usize] = scale;
        }

        // Elements per head
        let lines_per_head = comptime!(seq * dim / line_size);
        let head_offset = head_idx as usize * lines_per_head as usize;

        // Each thread processes multiple lines
        #[expect(clippy::manual_div_ceil, reason = "CubeCL macro doesn't support div_ceil")]
        let lines_per_thread = (lines_per_head + CUBE_DIM_X - 1) / CUBE_DIM_X;

        for i in 0..lines_per_thread {
            let line_idx = UNIT_POS_X + i * CUBE_DIM_X;
            if line_idx < lines_per_head {
                let in_line = input[head_offset + line_idx as usize];
                let mut out_line = Line::<i8>::empty(line_size as usize);
                for j in 0..line_size {
                    let val = f32::cast_from(in_line[j as usize]);
                    let quantized = f32::round(val * inv_scale);
                    let clamped = f32::clamp(quantized, -127.0f32, 127.0f32);
                    out_line[j as usize] = i8::cast_from(clamped);
                }
                output[head_offset + line_idx as usize] = out_line;
            }
        }
    }
}

/// Configuration for the quantization kernel.
#[derive(Debug, Clone)]
pub struct QuantizeConfig {
    /// The innermost dimension size (typically head_dim).
    pub dim: u32,
    /// Sequence length (needed for per-head/per-block quantization).
    pub seq: u32,
    /// Line size for vectorized loads.
    pub line_size: u32,
    /// Block size for per-block quantization (number of seq positions per block).
    pub block_size: u32,
}

/// Launch the per-head quantization kernel (recommended).
///
/// This uses two passes:
/// 1. Find max absolute value per head
/// 2. Quantize using the per-head scale
///
/// # Arguments
/// * `client` - The compute client
/// * `input` - Input tensor handle, shape `[batch, heads, seq, dim]`
/// * `output` - Output i8 tensor handle, same shape as input
/// * `scales` - Output f32 scales tensor, shape `[batch, heads]`
/// * `config` - Quantization configuration
pub fn launch_quantize_per_head<R: Runtime, EI: Numeric>(
    client: &ComputeClient<R>,
    input: &TensorHandleRef<R>,
    output: &TensorHandleRef<R>,
    scales: &TensorHandleRef<R>,
    config: &QuantizeConfig,
) {
    // Total number of heads = batch * heads
    let total_heads: usize = scales.shape.iter().product();

    // Allocate temporary buffer for max values
    let max_vals_handle = client.empty(total_heads * core::mem::size_of::<f32>());
    let max_vals = cubecl::std::tensor::TensorHandle::<R>::new(
        max_vals_handle,
        scales.shape.to_vec(),
        scales.strides.to_vec(),
        StorageType::Scalar(ElemType::Float(FloatKind::F32)),
    );

    // Use 32 threads per workgroup (one warp)
    let cube_dim = CubeDim::new_1d(32);

    // Phase 1: Find max per head
    unsafe {
        let _ = quantize_per_head_find_max::launch_unchecked::<EI, R>(
            client,
            CubeCount::new_1d(total_heads as u32),
            cube_dim,
            input.as_tensor_arg(config.line_size as usize),
            max_vals.as_ref().as_tensor_arg(1),
            config.seq,
            config.dim,
            config.line_size,
        );
    }

    // Phase 2: Quantize using per-head scales
    unsafe {
        let _ = quantize_per_head_apply::launch_unchecked::<EI, R>(
            client,
            CubeCount::new_1d(total_heads as u32),
            cube_dim,
            input.as_tensor_arg(config.line_size as usize),
            output.as_tensor_arg(config.line_size as usize),
            max_vals.as_ref().as_tensor_arg(1),
            scales.as_tensor_arg(1),
            config.seq,
            config.dim,
            config.line_size,
        );
    }
}

/// Launch the per-row quantization kernel (legacy).
///
/// # Arguments
/// * `client` - The compute client
/// * `input` - Input tensor handle, shape `[batch, heads, seq, dim]`
/// * `output` - Output i8 tensor handle, same shape as input
/// * `scales` - Output f32 scales tensor, shape `[batch, heads, seq]`
/// * `config` - Quantization configuration
#[allow(clippy::result_large_err, dead_code)]
pub fn launch_quantize_per_row<R: Runtime, EI: Numeric>(
    client: &ComputeClient<R>,
    input: &TensorHandleRef<R>,
    output: &TensorHandleRef<R>,
    scales: &TensorHandleRef<R>,
    config: &QuantizeConfig,
) {
    // Total number of rows = batch * heads * seq
    let total_rows: usize = scales.shape.iter().product();

    // Use 32 threads per workgroup (one warp)
    let cube_dim = CubeDim::new_1d(32);
    let cube_count = (total_rows as u32).div_ceil(32);

    unsafe {
        let _ = quantize_per_row::launch_unchecked::<EI, R>(
            client,
            CubeCount::new_1d(cube_count),
            cube_dim,
            input.as_tensor_arg(config.line_size as usize),
            output.as_tensor_arg(config.line_size as usize),
            scales.as_tensor_arg(1),
            config.dim,
            config.line_size,
        );
    }
}

/// Per-block INT8 quantization kernel - Phase 1: Find max absolute value per block.
///
/// Each workgroup processes one block (block_size rows) and computes the max abs
/// across all elements in that block.
///
/// # Arguments
/// * `input` - Input tensor in f16/f32, shape `[batch, heads, seq, dim]`
/// * `max_vals` - Output max absolute values per block, shape `[batch * heads * num_blocks]`
/// * `dim` - Head dimension
/// * `block_size` - Number of seq positions per block
#[cube(launch, launch_unchecked)]
pub fn quantize_per_block_find_max<EI: Numeric>(
    input: &Tensor<Line<EI>>,
    max_vals: &mut Tensor<f32>,
    #[comptime] dim: u32,
    #[comptime] block_size: u32,
    #[comptime] line_size: u32,
) {
    // Each workgroup handles one block
    // CUBE_POS_X gives us the block index (flattened batch*heads*block_idx)
    let block_idx = CUBE_POS_X;
    let total_blocks = max_vals.len() as u32;

    if block_idx < total_blocks {
        // Elements per block = block_size * dim
        let lines_per_block = comptime!(block_size * dim / line_size);

        // Base offset for this block in input tensor
        let block_offset = block_idx as usize * lines_per_block as usize;

        // Each thread processes multiple lines
        #[expect(clippy::manual_div_ceil, reason = "CubeCL macro doesn't support div_ceil")]
        let lines_per_thread = (lines_per_block + CUBE_DIM_X - 1) / CUBE_DIM_X;
        let mut local_max = 0.0f32;

        for i in 0..lines_per_thread {
            let line_idx = UNIT_POS_X + i * CUBE_DIM_X;
            if line_idx < lines_per_block {
                let line = input[block_offset + line_idx as usize];
                for j in 0..line_size {
                    let val = f32::cast_from(line[j as usize]);
                    local_max = f32::max(local_max, f32::abs(val));
                }
            }
        }

        // Reduce across workgroup to get block max
        let block_max = plane_max(local_max);

        // One thread writes the max
        if UNIT_POS_X == 0 {
            max_vals[block_idx as usize] = block_max;
        }
    }
}

/// Per-block INT8 quantization kernel - Phase 2: Quantize using per-block scales.
///
/// Uses the pre-computed max values to quantize the tensor with per-block scales.
///
/// # Arguments
/// * `input` - Input tensor in f16/f32, shape `[batch, heads, seq, dim]`
/// * `output` - Output tensor in i8, same shape as input
/// * `max_vals` - Max absolute values per block, shape `[batch * heads * num_blocks]`
/// * `scales` - Output scales per block, shape `[batch * heads * num_blocks]`
/// * `dim` - Head dimension
/// * `block_size` - Number of seq positions per block
#[cube(launch, launch_unchecked)]
pub fn quantize_per_block_apply<EI: Numeric>(
    input: &Tensor<Line<EI>>,
    output: &mut Tensor<Line<i8>>,
    max_vals: &Tensor<f32>,
    scales: &mut Tensor<f32>,
    #[comptime] dim: u32,
    #[comptime] block_size: u32,
    #[comptime] line_size: u32,
) {
    // Each workgroup handles one block
    let block_idx = CUBE_POS_X;
    let total_blocks = max_vals.len() as u32;

    if block_idx < total_blocks {
        // Get the max for this block and compute scale
        let block_max = max_vals[block_idx as usize];
        let scale = f32::max(block_max / 127.0f32, MIN_QUANT_SCALE);
        let inv_scale = 127.0f32 / f32::max(block_max, MIN_QUANT_SCALE);

        // One thread writes the scale
        if UNIT_POS_X == 0 {
            scales[block_idx as usize] = scale;
        }

        // Elements per block
        let lines_per_block = comptime!(block_size * dim / line_size);
        let block_offset = block_idx as usize * lines_per_block as usize;

        // Each thread processes multiple lines
        #[expect(clippy::manual_div_ceil, reason = "CubeCL macro doesn't support div_ceil")]
        let lines_per_thread = (lines_per_block + CUBE_DIM_X - 1) / CUBE_DIM_X;

        for i in 0..lines_per_thread {
            let line_idx = UNIT_POS_X + i * CUBE_DIM_X;
            if line_idx < lines_per_block {
                let in_line = input[block_offset + line_idx as usize];
                let mut out_line = Line::<i8>::empty(line_size as usize);
                for j in 0..line_size {
                    let val = f32::cast_from(in_line[j as usize]);
                    let quantized = f32::round(val * inv_scale);
                    let clamped = f32::clamp(quantized, -127.0f32, 127.0f32);
                    out_line[j as usize] = i8::cast_from(clamped);
                }
                output[block_offset + line_idx as usize] = out_line;
            }
        }
    }
}

/// Launch the per-block quantization kernel.
///
/// This uses two passes:
/// 1. Find max absolute value per block
/// 2. Quantize using the per-block scale
///
/// # Arguments
/// * `client` - The compute client
/// * `input` - Input tensor handle, shape `[batch, heads, seq, dim]`
/// * `output` - Output i8 tensor handle, same shape as input
/// * `scales` - Output f32 scales tensor, shape `[batch * heads * num_blocks]`
/// * `config` - Quantization configuration
pub fn launch_quantize_per_block<R: Runtime, EI: Numeric>(
    client: &ComputeClient<R>,
    input: &TensorHandleRef<R>,
    output: &TensorHandleRef<R>,
    scales: &TensorHandleRef<R>,
    config: &QuantizeConfig,
) {
    // Total number of blocks = batch * heads * num_blocks
    let total_blocks: usize = scales.shape.iter().product();

    // Allocate temporary buffer for max values
    let max_vals_handle = client.empty(total_blocks * core::mem::size_of::<f32>());
    let max_vals = cubecl::std::tensor::TensorHandle::<R>::new(
        max_vals_handle,
        scales.shape.to_vec(),
        scales.strides.to_vec(),
        StorageType::Scalar(ElemType::Float(FloatKind::F32)),
    );

    // Use 32 threads per workgroup (one warp)
    let cube_dim = CubeDim::new_1d(32);

    // Phase 1: Find max per block
    unsafe {
        let _ = quantize_per_block_find_max::launch_unchecked::<EI, R>(
            client,
            CubeCount::new_1d(total_blocks as u32),
            cube_dim,
            input.as_tensor_arg(config.line_size as usize),
            max_vals.as_ref().as_tensor_arg(1),
            config.dim,
            config.block_size,
            config.line_size,
        );
    }

    // Phase 2: Quantize using per-block scales
    unsafe {
        let _ = quantize_per_block_apply::launch_unchecked::<EI, R>(
            client,
            CubeCount::new_1d(total_blocks as u32),
            cube_dim,
            input.as_tensor_arg(config.line_size as usize),
            output.as_tensor_arg(config.line_size as usize),
            max_vals.as_ref().as_tensor_arg(1),
            scales.as_tensor_arg(1),
            config.dim,
            config.block_size,
            config.line_size,
        );
    }
}
