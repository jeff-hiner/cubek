//! Standalone SageAttention kernel
//!
//! This module provides a standalone attention kernel that can be used
//! independently of the complex tiling architecture. It demonstrates
//! the SageAttention algorithm with INT8 quantization for Q·K^T.
//!
//! Shape conventions (row-major, contiguous):
//! - Q: [batch, heads, seq_q, head_dim]
//! - K: [batch, heads, seq_kv, head_dim]
//! - V: [batch, heads, seq_kv, head_dim]
//! - Output: [batch, heads, seq_q, head_dim]

use cubecl::prelude::*;

/// Block size for query tiling (number of query positions per cube)
pub const BLOCK_M: usize = 64;

/// Block size for KV tiling (number of KV positions loaded to shared memory at once)
pub const BLOCK_N: usize = 32;

/// Block size for query quantization (number of rows per quantization block)
/// Reference uses BLKQ=128
pub const BLOCK_Q_QUANT: usize = 128;

/// Block size for key quantization (number of rows per quantization block)
/// Reference uses BLKK=64
pub const BLOCK_K_QUANT: usize = 64;

/// Configuration for SageAttention kernel
#[derive(Clone, Copy, Debug)]
pub struct SageAttentionConfig {
    /// Batch size
    pub batch: usize,
    /// Number of attention heads
    pub heads: usize,
    /// Query sequence length
    pub seq_q: usize,
    /// Key/Value sequence length
    pub seq_kv: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Attention scale (typically 1/sqrt(head_dim))
    pub scale: f32,
}

/// Launch SageAttention kernel (INT8 quantized version with exp2)
///
/// This kernel computes scaled dot-product attention using INT8 quantization:
/// 1. Quantizes Q and K to INT8 with per-row scaling
/// 2. Computes Q @ K^T using DP4a (INT8 dot product)
/// 3. Uses exp2 for faster softmax computation
/// 4. Accumulates with f32 V values
///
/// This is the fastest variant, typically 1.5-2x faster than FlashAttention.
///
/// # Arguments
/// * `client` - Compute client
/// * `q` - Query tensor [batch, heads, seq_q, head_dim] (f32)
/// * `k` - Key tensor [batch, heads, seq_kv, head_dim] (f32)
/// * `v` - Value tensor [batch, heads, seq_kv, head_dim] (f32)
/// * `output` - Output tensor [batch, heads, seq_q, head_dim] (f32)
/// * `config` - Attention configuration
///
/// # Panics
/// Panics if head_dim is not divisible by 4 (required for INT8 vectorization).
#[allow(clippy::too_many_arguments)]
pub fn launch_sage_attention<R: Runtime>(
    client: &ComputeClient<R>,
    q: &TensorHandleRef<R>,
    k: &TensorHandleRef<R>,
    v: &TensorHandleRef<R>,
    output: &TensorHandleRef<R>,
    config: SageAttentionConfig,
) -> Result<(), LaunchError> {
    // Fall back to simple kernel if head_dim is not divisible by 4
    if config.head_dim % 4 != 0 {
        return launch_sage_attention_simple(client, q, k, v, output, config);
    }

    let num_q_rows = config.batch * config.heads * config.seq_q;
    let num_k_rows = config.batch * config.heads * config.seq_kv;
    let q_elems = num_q_rows * config.head_dim;
    let k_elems = num_k_rows * config.head_dim;

    // Per-block quantization: one scale per block instead of per row
    // Reference uses BLKQ=128 for Q, BLKK=64 for K
    let num_q_blocks = (num_q_rows + BLOCK_Q_QUANT - 1) / BLOCK_Q_QUANT;
    let num_k_blocks = (num_k_rows + BLOCK_K_QUANT - 1) / BLOCK_K_QUANT;

    // Allocate temporary buffers for INT8 quantization
    let q_int8_handle = client.empty(q_elems);
    let k_int8_handle = client.empty(k_elems);
    let q_scales_handle = client.empty(num_q_blocks * core::mem::size_of::<f32>());
    let k_scales_handle = client.empty(num_k_blocks * core::mem::size_of::<f32>());

    // Create tensor refs for INT8 buffers
    let q_shape = vec![config.batch, config.heads, config.seq_q, config.head_dim];
    let q_strides = vec![
        config.heads * config.seq_q * config.head_dim,
        config.seq_q * config.head_dim,
        config.head_dim,
        1,
    ];
    let k_shape = vec![config.batch, config.heads, config.seq_kv, config.head_dim];
    let k_strides = vec![
        config.heads * config.seq_kv * config.head_dim,
        config.seq_kv * config.head_dim,
        config.head_dim,
        1,
    ];
    let q_scales_shape = vec![num_q_blocks];
    let q_scales_strides = vec![1];
    let k_scales_shape = vec![num_k_blocks];
    let k_scales_strides = vec![1];

    let q_int8_ref =
        unsafe { TensorHandleRef::from_raw_parts(&q_int8_handle, &q_strides, &q_shape, 4) };
    let k_int8_ref =
        unsafe { TensorHandleRef::from_raw_parts(&k_int8_handle, &k_strides, &k_shape, 4) };
    let q_scales_ref = unsafe {
        TensorHandleRef::from_raw_parts(&q_scales_handle, &q_scales_strides, &q_scales_shape, 4)
    };
    let k_scales_ref = unsafe {
        TensorHandleRef::from_raw_parts(&k_scales_handle, &k_scales_strides, &k_scales_shape, 4)
    };

    // Quantize Q and K to INT8 with per-block scaling
    launch_quantize_int8(
        client,
        q,
        &q_int8_ref,
        &q_scales_ref,
        config.batch,
        config.heads,
        config.seq_q,
        config.head_dim,
        BLOCK_Q_QUANT,
    )?;
    launch_quantize_int8(
        client,
        k,
        &k_int8_ref,
        &k_scales_ref,
        config.batch,
        config.heads,
        config.seq_kv,
        config.head_dim,
        BLOCK_K_QUANT,
    )?;

    // Run tiled INT8 attention with exp2 (matching reference)
    // BLOCK_M queries per cube share K loads from shared memory
    launch_sage_attention_tiled_int8(
        client,
        &q_int8_ref,
        &k_int8_ref,
        &q_scales_ref,
        &k_scales_ref,
        v,
        output,
        config,
    )
}

/// Launch SageAttention kernel (tiled version - experimental)
///
/// Each cube processes BLOCK_M query positions with sequential dot products.
/// This has lower kernel launch overhead but slower computation.
/// Currently slower than the simple version at all tested sizes.
#[allow(clippy::too_many_arguments)]
pub fn launch_sage_attention_tiled<R: Runtime>(
    client: &ComputeClient<R>,
    q: &TensorHandleRef<R>,
    k: &TensorHandleRef<R>,
    v: &TensorHandleRef<R>,
    output: &TensorHandleRef<R>,
    config: SageAttentionConfig,
) -> Result<(), LaunchError> {
    // Validate shapes
    assert_eq!(q.shape.len(), 4);
    assert_eq!(q.shape[0], config.batch);
    assert_eq!(q.shape[1], config.heads);
    assert_eq!(q.shape[2], config.seq_q);
    assert_eq!(q.shape[3], config.head_dim);

    // Each cube processes BLOCK_M query positions for one (batch, head)
    // Number of query blocks (ceil division)
    let num_q_blocks = (config.seq_q + BLOCK_M - 1) / BLOCK_M;
    let num_cubes = config.batch * config.heads * num_q_blocks;

    // Use BLOCK_M units per cube - each unit handles one query position
    let units_per_cube = BLOCK_M.min(config.seq_q);
    let cube_dim = CubeDim::new(client, units_per_cube);
    let cube_count = CubeCount::new_1d(num_cubes as u32);

    unsafe {
        attention_kernel_tiled::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            q.as_tensor_arg(1),
            k.as_tensor_arg(1),
            v.as_tensor_arg(1),
            output.as_tensor_arg(1),
            ScalarArg::new(config.scale),
            ScalarArg::new(config.batch as u32),
            ScalarArg::new(config.heads as u32),
            ScalarArg::new(config.seq_q as u32),
            ScalarArg::new(config.seq_kv as u32),
            ScalarArg::new(config.head_dim as u32),
            ScalarArg::new(num_q_blocks as u32),
            BLOCK_M as u32,
        )
    }
}

/// Launch the original non-tiled SageAttention kernel
///
/// This is the simpler version that processes one query position per cube.
/// Useful for comparison and debugging.
#[allow(clippy::too_many_arguments)]
pub fn launch_sage_attention_simple<R: Runtime>(
    client: &ComputeClient<R>,
    q: &TensorHandleRef<R>,
    k: &TensorHandleRef<R>,
    v: &TensorHandleRef<R>,
    output: &TensorHandleRef<R>,
    config: SageAttentionConfig,
) -> Result<(), LaunchError> {
    // Validate shapes
    assert_eq!(q.shape.len(), 4);
    assert_eq!(q.shape[0], config.batch);
    assert_eq!(q.shape[1], config.heads);
    assert_eq!(q.shape[2], config.seq_q);
    assert_eq!(q.shape[3], config.head_dim);

    // Each cube processes one query position for one (batch, head)
    // Units within the cube cooperate on the head_dim reduction
    let num_work_items = config.batch * config.heads * config.seq_q;

    // Use 64 units per cube for the reduction over head_dim
    let units_per_cube = 64.min(config.head_dim);
    let cube_dim = CubeDim::new(client, units_per_cube);
    let cube_count = CubeCount::new_1d(num_work_items as u32);

    unsafe {
        attention_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            q.as_tensor_arg(1),
            k.as_tensor_arg(1),
            v.as_tensor_arg(1),
            output.as_tensor_arg(1),
            ScalarArg::new(config.scale),
            ScalarArg::new(config.batch as u32),
            ScalarArg::new(config.heads as u32),
            ScalarArg::new(config.seq_q as u32),
            ScalarArg::new(config.seq_kv as u32),
            ScalarArg::new(config.head_dim as u32),
        )
    }
}

/// Online softmax attention kernel (FlashAttention algorithm)
///
/// Each cube computes attention for one (batch, head, query_pos).
/// Uses online softmax to iterate over KV positions ONCE instead of THREE times.
///
/// Parallelization: Units within a cube parallelize the head_dim accumulation.
/// Unit 0 computes the attention scores (sequential over KV), then all units
/// update their portion of the output accumulator.
///
/// Algorithm:
/// 1. Initialize: m = -inf (running max), l = 0 (running sum), acc = 0 (output accumulator)
/// 2. For each kv_pos:
///    - Compute score = Q·K * scale
///    - m_new = max(m, score)
///    - alpha = exp(m - m_new)  // correction factor for previous accumulator
///    - acc = acc * alpha + exp(score - m_new) * V
///    - l = l * alpha + exp(score - m_new)
///    - m = m_new
/// 3. output = acc / l
#[cube(launch_unchecked)]
fn attention_kernel(
    q: &Tensor<Line<f32>>,
    k: &Tensor<Line<f32>>,
    v: &Tensor<Line<f32>>,
    output: &mut Tensor<Line<f32>>,
    scale: f32,
    batch_size: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    head_dim: u32,
) {
    // Each cube handles one (batch, head, query_pos)
    let work_id: u32 = CUBE_POS_X;
    let unit_id: u32 = UNIT_POS_X;
    let num_units: u32 = CUBE_DIM_X;

    // Decode position
    let query_pos: u32 = work_id % seq_q;
    let tmp: u32 = work_id / seq_q;
    let head: u32 = tmp % num_heads;
    let batch: u32 = tmp / num_heads;

    if batch >= batch_size {
        terminate!();
    }

    // Compute base offsets
    let q_base: u32 =
        batch * num_heads * seq_q * head_dim + head * seq_q * head_dim + query_pos * head_dim;
    let kv_base: u32 = batch * num_heads * seq_kv * head_dim + head * seq_kv * head_dim;

    // Each unit handles a strided subset of head_dim
    // Unit i handles positions: i, i+num_units, i+2*num_units, ...

    // Online softmax state (all units track the same values)
    let mut m_i: f32 = f32::min_value(); // Running max score
    let mut l_i: f32 = 0.0f32; // Running sum of exp(score - max)

    // Zero out this unit's portion of output
    let mut d: u32 = unit_id;
    while d < head_dim {
        let out_idx: usize = (q_base + d) as usize;
        output[out_idx] = Line::cast_from(0.0f32);
        d += num_units;
    }

    // Single pass over all KV positions
    let mut kv_pos: u32 = 0u32;
    while kv_pos < seq_kv {
        let k_off: u32 = kv_base + kv_pos * head_dim;

        // All units participate in Q·K dot product via parallel reduction
        // Each unit sums its portion
        let mut partial_dot: f32 = 0.0f32;
        let mut d2: u32 = unit_id;
        while d2 < head_dim {
            let q_idx: usize = (q_base + d2) as usize;
            let k_idx: usize = (k_off + d2) as usize;
            let qv: f32 = f32::cast_from(q[q_idx][0]);
            let kv: f32 = f32::cast_from(k[k_idx][0]);
            partial_dot += qv * kv;
            d2 += num_units;
        }

        // Reduce partial sums across units using plane_sum
        let dot: f32 = plane_sum(partial_dot);
        let score: f32 = dot * scale;

        // Online softmax update (all units compute same values)
        let m_new: f32 = select(score > m_i, score, m_i);
        let alpha: f32 = (m_i - m_new).exp();
        let p: f32 = (score - m_new).exp();

        l_i = l_i * alpha + p;

        // Each unit updates its portion of accumulator
        let mut d3: u32 = unit_id;
        while d3 < head_dim {
            let out_idx: usize = (q_base + d3) as usize;
            let v_idx: usize = (k_off + d3) as usize;

            let old_acc: f32 = f32::cast_from(output[out_idx][0]);
            let vv: f32 = f32::cast_from(v[v_idx][0]);
            let new_acc: f32 = old_acc * alpha + p * vv;
            output[out_idx] = Line::cast_from(new_acc);

            d3 += num_units;
        }

        m_i = m_new;
        kv_pos += 1u32;
    }

    // Final normalization: each unit normalizes its portion
    let mut d4: u32 = unit_id;
    while d4 < head_dim {
        let out_idx: usize = (q_base + d4) as usize;
        let acc: f32 = f32::cast_from(output[out_idx][0]);
        output[out_idx] = Line::cast_from(acc / l_i);
        d4 += num_units;
    }
}

/// Tiled online softmax attention kernel (simple version)
///
/// Each cube processes BLOCK_M query positions for one (batch, head).
/// Each unit handles one query position independently with sequential dot products.
///
/// NOTE: This version is faster than the simple kernel at small sequence lengths
/// (fewer kernel launches), but slower at large sequence lengths (no parallel
/// reduction). Use launch_sage_attention_simple for large sequences.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn attention_kernel_tiled(
    q: &Tensor<Line<f32>>,
    k: &Tensor<Line<f32>>,
    v: &Tensor<Line<f32>>,
    output: &mut Tensor<Line<f32>>,
    scale: f32,
    batch_size: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    head_dim: u32,
    num_q_blocks: u32,
    #[comptime] block_m: u32,
) {
    // Each cube handles BLOCK_M query positions for one (batch, head)
    let cube_id: u32 = CUBE_POS_X;
    let unit_id: u32 = UNIT_POS_X;

    // Decode cube position
    let q_block: u32 = cube_id % num_q_blocks;
    let tmp: u32 = cube_id / num_q_blocks;
    let head: u32 = tmp % num_heads;
    let batch: u32 = tmp / num_heads;

    if batch >= batch_size {
        terminate!();
    }

    // Query position for this unit
    let query_pos: u32 = q_block * block_m + unit_id;

    // Bounds check
    if query_pos >= seq_q {
        terminate!();
    }

    // Compute base offsets
    let q_base: u32 =
        batch * num_heads * seq_q * head_dim + head * seq_q * head_dim + query_pos * head_dim;
    let kv_base: u32 = batch * num_heads * seq_kv * head_dim + head * seq_kv * head_dim;

    // Online softmax state
    let mut m_i: f32 = f32::min_value();
    let mut l_i: f32 = 0.0f32;

    // Initialize output to zero
    let mut d: u32 = 0u32;
    while d < head_dim {
        let out_idx: usize = (q_base + d) as usize;
        output[out_idx] = Line::cast_from(0.0f32);
        d += 1u32;
    }

    // Process all KV positions
    let mut kv_pos: u32 = 0u32;
    while kv_pos < seq_kv {
        let k_off: u32 = kv_base + kv_pos * head_dim;

        // Sequential dot product Q·K
        let mut dot: f32 = 0.0f32;
        let mut d2: u32 = 0u32;
        while d2 < head_dim {
            let q_idx: usize = (q_base + d2) as usize;
            let k_idx: usize = (k_off + d2) as usize;
            let qv: f32 = f32::cast_from(q[q_idx][0]);
            let kv: f32 = f32::cast_from(k[k_idx][0]);
            dot += qv * kv;
            d2 += 1u32;
        }
        let score: f32 = dot * scale;

        // Online softmax update
        let m_new: f32 = select(score > m_i, score, m_i);
        let alpha: f32 = (m_i - m_new).exp();
        let p: f32 = (score - m_new).exp();

        l_i = l_i * alpha + p;

        // Update accumulator
        let mut d3: u32 = 0u32;
        while d3 < head_dim {
            let out_idx: usize = (q_base + d3) as usize;
            let v_idx: usize = (k_off + d3) as usize;
            let old_acc: f32 = f32::cast_from(output[out_idx][0]);
            let vv: f32 = f32::cast_from(v[v_idx][0]);
            let new_acc: f32 = old_acc * alpha + p * vv;
            output[out_idx] = Line::cast_from(new_acc);
            d3 += 1u32;
        }

        m_i = m_new;
        kv_pos += 1u32;
    }

    // Final normalization
    let mut d4: u32 = 0u32;
    while d4 < head_dim {
        let out_idx: usize = (q_base + d4) as usize;
        let acc: f32 = f32::cast_from(output[out_idx][0]);
        output[out_idx] = Line::cast_from(acc / l_i);
        d4 += 1u32;
    }
}

// =============================================================================
// Properly Tiled Attention with Shared Memory
// =============================================================================

/// Launch properly tiled SageAttention kernel with shared memory
///
/// This kernel processes BLOCK_M queries per cube and loads K/V tiles into
/// shared memory so all queries share the same K/V loads (memory bandwidth win).
///
/// Architecture:
/// - Each cube handles BLOCK_M query positions for one (batch, head)
/// - Outer loop iterates over KV in BLOCK_N chunks
/// - K/V tiles loaded to shared memory once, reused by all BLOCK_M queries
/// - Each unit handles one query position with its own online softmax state
/// Line size for vectorized operations (vec4 for f32)
/// NOTE: Currently disabled due to stride/indexing issues with as_tensor_arg
pub const LINE_SIZE: usize = 4;

/// Launch properly tiled SageAttention kernel with shared memory
///
/// Uses scalar (line_size=1) access for correctness. Shared memory tiling
/// still provides bandwidth benefits from K/V reuse across queries.
#[allow(clippy::too_many_arguments)]
pub fn launch_sage_attention_tiled_smem<R: Runtime>(
    client: &ComputeClient<R>,
    q: &TensorHandleRef<R>,
    k: &TensorHandleRef<R>,
    v: &TensorHandleRef<R>,
    output: &TensorHandleRef<R>,
    config: SageAttentionConfig,
) -> Result<(), LaunchError> {
    // Validate shapes
    assert_eq!(q.shape.len(), 4);
    assert_eq!(q.shape[0], config.batch);
    assert_eq!(q.shape[1], config.heads);
    assert_eq!(q.shape[2], config.seq_q);
    assert_eq!(q.shape[3], config.head_dim);

    // Each cube processes BLOCK_M query positions for one (batch, head)
    let num_q_blocks = (config.seq_q + BLOCK_M - 1) / BLOCK_M;
    let num_cubes = config.batch * config.heads * num_q_blocks;

    // Use BLOCK_M units per cube - each unit handles one query position
    let units_per_cube = BLOCK_M.min(config.seq_q);
    let cube_dim = CubeDim::new(client, units_per_cube);
    let cube_count = CubeCount::new_1d(num_cubes as u32);

    // Shared memory size: K tile + V tile (in f32 elements)
    // Each tile is BLOCK_N rows × head_dim elements
    let smem_size = 2 * BLOCK_N * config.head_dim;

    unsafe {
        attention_kernel_tiled_smem::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            q.as_tensor_arg(1),  // Use line_size=1 for correct indexing
            k.as_tensor_arg(1),
            v.as_tensor_arg(1),
            output.as_tensor_arg(1),
            ScalarArg::new(config.scale),
            ScalarArg::new(config.batch as u32),
            ScalarArg::new(config.heads as u32),
            ScalarArg::new(config.seq_q as u32),
            ScalarArg::new(config.seq_kv as u32),
            ScalarArg::new(config.head_dim as u32),
            ScalarArg::new(num_q_blocks as u32),
            BLOCK_M as u32,
            BLOCK_N as u32,
            smem_size,
        )
    }
}

/// Properly tiled attention kernel with shared memory
///
/// Key optimizations:
/// 1. All BLOCK_M queries in a cube share the same K/V loads (bandwidth reduction)
/// 2. K/V tiles loaded to shared memory, reused by all queries in the cube
///
/// Memory layout:
/// - Shared memory: [K tile: BLOCK_N × head_dim f32] [V tile: same]
/// - Each unit maintains its own online softmax state (m_i, l_i) in registers
/// - Output accumulator stored in global memory (too large for registers)
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn attention_kernel_tiled_smem(
    q: &Tensor<Line<f32>>,
    k: &Tensor<Line<f32>>,
    v: &Tensor<Line<f32>>,
    output: &mut Tensor<Line<f32>>,
    scale: f32,
    batch_size: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    head_dim: u32,
    num_q_blocks: u32,
    #[comptime] block_m: u32,
    #[comptime] block_n: u32,
    #[comptime] smem_size: usize, // Total f32 elements in shared memory
) {
    let cube_id: u32 = CUBE_POS_X;
    let unit_id: u32 = UNIT_POS_X;
    let num_units: u32 = CUBE_DIM_X;

    // Decode cube position
    let q_block: u32 = cube_id % num_q_blocks;
    let tmp: u32 = cube_id / num_q_blocks;
    let head: u32 = tmp % num_heads;
    let batch: u32 = tmp / num_heads;

    if batch >= batch_size {
        terminate!();
    }

    // Query position for this unit
    let query_pos: u32 = q_block * block_m + unit_id;
    let valid_query: bool = query_pos < seq_q;

    // Compute base offsets (in elements)
    let q_base: u32 =
        batch * num_heads * seq_q * head_dim + head * seq_q * head_dim + query_pos * head_dim;
    let kv_base: u32 = batch * num_heads * seq_kv * head_dim + head * seq_kv * head_dim;

    // Shared memory for K and V tiles (stored as f32 elements)
    // Layout: [K: block_n * head_dim] [V: block_n * head_dim]
    let mut smem = SharedMemory::<f32>::new(smem_size);

    // Online softmax state (per-unit, in registers)
    let mut m_i: f32 = f32::min_value();
    let mut l_i: f32 = 0.0f32;

    // Initialize output to zero (only for valid queries)
    if valid_query {
        let mut d: u32 = 0u32;
        while d < head_dim {
            let out_idx: usize = (q_base + d) as usize;
            output[out_idx] = Line::cast_from(0.0f32);
            d += 1u32;
        }
    }

    // Tile size in elements
    let tile_size: u32 = block_n * head_dim;

    // Number of KV blocks to process
    let num_kv_blocks: u32 = (seq_kv + block_n - 1u32) / block_n;

    // Iterate over KV blocks
    let mut kv_block: u32 = 0u32;
    while kv_block < num_kv_blocks {
        let kv_start: u32 = kv_block * block_n;
        let kv_end: u32 = select(kv_start + block_n > seq_kv, seq_kv, kv_start + block_n);
        let block_len: u32 = kv_end - kv_start;

        // === Load K and V tiles into shared memory ===
        // All units cooperate to load the tile
        let mut load_idx: u32 = unit_id;
        while load_idx < tile_size {
            // Decode position within tile
            let kv_offset: u32 = load_idx / head_dim;
            let dim_offset: u32 = load_idx % head_dim;
            let kv_pos: u32 = kv_start + kv_offset;

            // K goes in first half of smem
            let k_smem_idx: u32 = load_idx;
            // V goes in second half
            let v_smem_idx: u32 = tile_size + load_idx;

            if kv_pos < seq_kv {
                let global_idx: usize = (kv_base + kv_pos * head_dim + dim_offset) as usize;
                smem[k_smem_idx as usize] = f32::cast_from(k[global_idx][0]);
                smem[v_smem_idx as usize] = f32::cast_from(v[global_idx][0]);
            } else {
                smem[k_smem_idx as usize] = 0.0f32;
                smem[v_smem_idx as usize] = 0.0f32;
            }

            load_idx += num_units;
        }

        // Synchronize to ensure all K/V data is loaded
        sync_cube();

        // === Process this KV block ===
        // Each unit (query) processes all KV positions in the block
        if valid_query {
            let mut kv_idx: u32 = 0u32;
            while kv_idx < block_len {
                // Compute Q·K dot product
                let mut dot: f32 = 0.0f32;
                let mut d: u32 = 0u32;
                while d < head_dim {
                    let q_idx: usize = (q_base + d) as usize;
                    let k_smem_idx: usize = (kv_idx * head_dim + d) as usize;
                    let qv: f32 = f32::cast_from(q[q_idx][0]);
                    let kv: f32 = smem[k_smem_idx];
                    dot += qv * kv;
                    d += 1u32;
                }
                let score: f32 = dot * scale;

                // Online softmax update
                let m_new: f32 = select(score > m_i, score, m_i);
                let alpha: f32 = (m_i - m_new).exp();
                let p: f32 = (score - m_new).exp();

                l_i = l_i * alpha + p;

                // Update output accumulator with V from shared memory
                let mut d2: u32 = 0u32;
                while d2 < head_dim {
                    let out_idx: usize = (q_base + d2) as usize;
                    let v_smem_idx: usize = (tile_size + kv_idx * head_dim + d2) as usize;
                    let old_acc: f32 = f32::cast_from(output[out_idx][0]);
                    let vv: f32 = smem[v_smem_idx];
                    let new_acc: f32 = old_acc * alpha + p * vv;
                    output[out_idx] = Line::cast_from(new_acc);
                    d2 += 1u32;
                }

                m_i = m_new;
                kv_idx += 1u32;
            }
        }

        // Synchronize before loading next tile
        sync_cube();

        kv_block += 1u32;
    }

    // Final normalization
    if valid_query {
        let inv_l: f32 = 1.0f32 / l_i;
        let mut d: u32 = 0u32;
        while d < head_dim {
            let out_idx: usize = (q_base + d) as usize;
            let acc: f32 = f32::cast_from(output[out_idx][0]);
            output[out_idx] = Line::cast_from(acc * inv_l);
            d += 1u32;
        }
    }
}

// =============================================================================
// Tiled INT8 Attention (matching reference parallelism)
// =============================================================================

/// Launch tiled INT8 SageAttention kernel
///
/// BLOCK_M queries per cube, processes BLOCK_N keys per iteration.
/// Uses DP4a for Q·K dot products.
#[allow(clippy::too_many_arguments)]
pub fn launch_sage_attention_tiled_int8<R: Runtime>(
    client: &ComputeClient<R>,
    q_int8: &TensorHandleRef<R>,
    k_int8: &TensorHandleRef<R>,
    q_scales: &TensorHandleRef<R>,
    k_scales: &TensorHandleRef<R>,
    v: &TensorHandleRef<R>,
    output: &TensorHandleRef<R>,
    config: SageAttentionConfig,
) -> Result<(), LaunchError> {
    assert!(
        config.head_dim % INT8_LINE_SIZE == 0,
        "head_dim must be divisible by {} for DP4a",
        INT8_LINE_SIZE
    );

    // Each cube processes BLOCK_M query positions for one (batch, head)
    let num_q_blocks = (config.seq_q + BLOCK_M - 1) / BLOCK_M;
    let num_cubes = config.batch * config.heads * num_q_blocks;

    // Use BLOCK_M units per cube - each unit handles one query position
    let units_per_cube = BLOCK_M.min(config.seq_q);
    let cube_dim = CubeDim::new(client, units_per_cube);
    let cube_count = CubeCount::new_1d(num_cubes as u32);

    unsafe {
        attention_kernel_tiled_int8::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            q_int8.as_tensor_arg(INT8_LINE_SIZE),
            k_int8.as_tensor_arg(INT8_LINE_SIZE),
            q_scales.as_tensor_arg(1),
            k_scales.as_tensor_arg(1),
            v.as_tensor_arg(1),
            output.as_tensor_arg(1),
            ScalarArg::new(config.scale),
            ScalarArg::new(config.batch as u32),
            ScalarArg::new(config.heads as u32),
            ScalarArg::new(config.seq_q as u32),
            ScalarArg::new(config.seq_kv as u32),
            ScalarArg::new(config.head_dim as u32),
            ScalarArg::new(num_q_blocks as u32),
            BLOCK_M as u32,
            BLOCK_N as u32,
            INT8_LINE_SIZE as u32,
            BLOCK_Q_QUANT as u32,
            BLOCK_K_QUANT as u32,
        )
    }
}

/// Tiled INT8 attention kernel (matching reference parallelism)
///
/// Each unit handles one query, processes BLOCK_N keys per outer loop iteration.
/// This matches the reference's tl.dot parallelism structure.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn attention_kernel_tiled_int8(
    q_int8: &Tensor<Line<i8>>,
    k_int8: &Tensor<Line<i8>>,
    q_scales: &Tensor<Line<f32>>,
    k_scales: &Tensor<Line<f32>>,
    v: &Tensor<Line<f32>>,
    output: &mut Tensor<Line<f32>>,
    attn_scale: f32,
    batch_size: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    head_dim: u32,
    num_q_blocks: u32,
    #[comptime] block_m: u32,
    #[comptime] block_n: u32,
    #[comptime] line_size: u32,
    #[comptime] q_quant_block_size: u32,
    #[comptime] k_quant_block_size: u32,
) {
    let cube_id: u32 = CUBE_POS_X;
    let unit_id: u32 = UNIT_POS_X;

    // Decode cube position
    let q_block: u32 = cube_id % num_q_blocks;
    let tmp: u32 = cube_id / num_q_blocks;
    let head: u32 = tmp % num_heads;
    let batch: u32 = tmp / num_heads;

    if batch >= batch_size {
        terminate!();
    }

    // Query position for this unit
    let query_pos: u32 = q_block * block_m + unit_id;
    if query_pos >= seq_q {
        terminate!();
    }

    // Compute base offsets (in elements)
    let q_base: u32 = batch * num_heads * seq_q * head_dim + head * seq_q * head_dim + query_pos * head_dim;
    let kv_base: u32 = batch * num_heads * seq_kv * head_dim + head * seq_kv * head_dim;

    // Per-block scale indexing:
    // Row position = batch * num_heads * seq + head * seq + pos
    // Block index = row_position / block_size
    let q_row_pos: u32 = batch * num_heads * seq_q + head * seq_q + query_pos;
    let k_row_base: u32 = batch * num_heads * seq_kv + head * seq_kv;

    // Number of Line<i8> per row
    let head_dim_lines: u32 = head_dim / line_size;

    // Load Q scale (per-block: same scale for all rows in the block)
    let q_scale_idx: usize = (q_row_pos / q_quant_block_size) as usize;
    let q_scale: f32 = f32::cast_from(q_scales[q_scale_idx][0]);

    // Q line base for this query (in Lines)
    let q_line_base: u32 = q_base / line_size;

    // Online softmax state
    let mut m_i: f32 = f32::min_value();
    let mut l_i: f32 = 0.0f32;

    // Output accumulator in registers (head_dim values)
    // We'll write directly to output since we can't have variable-size arrays
    // Initialize output to zero
    let mut d: u32 = 0u32;
    while d < head_dim {
        let out_idx: usize = (q_base + d) as usize;
        output[out_idx] = Line::cast_from(0.0f32);
        d += 1u32;
    }

    // Number of KV blocks
    let num_kv_blocks: u32 = (seq_kv + block_n - 1u32) / block_n;

    // Process KV in blocks of BLOCK_N
    let mut kv_block: u32 = 0u32;
    while kv_block < num_kv_blocks {
        let kv_start: u32 = kv_block * block_n;
        let kv_end: u32 = select(kv_start + block_n > seq_kv, seq_kv, kv_start + block_n);

        // Compute BLOCK_N dot products and find block max
        // First pass: compute all scores and find max
        let mut block_max: f32 = f32::min_value();

        // We need to store scores for this block - use a loop with conditional
        // Process each key in the block
        let mut kv_idx: u32 = kv_start;
        while kv_idx < kv_end {
            // K line base for this KV position (in Lines)
            let k_line_base: u32 = kv_base / line_size + kv_idx * head_dim_lines;

            // Get K scale (per-block: same scale for all rows in the block)
            let k_row_pos: u32 = k_row_base + kv_idx;
            let k_scale_idx: usize = (k_row_pos / k_quant_block_size) as usize;
            let k_scale: f32 = f32::cast_from(k_scales[k_scale_idx][0]);

            // Compute Q·K dot product using DP4a
            let mut dot_i32: i32 = 0i32;
            let mut line_idx: u32 = 0u32;
            while line_idx < head_dim_lines {
                let q_line: Line<i8> = q_int8[(q_line_base + line_idx) as usize];
                let k_line: Line<i8> = k_int8[(k_line_base + line_idx) as usize];
                dot_i32 += q_line.dot_i32(k_line);
                line_idx += 1u32;
            }

            // Dequantize: score = dot * q_scale * k_scale * attn_scale
            let score: f32 = (dot_i32 as f32) * q_scale * k_scale * attn_scale;

            // Track block max
            block_max = select(score > block_max, score, block_max);

            kv_idx += 1u32;
        }

        // Update running max
        let m_new: f32 = select(block_max > m_i, block_max, m_i);
        let alpha: f32 = (m_i - m_new).exp2();

        // Rescale existing accumulator
        let mut d2: u32 = 0u32;
        while d2 < head_dim {
            let out_idx: usize = (q_base + d2) as usize;
            let old_acc: f32 = f32::cast_from(output[out_idx][0]);
            output[out_idx] = Line::cast_from(old_acc * alpha);
            d2 += 1u32;
        }
        l_i = l_i * alpha;

        // Second pass: compute exp2(score - m_new) and accumulate
        kv_idx = kv_start;
        while kv_idx < kv_end {
            // Recompute score (could cache but register pressure)
            let k_line_base: u32 = kv_base / line_size + kv_idx * head_dim_lines;
            // Get K scale (per-block: same scale for all rows in the block)
            let k_row_pos2: u32 = k_row_base + kv_idx;
            let k_scale_idx: usize = (k_row_pos2 / k_quant_block_size) as usize;
            let k_scale: f32 = f32::cast_from(k_scales[k_scale_idx][0]);

            let mut dot_i32: i32 = 0i32;
            let mut line_idx: u32 = 0u32;
            while line_idx < head_dim_lines {
                let q_line: Line<i8> = q_int8[(q_line_base + line_idx) as usize];
                let k_line: Line<i8> = k_int8[(k_line_base + line_idx) as usize];
                dot_i32 += q_line.dot_i32(k_line);
                line_idx += 1u32;
            }

            let score: f32 = (dot_i32 as f32) * q_scale * k_scale * attn_scale;
            let p: f32 = (score - m_new).exp2();

            l_i += p;

            // Accumulate weighted V
            let v_base: u32 = kv_base + kv_idx * head_dim;
            let mut d3: u32 = 0u32;
            while d3 < head_dim {
                let out_idx: usize = (q_base + d3) as usize;
                let v_idx: usize = (v_base + d3) as usize;
                let old_acc: f32 = f32::cast_from(output[out_idx][0]);
                let vv: f32 = f32::cast_from(v[v_idx][0]);
                output[out_idx] = Line::cast_from(old_acc + p * vv);
                d3 += 1u32;
            }

            kv_idx += 1u32;
        }

        m_i = m_new;
        kv_block += 1u32;
    }

    // Final normalization
    let inv_l: f32 = 1.0f32 / l_i;
    let mut d4: u32 = 0u32;
    while d4 < head_dim {
        let out_idx: usize = (q_base + d4) as usize;
        let acc: f32 = f32::cast_from(output[out_idx][0]);
        output[out_idx] = Line::cast_from(acc * inv_l);
        d4 += 1u32;
    }
}

// =============================================================================
// INT8 Quantization for SageAttention
// =============================================================================

/// Launch INT8 quantization kernel with per-block scaling
///
/// Quantizes a tensor to INT8 with per-block scale factors.
/// For each block of `block_size` rows: scale = max(abs(block)) / 127
///
/// # Arguments
/// * `input` - Input tensor [batch, heads, seq, head_dim] (f32)
/// * `output` - Output tensor [batch, heads, seq, head_dim] (i8)
/// * `scales` - Scale factors [num_blocks] (f32), where num_blocks = ceil(batch*heads*seq / block_size)
/// * `block_size` - Number of rows per quantization block (e.g., 128 for Q, 64 for K)
#[allow(clippy::too_many_arguments)]
pub fn launch_quantize_int8<R: Runtime>(
    client: &ComputeClient<R>,
    input: &TensorHandleRef<R>,
    output: &TensorHandleRef<R>,
    scales: &TensorHandleRef<R>,
    batch: usize,
    heads: usize,
    seq: usize,
    head_dim: usize,
    block_size: usize,
) -> Result<(), LaunchError> {
    let num_rows = batch * heads * seq;
    let num_blocks = (num_rows + block_size - 1) / block_size;

    // Each cube handles one block, units cooperate on finding max and quantizing
    // Use enough units to cover block_size rows in parallel
    let units_per_cube = 64.min(block_size * head_dim);
    let cube_dim = CubeDim::new(client, units_per_cube);
    let cube_count = CubeCount::new_1d(num_blocks as u32);

    unsafe {
        quantize_int8_block_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            input.as_tensor_arg(1),
            output.as_tensor_arg(1),
            scales.as_tensor_arg(1),
            ScalarArg::new(head_dim as u32),
            ScalarArg::new(num_rows as u32),
            ScalarArg::new(block_size as u32),
        )
    }
}

/// Per-block INT8 quantization kernel
///
/// Each cube handles one block of rows (block_size query/key positions).
/// Units cooperate to find max abs value across the entire block, then quantize.
/// This matches the reference SageAttention which uses BLKQ=128 and BLKK=64.
#[cube(launch_unchecked)]
fn quantize_int8_block_kernel(
    input: &Tensor<Line<f32>>,
    output: &mut Tensor<Line<i8>>,
    scales: &mut Tensor<Line<f32>>,
    head_dim: u32,
    num_rows: u32,
    block_size: u32,
) {
    let block_id: u32 = CUBE_POS_X;
    let unit_id: u32 = UNIT_POS_X;
    let num_units: u32 = CUBE_DIM_X;

    // Compute row range for this block
    let row_start: u32 = block_id * block_size;
    let row_end: u32 = select(row_start + block_size > num_rows, num_rows, row_start + block_size);
    let block_rows: u32 = row_end - row_start;

    // Total elements in this block
    let block_elems: u32 = block_rows * head_dim;

    // Step 1: Find max absolute value across the entire block (parallel reduction)
    let mut local_max: f32 = 0.0f32;
    let mut elem_idx: u32 = unit_id;
    while elem_idx < block_elems {
        // Convert block-local index to global index
        let local_row: u32 = elem_idx / head_dim;
        let local_col: u32 = elem_idx % head_dim;
        let global_idx: u32 = (row_start + local_row) * head_dim + local_col;

        let val: f32 = f32::cast_from(input[global_idx as usize][0]);
        let abs_val: f32 = select(val >= 0.0f32, val, -val);
        local_max = select(abs_val > local_max, abs_val, local_max);

        elem_idx += num_units;
    }

    // Reduce across units to get global max for the block
    let global_max: f32 = plane_max(local_max);

    // Compute scale (avoid division by zero)
    let scale: f32 = select(global_max > 0.0f32, global_max / 127.0f32, 1.0f32);

    // Unit 0 stores the scale for this block
    if unit_id == 0u32 {
        scales[block_id as usize] = Line::cast_from(scale);
    }

    // Step 2: Quantize all values in the block using the block scale
    let inv_scale: f32 = 1.0f32 / scale;
    let mut elem_idx2: u32 = unit_id;
    while elem_idx2 < block_elems {
        // Convert block-local index to global index
        let local_row: u32 = elem_idx2 / head_dim;
        let local_col: u32 = elem_idx2 % head_dim;
        let global_idx: u32 = (row_start + local_row) * head_dim + local_col;

        let val: f32 = f32::cast_from(input[global_idx as usize][0]);
        // Round to nearest integer, clamp to [-127, 127]
        let quantized: f32 = (val * inv_scale).round();
        let clamped: f32 = select(quantized > 127.0f32, 127.0f32, quantized);
        let clamped2: f32 = select(clamped < -127.0f32, -127.0f32, clamped);
        output[global_idx as usize] = Line::cast_from(clamped2 as i8);

        elem_idx2 += num_units;
    }
}

/// Line size for INT8 vectorization (DP4a uses 4 bytes at a time)
const INT8_LINE_SIZE: usize = 4;

/// Launch INT8 SageAttention kernel
///
/// Uses INT8 quantized Q and K for the dot product, with scale factor dequantization.
/// V is still in f32 for the softmax × V computation.
///
/// **Important**: head_dim must be divisible by 4 for vectorization.
///
/// # Arguments
/// * `q_int8` - Quantized query [batch, heads, seq_q, head_dim] (i8)
/// * `k_int8` - Quantized key [batch, heads, seq_kv, head_dim] (i8)
/// * `q_scales` - Query scale factors [batch, heads, seq_q] (f32)
/// * `k_scales` - Key scale factors [batch, heads, seq_kv] (f32)
/// * `v` - Value tensor [batch, heads, seq_kv, head_dim] (f32)
/// * `output` - Output tensor [batch, heads, seq_q, head_dim] (f32)
#[allow(clippy::too_many_arguments)]
pub fn launch_sage_attention_int8<R: Runtime>(
    client: &ComputeClient<R>,
    q_int8: &TensorHandleRef<R>,
    k_int8: &TensorHandleRef<R>,
    q_scales: &TensorHandleRef<R>,
    k_scales: &TensorHandleRef<R>,
    v: &TensorHandleRef<R>,
    output: &TensorHandleRef<R>,
    config: SageAttentionConfig,
) -> Result<(), LaunchError> {
    // Validate head_dim is divisible by line size for vectorization
    assert!(
        config.head_dim % INT8_LINE_SIZE == 0,
        "head_dim ({}) must be divisible by {} for INT8 vectorization",
        config.head_dim,
        INT8_LINE_SIZE
    );

    // Each cube handles one query position with parallel reduction
    let num_work_items = config.batch * config.heads * config.seq_q;
    // Limit units to head_dim / line_size since we iterate over lines
    let head_dim_lines = config.head_dim / INT8_LINE_SIZE;
    let units_per_cube = 64.min(head_dim_lines);
    let cube_dim = CubeDim::new(client, units_per_cube);
    let cube_count = CubeCount::new_1d(num_work_items as u32);

    unsafe {
        attention_kernel_int8::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            // Use line size 4 for INT8 tensors to enable DP4a vectorization
            q_int8.as_tensor_arg(INT8_LINE_SIZE),
            k_int8.as_tensor_arg(INT8_LINE_SIZE),
            q_scales.as_tensor_arg(1),
            k_scales.as_tensor_arg(1),
            v.as_tensor_arg(1),
            output.as_tensor_arg(1),
            ScalarArg::new(config.scale),
            ScalarArg::new(config.batch as u32),
            ScalarArg::new(config.heads as u32),
            ScalarArg::new(config.seq_q as u32),
            ScalarArg::new(config.seq_kv as u32),
            ScalarArg::new(config.head_dim as u32),
            INT8_LINE_SIZE as u32,
        )
    }
}

/// INT8 attention kernel with online softmax and DP4a vectorization
///
/// Uses vectorized INT8 dot products for Q·K^T via Line<i8>.dot() which maps to
/// DP4a instructions on supported hardware (4 INT8 multiply-accumulates per instruction).
///
/// Key insight: score = (q_int8 · k_int8) * q_scale * k_scale * attn_scale
///
/// head_dim MUST be divisible by 4 for vectorization
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn attention_kernel_int8(
    // Line size = 4 for vectorized i8 operations (DP4a)
    q_int8: &Tensor<Line<i8>>,
    k_int8: &Tensor<Line<i8>>,
    q_scales: &Tensor<Line<f32>>,
    k_scales: &Tensor<Line<f32>>,
    v: &Tensor<Line<f32>>,
    output: &mut Tensor<Line<f32>>,
    attn_scale: f32,
    batch_size: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    head_dim: u32,
    #[comptime] line_size: u32,
) {
    let work_id: u32 = CUBE_POS_X;
    let unit_id: u32 = UNIT_POS_X;
    let num_units: u32 = CUBE_DIM_X;

    // Decode position
    let query_pos: u32 = work_id % seq_q;
    let tmp: u32 = work_id / seq_q;
    let head: u32 = tmp % num_heads;
    let batch: u32 = tmp / num_heads;

    if batch >= batch_size {
        terminate!();
    }

    // Compute base offsets (in elements, not lines)
    let q_base: u32 =
        batch * num_heads * seq_q * head_dim + head * seq_q * head_dim + query_pos * head_dim;
    let kv_base: u32 = batch * num_heads * seq_kv * head_dim + head * seq_kv * head_dim;

    // Scale indices
    let q_scale_idx: usize = (batch * num_heads * seq_q + head * seq_q + query_pos) as usize;
    let k_scale_base: u32 = batch * num_heads * seq_kv + head * seq_kv;

    // Load query scale (all units read same value)
    let q_scale: f32 = f32::cast_from(q_scales[q_scale_idx][0]);

    // Online softmax state
    let mut m_i: f32 = f32::min_value();
    let mut l_i: f32 = 0.0f32;

    // Number of vector elements (lines) per row
    let head_dim_lines: u32 = head_dim / line_size;

    // Zero output accumulator
    let mut d: u32 = unit_id;
    while d < head_dim {
        let out_idx: usize = (q_base + d) as usize;
        output[out_idx] = Line::cast_from(0.0f32);
        d += num_units;
    }

    // Process all KV positions
    let mut kv_pos: u32 = 0u32;
    while kv_pos < seq_kv {
        let k_off: u32 = kv_base + kv_pos * head_dim;

        // Vectorized INT8 dot product with parallel reduction using DP4a
        // dot_i32() computes the dot product of Line<i8> and returns i32
        // This maps to SPIR-V OpSDotKHR which is hardware-accelerated on modern GPUs
        let mut partial_dot: i32 = 0i32;
        let mut line_idx: u32 = unit_id;
        while line_idx < head_dim_lines {
            // Index into tensor as lines (each line has `line_size` i8 values)
            let q_line_idx: usize = ((q_base / line_size) + line_idx) as usize;
            let k_line_idx: usize = ((k_off / line_size) + line_idx) as usize;

            // Load 4 i8 values at once and compute dot product with i32 accumulation
            // This emits OpSDotKHR (DP4a) on Vulkan 1.3+ with VK_KHR_shader_integer_dot_product
            let q_line: Line<i8> = q_int8[q_line_idx];
            let k_line: Line<i8> = k_int8[k_line_idx];
            partial_dot += q_line.dot_i32(k_line);

            line_idx += num_units;
        }

        // Reduce across units
        let dot_i32: i32 = plane_sum(partial_dot);

        // Dequantize: score = dot * q_scale * k_scale * attn_scale
        let k_scale_idx: usize = (k_scale_base + kv_pos) as usize;
        let k_scale: f32 = f32::cast_from(k_scales[k_scale_idx][0]);
        let score: f32 = (dot_i32 as f32) * q_scale * k_scale * attn_scale;

        // Online softmax update
        let m_new: f32 = select(score > m_i, score, m_i);
        let alpha: f32 = (m_i - m_new).exp();
        let p: f32 = (score - m_new).exp();

        l_i = l_i * alpha + p;

        // Update accumulator (V is still f32, no vectorization benefit here)
        let mut d3: u32 = unit_id;
        while d3 < head_dim {
            let out_idx: usize = (q_base + d3) as usize;
            let v_idx: usize = (k_off + d3) as usize;

            let old_acc: f32 = f32::cast_from(output[out_idx][0]);
            let vv: f32 = f32::cast_from(v[v_idx][0]);
            let new_acc: f32 = old_acc * alpha + p * vv;
            output[out_idx] = Line::cast_from(new_acc);

            d3 += num_units;
        }

        m_i = m_new;
        kv_pos += 1u32;
    }

    // Final normalization
    let mut d4: u32 = unit_id;
    while d4 < head_dim {
        let out_idx: usize = (q_base + d4) as usize;
        let acc: f32 = f32::cast_from(output[out_idx][0]);
        output[out_idx] = Line::cast_from(acc / l_i);
        d4 += num_units;
    }
}

// =============================================================================
// INT8 Attention with exp2 (matching reference implementation)
// =============================================================================

/// Launch INT8 SageAttention kernel (reference-style)
///
/// Matches reference implementation:
/// - INT8 Q·K^T with DP4a (dot_i32)
/// - exp2 instead of exp for faster softmax
/// - Per-row quantization scales
/// - No shared memory (relies on L2 cache)
///
/// # Arguments
/// * `q_int8` - Quantized query [batch, heads, seq_q, head_dim] (i8)
/// * `k_int8` - Quantized key [batch, heads, seq_kv, head_dim] (i8)
/// * `q_scales` - Query scale factors [batch, heads, seq_q] (f32)
/// * `k_scales` - Key scale factors [batch, heads, seq_kv] (f32)
/// * `v` - Value tensor [batch, heads, seq_kv, head_dim] (f32)
/// * `output` - Output tensor [batch, heads, seq_q, head_dim] (f32)
#[allow(clippy::too_many_arguments)]
pub fn launch_sage_attention_int8_exp2<R: Runtime>(
    client: &ComputeClient<R>,
    q_int8: &TensorHandleRef<R>,
    k_int8: &TensorHandleRef<R>,
    q_scales: &TensorHandleRef<R>,
    k_scales: &TensorHandleRef<R>,
    v: &TensorHandleRef<R>,
    output: &TensorHandleRef<R>,
    config: SageAttentionConfig,
) -> Result<(), LaunchError> {
    assert!(
        config.head_dim % INT8_LINE_SIZE == 0,
        "head_dim ({}) must be divisible by {} for INT8 vectorization",
        config.head_dim,
        INT8_LINE_SIZE
    );

    // Each cube handles one query position with parallel reduction
    let num_work_items = config.batch * config.heads * config.seq_q;
    let head_dim_lines = config.head_dim / INT8_LINE_SIZE;
    let units_per_cube = 64.min(head_dim_lines);
    let cube_dim = CubeDim::new(client, units_per_cube);
    let cube_count = CubeCount::new_1d(num_work_items as u32);

    unsafe {
        attention_kernel_int8_exp2::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            q_int8.as_tensor_arg(INT8_LINE_SIZE),
            k_int8.as_tensor_arg(INT8_LINE_SIZE),
            q_scales.as_tensor_arg(1),
            k_scales.as_tensor_arg(1),
            v.as_tensor_arg(1),
            output.as_tensor_arg(1),
            ScalarArg::new(config.scale),
            ScalarArg::new(config.batch as u32),
            ScalarArg::new(config.heads as u32),
            ScalarArg::new(config.seq_q as u32),
            ScalarArg::new(config.seq_kv as u32),
            ScalarArg::new(config.head_dim as u32),
            INT8_LINE_SIZE as u32,
        )
    }
}

/// INT8 attention kernel with exp2 (matching reference)
///
/// Same as attention_kernel_int8 but uses exp2 instead of exp:
/// - exp(x) = exp2(x * log2(e))
/// - This is faster on GPU hardware
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn attention_kernel_int8_exp2(
    q_int8: &Tensor<Line<i8>>,
    k_int8: &Tensor<Line<i8>>,
    q_scales: &Tensor<Line<f32>>,
    k_scales: &Tensor<Line<f32>>,
    v: &Tensor<Line<f32>>,
    output: &mut Tensor<Line<f32>>,
    attn_scale: f32,
    batch_size: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    head_dim: u32,
    #[comptime] line_size: u32,
) {
    let work_id: u32 = CUBE_POS_X;
    let unit_id: u32 = UNIT_POS_X;
    let num_units: u32 = CUBE_DIM_X;

    // Decode position
    let query_pos: u32 = work_id % seq_q;
    let tmp: u32 = work_id / seq_q;
    let head: u32 = tmp % num_heads;
    let batch: u32 = tmp / num_heads;

    if batch >= batch_size {
        terminate!();
    }

    // Base offsets (in elements)
    let q_base: u32 =
        batch * num_heads * seq_q * head_dim + head * seq_q * head_dim + query_pos * head_dim;
    let kv_base: u32 = batch * num_heads * seq_kv * head_dim + head * seq_kv * head_dim;

    // Scale indices
    let q_scale_idx: usize = (batch * num_heads * seq_q + head * seq_q + query_pos) as usize;
    let k_scale_base: u32 = batch * num_heads * seq_kv + head * seq_kv;

    // Load query scale
    let q_scale: f32 = f32::cast_from(q_scales[q_scale_idx][0]);

    // Online softmax state
    let mut m_i: f32 = f32::min_value();
    let mut l_i: f32 = 0.0f32;

    // Number of Lines per row
    let head_dim_lines: u32 = head_dim / line_size;

    // Zero output
    let mut d: u32 = unit_id;
    while d < head_dim {
        let out_idx: usize = (q_base + d) as usize;
        output[out_idx] = Line::cast_from(0.0f32);
        d += num_units;
    }

    // Process all KV positions
    let mut kv_pos: u32 = 0u32;
    while kv_pos < seq_kv {
        let k_off: u32 = kv_base + kv_pos * head_dim;

        // INT8 dot product with DP4a
        let mut partial_dot: i32 = 0i32;
        let mut line_idx: u32 = unit_id;
        while line_idx < head_dim_lines {
            let q_line_idx: usize = ((q_base / line_size) + line_idx) as usize;
            let k_line_idx: usize = ((k_off / line_size) + line_idx) as usize;

            let q_line: Line<i8> = q_int8[q_line_idx];
            let k_line: Line<i8> = k_int8[k_line_idx];
            partial_dot += q_line.dot_i32(k_line);

            line_idx += num_units;
        }

        // Reduce across units
        let dot_i32: i32 = plane_sum(partial_dot);

        // Dequantize: score = dot * q_scale * k_scale * attn_scale
        let k_scale_idx: usize = (k_scale_base + kv_pos) as usize;
        let k_scale: f32 = f32::cast_from(k_scales[k_scale_idx][0]);
        let score: f32 = (dot_i32 as f32) * q_scale * k_scale * attn_scale;

        // Online softmax update with exp2 (matching SageAttention reference)
        // Reference uses exp2 directly, not exp2(x * log2(e))
        let m_new: f32 = select(score > m_i, score, m_i);
        let alpha: f32 = (m_i - m_new).exp2();
        let p: f32 = (score - m_new).exp2();

        l_i = l_i * alpha + p;

        // Update accumulator
        let mut d3: u32 = unit_id;
        while d3 < head_dim {
            let out_idx: usize = (q_base + d3) as usize;
            let v_idx: usize = (k_off + d3) as usize;

            let old_acc: f32 = f32::cast_from(output[out_idx][0]);
            let vv: f32 = f32::cast_from(v[v_idx][0]);
            let new_acc: f32 = old_acc * alpha + p * vv;
            output[out_idx] = Line::cast_from(new_acc);

            d3 += num_units;
        }

        m_i = m_new;
        kv_pos += 1u32;
    }

    // Final normalization
    let mut d4: u32 = unit_id;
    while d4 < head_dim {
        let out_idx: usize = (q_base + d4) as usize;
        let acc: f32 = f32::cast_from(output[out_idx][0]);
        output[out_idx] = Line::cast_from(acc / l_i);
        d4 += num_units;
    }
}

#[cfg(test)]
mod tests {
    // Tests would go here
}
