//! SageAttention tile computation with INT8 Q·K^T via DP4a

use cubecl::prelude::*;
use cubecl::std::tensor::layout::Coords2d;
use cubek_matmul::components::tile::StridedTile;

use crate::components::tile::sage::setup::SageTileAttentionConfig;
use crate::components::tile::TileAttention;
use crate::components::tile::LOGIT_MASKED;
use crate::components::tile::RowVal;
use crate::components::tile::RowWise;
use crate::components::tile::{FragmentAccumulator, FragmentAccumulatorExpand};
use crate::components::tile::{FragmentMask, FragmentMaskExpand};
use crate::components::tile::{FragmentSoftmax, FragmentSoftmaxExpand};
use crate::components::tile::{RowwiseFormat, RowwiseFormatExpand};
use crate::components::tile::{FragmentLayout, FragmentLayoutExpand};
use crate::definition::{AttentionPrecision, attention_types::*};

/// SageAttention tile implementation using INT8 quantization for Q·K^T
pub struct SageTileAttention;

/// Tile storing FP16/FP32 values with per-tile scale for INT8 quantization
#[derive(CubeType)]
pub struct SageTile<E: Numeric> {
    /// The data array
    data: Array<E>,
    /// Tile layout
    layout: SageTileLayout,
}

/// INT8 quantized tile with scale factor
#[derive(CubeType)]
pub struct QuantizedTile {
    /// INT8 quantized data
    pub data: Array<i8>,
    /// Scale factor for dequantization: original = quantized * scale
    pub scale: f32,
    /// Tile layout
    pub layout: SageTileLayout,
}

#[derive(CubeType, Copy, Clone)]
pub struct SageTileLayout {
    #[cube(comptime)]
    num_rows: u32,
    #[cube(comptime)]
    num_cols: u32,
}

#[cube]
impl<E: Numeric> SageTile<E> {
    pub fn new(layout: SageTileLayout) -> SageTile<E> {
        let data = Array::<E>::new(comptime!(layout.num_rows * layout.num_cols) as usize);
        SageTile::<E> { data, layout }
    }

    pub fn zero(&mut self) {
        for i in 0..self.layout.num_rows * self.layout.num_cols {
            self.data[i as usize] = E::from_int(0);
        }
    }

    pub fn get(&self, i: u32, j: u32) -> E {
        self.data[(i * self.layout.num_cols + j) as usize]
    }

    pub fn set(&mut self, i: u32, j: u32, val: E) {
        self.data[(i * self.layout.num_cols + j) as usize] = val;
    }

    pub fn accumulate(&mut self, i: u32, j: u32, val: E) {
        self.data[(i * self.layout.num_cols + j) as usize] += val;
    }
}

#[cube]
impl SageTileLayout {
    pub fn new(#[comptime] num_rows: u32, #[comptime] num_cols: u32) -> SageTileLayout {
        SageTileLayout { num_rows, num_cols }
    }
}

#[cube]
impl FragmentLayout for SageTileLayout {
    fn absolute_pos(&self, local_pos: Coords2d) -> Coords2d {
        local_pos
    }

    fn num_units_per_row(&self) -> comptime_type!(u32) {
        1u32
    }
}

/// Quantize a float tile to INT8 with symmetric quantization
/// Returns the scale factor used: scale = max(abs(tile)) / 127
#[cube]
fn quantize_tile_to_int8<E: Float>(tile: &SageTile<E>, out: &mut QuantizedTile) {
    // Find max absolute value
    let mut max_abs = E::from_int(0);
    for i in 0..tile.layout.num_rows * tile.layout.num_cols {
        let val = tile.data[i as usize];
        let abs_val = select(val >= E::from_int(0), val, E::from_int(0) - val);
        max_abs = select(abs_val > max_abs, abs_val, max_abs);
    }

    // Compute scale (avoid division by zero)
    let scale = f32::cast_from(max_abs) / 127.0 + 1e-10;
    out.scale = scale;

    // Quantize to INT8
    let inv_scale = 1.0 / scale;
    for i in 0..tile.layout.num_rows * tile.layout.num_cols {
        let val = f32::cast_from(tile.data[i as usize]);
        let quantized = f32::round(val * inv_scale);
        // Clamp to [-127, 127] and cast directly to i8
        let clamped = clamp(quantized, -127.0, 127.0);
        // Cast f32 -> i8 directly (cubecl handles the truncation)
        out.data[i as usize] = i8::cast_from(clamped);
    }
}

/// Compute INT8 matmul using scalar operations (fallback without DP4a)
/// lhs: [m, k] in row-major (Q)
/// rhs: [k, n] in row-major (K^T)
/// out: [m, n] with scores = q_scale * k_scale * sum(q_i8 * k_i8)
///
/// Note: For full DP4a support, we'd need to use Line<i8>.dot() but that
/// requires careful index management in cubecl. This scalar version still
/// benefits from the reduced memory bandwidth of INT8 storage.
#[cube]
fn int8_score_matmul(
    lhs: &QuantizedTile,
    rhs: &QuantizedTile,
    out: &mut SageTile<f32>,
    #[comptime] m: u32,
    #[comptime] n: u32,
    #[comptime] k: u32,
) {
    // Combined scale for dequantization
    let combined_scale = lhs.scale * rhs.scale;

    for m_ in 0..m {
        for n_ in 0..n {
            // Accumulate as f32
            let mut sum_f32 = 0.0f32;

            for k_ in 0..k {
                // Compute indices using u32 arithmetic
                let lhs_idx = m_ * k + k_;
                let rhs_idx = k_ * n + n_;

                // Load i8 values and multiply
                let lhs_val = f32::cast_from(lhs.data[lhs_idx as usize]);
                let rhs_val = f32::cast_from(rhs.data[rhs_idx as usize]);
                sum_f32 += lhs_val * rhs_val;
            }

            // Dequantize and store
            let dequantized = sum_f32 * combined_scale;
            out.accumulate(m_, n_, dequantized);
        }
    }
}

// Implement RowwiseFormat for SageTile (same as UnitTile)
#[cube]
impl<E: Float> RowwiseFormat<E> for SageTile<E> {
    type Layout = SageTileLayout;

    fn rowwise_max(&self) -> RowWise<E> {
        let mut vals = Sequence::new();

        #[unroll]
        for r in 0..self.layout.num_rows {
            let row_offset = r * self.layout.num_cols;
            let mut val = E::min_value();

            #[unroll]
            for c in 0..self.layout.num_cols {
                let index = row_offset + c;
                val = max(val, self.data[index as usize]);
            }

            vals.push(RowVal::<E> { val });
        }

        RowWise::<E> {
            num_rows: self.layout.num_rows.comptime() as usize,
            vals,
        }
    }

    fn rowwise_sum(&self) -> RowWise<E> {
        let mut vals = Sequence::new();

        #[unroll]
        for r in 0..self.layout.num_rows {
            let row_offset = r * self.layout.num_cols;
            let mut val = E::from_int(0);

            #[unroll]
            for c in 0..self.layout.num_cols {
                let index = row_offset + c;
                val += self.data[index as usize];
            }

            vals.push(RowVal::<E> { val });
        }

        RowWise::<E> {
            num_rows: self.layout.num_rows.comptime() as usize,
            vals,
        }
    }

    fn scale_and_mask<M: FragmentMask>(this: &mut Self, scale: E, mask: &M) {
        #[unroll]
        for r in 0..this.layout.num_rows {
            let row_offset = r * this.layout.num_cols;
            #[unroll]
            for c in 0..this.layout.num_cols {
                let index = row_offset + c;
                this.data[index as usize] = this.data[index as usize] * scale
                    + E::cast_from(mask.should_mask((r, c).runtime())) * E::min_value();
            }
        }
    }

    fn exp_diff(&mut self, val: &RowWise<E>) {
        let threshold = E::new(LOGIT_MASKED);

        #[unroll]
        for r in 0..self.layout.num_rows as usize {
            let row_offset = r as u32 * self.layout.num_cols;

            let val = val.index(r);

            #[unroll]
            for c in 0..self.layout.num_cols {
                let index = row_offset + c;

                let safe_val = clamp_min(val, threshold);
                let not_masked = E::cast_from(val >= threshold);
                self.data[index as usize] =
                    not_masked * (self.data[index as usize] - safe_val).exp2();
            }
        }
    }

    fn num_units_per_row(&self) -> comptime_type!(u32) {
        self.layout.num_units_per_row()
    }
}

#[cube]
impl<E: Float> FragmentAccumulator<E> for SageTile<E> {
    fn rowwise_scale(&mut self, scale: &RowWise<E>) {
        #[unroll]
        for r in 0..self.layout.num_rows as usize {
            let row_offset = r as u32 * self.layout.num_cols;
            #[unroll]
            for c in 0..self.layout.num_cols {
                let index = row_offset + c;
                self.data[index as usize] = self.data[index as usize] * scale.index(r);
            }
        }
    }

    fn zero(&mut self) {
        self.zero()
    }
}

#[cube]
impl<E: Float> FragmentSoftmax<E> for SageTile<E> {
    type Layout = SageTileLayout;
    type SoftmaxScore = SageTile<E>;
    type SoftmaxRowFormat = SageTile<E>;
    type SoftmaxVal = SageTile<E>;

    fn rowwise_mut(&mut self) -> &mut SageTile<E> {
        self
    }

    fn update_from_rowwise(&mut self) {
        // Nothing to do
    }

    fn zero(&mut self) {
        self.zero()
    }
}

#[cube]
impl<E: Numeric> FragmentMask for SageTile<E> {
    type Layout = SageTileLayout;

    fn should_mask(&self, local_pos: Coords2d) -> bool {
        bool::cast_from(self.data[(local_pos.0 * self.layout.num_cols + local_pos.1) as usize])
    }
}

// Helper functions for loading/storing tiles
#[cube]
fn strided_tile_to_sage_tile<E: Numeric, E2: Numeric>(
    strided_tile: &StridedTile<E>,
    sage_tile: &mut SageTile<E2>,
) {
    let line_size = strided_tile.line_size;
    assert!(sage_tile.layout.num_cols % line_size == 0);

    let col_iterations = comptime!(sage_tile.layout.num_cols / line_size);

    for row in 0..sage_tile.layout.num_rows {
        for col in 0..col_iterations {
            let line_read = strided_tile.get_line(row, col);
            #[unroll]
            for i in 0..line_size {
                sage_tile.data[(row * sage_tile.layout.num_cols + col * line_size + i) as usize] =
                    E2::cast_from(line_read[i as usize]);
            }
        }
    }
}

#[cube]
fn strided_tile_to_transposed_sage_tile<E: Numeric, E2: Numeric>(
    strided_tile: &StridedTile<E>,
    sage_tile: &mut SageTile<E2>,
) {
    let line_size = strided_tile.line_size;
    assert!(sage_tile.layout.num_cols % line_size == 0);

    let input_num_rows = sage_tile.layout.num_cols.comptime();
    let input_num_cols = sage_tile.layout.num_rows.comptime();
    let line_iterations = input_num_cols / line_size;

    for input_row in 0..input_num_rows {
        for input_col_line in 0..line_iterations {
            let line_read = strided_tile.get_line(input_row, input_col_line);

            #[unroll]
            for i in 0..line_size {
                sage_tile.data[((input_col_line + i) * input_num_rows + input_row) as usize] =
                    E2::cast_from(line_read[i as usize]);
            }
        }
    }
}

#[cube]
fn sage_tile_to_slice<E: Numeric, E2: Numeric>(
    sage_tile: &SageTile<E>,
    slice: &mut SliceMut<Line<E2>>,
) {
    let line_size = slice.line_size().comptime() as u32;
    assert!(sage_tile.layout.num_cols % line_size == 0);

    let col_iterations = comptime!(sage_tile.layout.num_cols / line_size);

    for row in 0..sage_tile.layout.num_rows {
        for col in 0..col_iterations {
            let mut out_line = Line::empty(line_size as usize);

            #[unroll]
            for i in 0..line_size {
                let index = row * sage_tile.layout.num_cols + col * line_size + i;
                out_line[i as usize] = E2::cast_from(sage_tile.data[index as usize]);
            }

            let line_index = row * col_iterations + col;
            slice[line_index as usize] = out_line;
        }
    }
}

/// Standard FP matmul for value multiplication (keep as FP16/FP32)
#[cube]
fn sage_inner_matmul<Lhs: Float, Rhs: Float, Acc: Float>(
    lhs: &SageTile<Lhs>,
    rhs: &SageTile<Rhs>,
    out: &mut SageTile<Acc>,
    #[comptime] m: u32,
    #[comptime] n: u32,
    #[comptime] k: u32,
) {
    for m_ in 0..m {
        for n_ in 0..n {
            let mut sum = Acc::from_int(0);
            for k_ in 0..k {
                let lhs_val = lhs.get(m_, k_);
                let rhs_val = rhs.get(k_, n_);
                sum += Acc::cast_from(lhs_val) * Acc::cast_from(rhs_val);
            }
            out.accumulate(m_, n_, sum);
        }
    }
}

// ============================================================================
// TileAttention implementation for SageTileAttention
// ============================================================================
//
// This implements the TileAttention trait using INT8 quantization for Q·K^T:
// 1. load_query: loads float data into SageTile, then quantize to QuantizedTile
// 2. load_key_transposed: same pattern
// 3. score_matmul: uses int8_score_matmul with combined_scale = q_scale * k_scale
// 4. value_matmul: standard float matmul (V stays float)
// 5. write_results: write SageTile to output slice

/// Wrapper holding both float and quantized versions of Q tile.
/// The float version is loaded first, then quantized for INT8 matmul.
#[derive(CubeType)]
pub struct SageQueryTile {
    /// Quantized INT8 version for Q·K^T CMMA
    pub quantized: QuantizedTile,
}

/// Wrapper holding both float and quantized versions of K tile (transposed).
/// K is loaded, transposed, and quantized for INT8 matmul.
#[derive(CubeType)]
pub struct SageKeyTile {
    /// Quantized INT8 version for Q·K^T CMMA
    pub quantized: QuantizedTile,
}

#[cube]
impl<AP: AttentionPrecision> TileAttention<AP> for SageTileAttention {
    type Config = SageTileAttentionConfig;

    type Query = SageQueryTile;
    type Key = SageKeyTile;
    type Value = SageTile<VT<AP>>;
    type Mask = SageTile<MSK<AP>>;
    type Softmax = SageTile<SM<AP>>;
    type SoftmaxRow = SageTile<SM<AP>>;
    type Accumulator = SageTile<ACC<AP>>;
    type FragmentLayout = SageTileLayout;

    fn softmax_layout(#[comptime] config: Self::Config) -> Self::FragmentLayout {
        SageTileLayout {
            num_rows: config.shared.attention_tile_size.seq_q,
            num_cols: config.shared.attention_tile_size.seq_kv,
        }
    }

    fn score_matmul(
        lhs: &Self::Query,
        rhs: &Self::Key,
        _key_tile: &StridedTile<KS<AP>>,
        out: &mut Self::Softmax,
        #[comptime] config: Self::Config,
    ) {
        // INT8 quantized matmul with proper scale handling.
        // Computes: out[m,n] += combined_scale * sum_k(lhs_i8[m,k] * rhs_i8[k,n])
        let (m, n, k) = comptime! {
            let size = config.shared.attention_tile_size.to_score_matmul_tile_size();
            let (m, n, k): (u32, u32, u32) = size.into();
            (m, n, k)
        };

        // Combined scale for dequantization: q_scale * k_scale
        let combined_scale = lhs.quantized.scale * rhs.quantized.scale;

        for m_ in 0..m {
            for n_ in 0..n {
                // Accumulate as f32
                let mut sum_f32 = 0.0f32;

                for k_ in 0..k {
                    // Compute indices
                    let lhs_idx = m_ * k + k_;
                    let rhs_idx = k_ * n + n_;

                    // Load i8 values and multiply
                    let lhs_val = f32::cast_from(lhs.quantized.data[lhs_idx as usize]);
                    let rhs_val = f32::cast_from(rhs.quantized.data[rhs_idx as usize]);
                    sum_f32 += lhs_val * rhs_val;
                }

                // Dequantize and accumulate into output
                let dequantized = sum_f32 * combined_scale;
                out.data[(m_ * n + n_) as usize] += SM::<AP>::cast_from(dequantized);
            }
        }
    }

    fn value_matmul(
        lhs: &Self::Softmax,
        rhs: &Self::Value,
        out: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        // Standard float matmul for P×V (V stays float, not quantized).
        // Computes: out[m,n] += sum_k(lhs[m,k] * rhs[k,n])
        let (m, n, k) = comptime! {
            let size = config.shared.attention_tile_size.to_value_matmul_tile_size();
            let (m, n, k): (u32, u32, u32) = size.into();
            (m, n, k)
        };

        for m_ in 0..m {
            for n_ in 0..n {
                let mut sum = ACC::<AP>::from_int(0);
                for k_ in 0..k {
                    let lhs_val = lhs.get(m_, k_);
                    let rhs_val = rhs.get(k_, n_);
                    sum += ACC::<AP>::cast_from(lhs_val) * ACC::<AP>::cast_from(rhs_val);
                }
                out.accumulate(m_, n_, sum);
            }
        }
    }

    fn allocate_query(#[comptime] config: Self::Config) -> Self::Query {
        let layout = SageTileLayout::new(
            config.shared.attention_tile_size.seq_q,
            config.shared.attention_tile_size.head_dim,
        );
        let size = comptime!(layout.num_rows * layout.num_cols) as usize;
        SageQueryTile {
            quantized: QuantizedTile {
                data: Array::<i8>::new(size),
                scale: 0.0f32,
                layout,
            },
        }
    }

    fn allocate_key(#[comptime] config: Self::Config) -> Self::Key {
        // K^T layout: [head_dim, seq_kv]
        let layout = SageTileLayout::new(
            config.shared.attention_tile_size.head_dim,
            config.shared.attention_tile_size.seq_kv,
        );
        let size = comptime!(layout.num_rows * layout.num_cols) as usize;
        SageKeyTile {
            quantized: QuantizedTile {
                data: Array::<i8>::new(size),
                scale: 0.0f32,
                layout,
            },
        }
    }

    fn allocate_value(#[comptime] config: Self::Config) -> Self::Value {
        SageTile::new(SageTileLayout::new(
            config.shared.attention_tile_size.seq_kv,
            config.shared.attention_tile_size.val_dim,
        ))
    }

    fn allocate_mask(#[comptime] config: Self::Config) -> Self::Mask {
        SageTile::new(<Self as TileAttention<AP>>::softmax_layout(config))
    }

    fn allocate_softmax(#[comptime] config: Self::Config) -> Self::Softmax {
        SageTile::new(<Self as TileAttention<AP>>::softmax_layout(config))
    }

    fn allocate_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        SageTile::new(SageTileLayout::new(
            config.shared.attention_tile_size.seq_q,
            config.shared.attention_tile_size.val_dim,
        ))
    }

    fn load_query<E: Numeric>(tile: &StridedTile<E>, fragment: &mut Self::Query) {
        // Load float tile, then quantize
        let layout = fragment.quantized.layout;
        let mut float_tile = SageTile::<f32>::new(layout);
        strided_tile_to_sage_tile(tile, &mut float_tile);
        quantize_tile_to_int8(&float_tile, &mut fragment.quantized);
    }

    fn load_key_transposed<E: Numeric>(
        tile: &StridedTile<E>,
        fragment: &mut Self::Key,
        #[comptime] _config: Self::Config,
    ) {
        // Load transposed float tile, then quantize
        let layout = fragment.quantized.layout;
        let mut float_tile = SageTile::<f32>::new(layout);
        strided_tile_to_transposed_sage_tile(tile, &mut float_tile);
        quantize_tile_to_int8(&float_tile, &mut fragment.quantized);
    }

    fn load_value<E: Numeric>(
        tile: &StridedTile<E>,
        fragment: &mut Self::Value,
        #[comptime] _config: Self::Config,
    ) {
        // V stays float, no quantization
        strided_tile_to_sage_tile(tile, fragment);
    }

    fn load_mask<E: Numeric>(
        tile: &StridedTile<E>,
        fragment: &mut Self::Mask,
        #[comptime] _config: Self::Config,
    ) {
        strided_tile_to_sage_tile(tile, fragment);
    }

    fn write_results<E: Float>(
        out: &Self::Accumulator,
        slice: &mut SliceMut<Line<E>>,
        #[comptime] _config: Self::Config,
    ) {
        sage_tile_to_slice(out, slice);
    }
}

#[cfg(all(test, any(
    test_runtime_default,
    test_runtime_wgpu,
    test_runtime_cpu,
    test_runtime_cuda,
    test_runtime_hip
)))]
mod tests {
    use super::*;
    use cubecl::ir::{ElemType, FloatKind};
    use cubecl::std::tensor::TensorHandle;
    use cubecl::{CubeElement, Runtime, TestRuntime};

    /// Test kernel that exercises INT8 quantization and matmul
    #[cube(launch)]
    fn int8_attention_test_kernel(
        q_in: &Tensor<Line<f32>>,
        k_in: &Tensor<Line<f32>>,
        v_in: &Tensor<Line<f32>>,
        out: &mut Tensor<Line<f32>>,
        #[comptime] tile_m: u32,
        #[comptime] tile_n: u32,
        #[comptime] tile_k: u32,
    ) {
        let layout_qk = SageTileLayout::new(tile_m, tile_k);
        let layout_kt = SageTileLayout::new(tile_k, tile_n);
        let layout_scores = SageTileLayout::new(tile_m, tile_n);
        let layout_v = SageTileLayout::new(tile_n, tile_k); // V: [seq_kv, head_dim]
        let layout_out = SageTileLayout::new(tile_m, tile_k);

        // Load Q tile
        let mut q_tile = SageTile::<f32>::new(layout_qk);
        for i in 0..tile_m {
            for j in 0..tile_k {
                q_tile.set(i, j, q_in[(i * tile_k + j) as usize][0]);
            }
        }

        // Load K tile (need K^T for Q·K^T, so load transposed)
        let mut kt_tile = SageTile::<f32>::new(layout_kt);
        for i in 0..tile_k {
            for j in 0..tile_n {
                // K is [seq_kv, head_dim], K^T is [head_dim, seq_kv]
                kt_tile.set(i, j, k_in[(j * tile_k + i) as usize][0]);
            }
        }

        // Quantize Q and K^T to INT8
        let mut q_int8 = QuantizedTile {
            data: Array::<i8>::new((tile_m * tile_k) as usize),
            scale: 0.0,
            layout: layout_qk,
        };
        let mut kt_int8 = QuantizedTile {
            data: Array::<i8>::new((tile_k * tile_n) as usize),
            scale: 0.0,
            layout: layout_kt,
        };
        quantize_tile_to_int8(&q_tile, &mut q_int8);
        quantize_tile_to_int8(&kt_tile, &mut kt_int8);

        // Compute scores = Q @ K^T using INT8 matmul
        let mut scores = SageTile::<f32>::new(layout_scores);
        scores.zero();
        int8_score_matmul(&q_int8, &kt_int8, &mut scores, tile_m, tile_n, tile_k);

        // Apply softmax (simplified: just exp and normalize per row)
        let row_max = <SageTile<f32> as RowwiseFormat<f32>>::rowwise_max(&scores);
        <SageTile<f32> as RowwiseFormat<f32>>::exp_diff(&mut scores, &row_max);
        let row_sum = <SageTile<f32> as RowwiseFormat<f32>>::rowwise_sum(&scores);

        // Normalize by row sum
        #[unroll]
        for i in 0..tile_m as usize {
            let sum: f32 = row_sum.index(i);
            let inv_sum = 1.0 / (sum + 1e-10);
            #[unroll]
            for j in 0..tile_n {
                let val = scores.get(i as u32, j);
                scores.set(i as u32, j, val * inv_sum);
            }
        }

        // Load V tile
        let mut v_tile = SageTile::<f32>::new(layout_v);
        for i in 0..tile_n {
            for j in 0..tile_k {
                v_tile.set(i, j, v_in[(i * tile_k + j) as usize][0]);
            }
        }

        // Compute output = scores @ V
        let mut out_tile = SageTile::<f32>::new(layout_out);
        out_tile.zero();
        sage_inner_matmul(&scores, &v_tile, &mut out_tile, tile_m, tile_k, tile_n);

        // Write output
        for i in 0..tile_m {
            for j in 0..tile_k {
                out[(i * tile_k + j) as usize] = Line::cast_from(out_tile.get(i, j));
            }
        }
    }

    #[test]
    fn test_int8_attention_sanity() {
        let client = <TestRuntime as Runtime>::client(&Default::default());

        // Small tile dimensions
        let tile_m = 4u32; // seq_q
        let tile_n = 4u32; // seq_kv
        let tile_k = 8u32; // head_dim

        let total_q = (tile_m * tile_k) as usize;
        let total_k = (tile_n * tile_k) as usize;
        let total_v = (tile_n * tile_k) as usize;
        let total_out = (tile_m * tile_k) as usize;

        // Create input data
        let q_data: Vec<f32> = (0..total_q).map(|i| (i as f32) * 0.1 - 1.6).collect();
        let k_data: Vec<f32> = (0..total_k).map(|i| (i as f32) * 0.1 - 1.6).collect();
        let v_data: Vec<f32> = (0..total_v).map(|i| (i as f32) * 0.1).collect();
        let out_data: Vec<f32> = vec![0.0; total_out];

        let dtype = StorageType::Scalar(ElemType::Float(FloatKind::F32));
        let line_size = 1usize;

        // Create tensor handles
        let q_handle: TensorHandle<TestRuntime> = TensorHandle::new(
            client.create_from_slice(f32::as_bytes(&q_data)),
            vec![total_q],
            vec![1],
            dtype,
        );
        let k_handle: TensorHandle<TestRuntime> = TensorHandle::new(
            client.create_from_slice(f32::as_bytes(&k_data)),
            vec![total_k],
            vec![1],
            dtype,
        );
        let v_handle: TensorHandle<TestRuntime> = TensorHandle::new(
            client.create_from_slice(f32::as_bytes(&v_data)),
            vec![total_v],
            vec![1],
            dtype,
        );
        let out_handle: TensorHandle<TestRuntime> = TensorHandle::new(
            client.create_from_slice(f32::as_bytes(&out_data)),
            vec![total_out],
            vec![1],
            dtype,
        );

        // Launch kernel - single thread
        int8_attention_test_kernel::launch::<TestRuntime>(
            &client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(1),
            unsafe {
                TensorArg::from_raw_parts_and_size(
                    &q_handle.handle,
                    &q_handle.strides,
                    &q_handle.shape,
                    line_size,
                    dtype.size(),
                )
            },
            unsafe {
                TensorArg::from_raw_parts_and_size(
                    &k_handle.handle,
                    &k_handle.strides,
                    &k_handle.shape,
                    line_size,
                    dtype.size(),
                )
            },
            unsafe {
                TensorArg::from_raw_parts_and_size(
                    &v_handle.handle,
                    &v_handle.strides,
                    &v_handle.shape,
                    line_size,
                    dtype.size(),
                )
            },
            unsafe {
                TensorArg::from_raw_parts_and_size(
                    &out_handle.handle,
                    &out_handle.strides,
                    &out_handle.shape,
                    line_size,
                    dtype.size(),
                )
            },
            tile_m,
            tile_n,
            tile_k,
        )
        .expect("Kernel launch failed");

        // Read results
        let result = client.read_one(out_handle.handle);
        let result_f32: &[f32] = bytemuck::cast_slice(&result);

        // Sanity checks
        let sum: f32 = result_f32.iter().map(|x| x.abs()).sum();
        assert!(sum > 0.01, "Expected non-zero output, got sum={sum}");

        for (i, val) in result_f32.iter().enumerate() {
            assert!(!val.is_nan(), "NaN at index {i}");
            assert!(!val.is_infinite(), "Inf at index {i}");
        }

        eprintln!("INT8 attention sanity test passed!");
        eprintln!(
            "Output (first 8): {:?}",
            &result_f32[..8.min(result_f32.len())]
        );
    }

    /// Test the production INT8 CMMA kernel via the public launch() API
    #[test]
    fn test_int8_cmma_production_kernel() {
        use crate::definition::{AccumulatorPrecision, AttentionGlobalTypes, AttentionOptions};
        use crate::launch::{launch, BlueprintStrategy, Strategy};
        use cubecl::ir::FloatKind;

        let client = <TestRuntime as Runtime>::client(&Default::default());

        // Dimensions that work with INT8 CMMA
        // head_dim must be 64 or 128 (padded like production code)
        let batch = 1usize;
        let heads = 1usize;
        let seq_q = 64usize;
        let seq_kv = 64usize;
        let head_dim = 64usize;

        let shape_q = vec![batch, heads, seq_q, head_dim];
        let shape_kv = vec![batch, heads, seq_kv, head_dim];
        let shape_out = vec![batch, heads, seq_q, head_dim];

        let strides_q = vec![heads * seq_q * head_dim, seq_q * head_dim, head_dim, 1];
        let strides_kv = vec![heads * seq_kv * head_dim, seq_kv * head_dim, head_dim, 1];

        let total_q = batch * heads * seq_q * head_dim;
        let total_kv = batch * heads * seq_kv * head_dim;

        let f16_dtype = StorageType::Scalar(ElemType::Float(FloatKind::F16));

        let q_data: Vec<half::f16> = (0..total_q)
            .map(|i| half::f16::from_f32((i as f32 % 100.0) * 0.01 - 0.5))
            .collect();
        let k_data: Vec<half::f16> = (0..total_kv)
            .map(|i| half::f16::from_f32((i as f32 % 100.0) * 0.01 - 0.5))
            .collect();
        let v_data: Vec<half::f16> = (0..total_kv)
            .map(|i| half::f16::from_f32((i as f32 % 100.0) * 0.01))
            .collect();
        let out_data: Vec<half::f16> = vec![half::f16::ZERO; total_q];

        let q_handle: TensorHandle<TestRuntime> = TensorHandle::new(
            client.create_from_slice(half::f16::as_bytes(&q_data)),
            shape_q.clone(),
            strides_q.clone(),
            f16_dtype,
        );
        let k_handle: TensorHandle<TestRuntime> = TensorHandle::new(
            client.create_from_slice(half::f16::as_bytes(&k_data)),
            shape_kv.clone(),
            strides_kv.clone(),
            f16_dtype,
        );
        let v_handle: TensorHandle<TestRuntime> = TensorHandle::new(
            client.create_from_slice(half::f16::as_bytes(&v_data)),
            shape_kv.clone(),
            strides_kv.clone(),
            f16_dtype,
        );
        let out_handle: TensorHandle<TestRuntime> = TensorHandle::new(
            client.create_from_slice(half::f16::as_bytes(&out_data)),
            shape_out,
            strides_q,
            f16_dtype,
        );

        let global_types = AttentionGlobalTypes::from_single_dtype(f16_dtype);
        let options = AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
            int8_cmma: true,
        };

        let strategy = Strategy::BlackboxAccelerated(BlueprintStrategy::Inferred(()));

        let runtime_name = TestRuntime::name(&client);
        eprintln!("INT8 CMMA test using runtime: {runtime_name}");
        eprintln!("Dimensions: batch={batch}, heads={heads}, seq_q={seq_q}, seq_kv={seq_kv}, head_dim={head_dim}");

        launch(
            strategy,
            &client,
            q_handle,
            k_handle,
            v_handle,
            None,
            out_handle.clone(),
            &global_types,
            options,
        )
        .expect("INT8 CMMA kernel launch failed");

        // Read results
        let output = client.read_one(out_handle.handle);
        let output_f16: &[half::f16] = half::f16::from_bytes(&output);

        // Sanity checks
        let sum: f32 = output_f16.iter().map(|x| x.to_f32().abs()).sum();
        assert!(
            sum > 0.01,
            "Expected non-zero output from INT8 CMMA kernel, got sum={sum}"
        );

        for (i, val) in output_f16.iter().enumerate() {
            let v = val.to_f32();
            assert!(!v.is_nan(), "NaN at index {i}");
            assert!(!v.is_infinite(), "Inf at index {i}");
        }

        eprintln!("INT8 CMMA production kernel test passed!");
        eprintln!(
            "Output (first 8): {:?}",
            output_f16[..8.min(output_f16.len())]
                .iter()
                .map(|x| x.to_f32())
                .collect::<Vec<_>>()
        );
    }
}
