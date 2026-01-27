//! SageAttention tile computation with INT8 Q·K^T via DP4a

use cubecl::prelude::*;
use cubecl::std::tensor::layout::Coords2d;
use cubek_matmul::components::tile::StridedTile;

use crate::components::tile::LOGIT_MASKED;
use crate::components::tile::RowVal;
use crate::components::tile::RowWise;
use crate::components::tile::{FragmentAccumulator, FragmentAccumulatorExpand};
use crate::components::tile::{FragmentMask, FragmentMaskExpand};
use crate::components::tile::{FragmentSoftmax, FragmentSoftmaxExpand};
use crate::components::tile::{RowwiseFormat, RowwiseFormatExpand};
use crate::components::tile::{FragmentLayout, FragmentLayoutExpand};

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
    data: Array<i8>,
    /// Scale factor for dequantization: original = quantized * scale
    scale: f32,
    /// Layout (stored for future tile iteration support)
    #[expect(dead_code, reason = "Layout not yet used in scalar fallback; needed for DP4a path")]
    layout: SageTileLayout,
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
                    not_masked * (self.data[index as usize] - safe_val).exp();
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

// Note: Full TileAttention implementation would go here
// For now, this module provides the building blocks for SageAttention
