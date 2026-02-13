//! Exp2-based local tile for INT8 CMMA attention.
//!
//! This wrapper around `LocalTile` uses `exp2()` instead of `exp()` in the softmax
//! computation. This is required for INT8 CMMA attention where `log2(e)` is baked
//! into the Q quantization scale, enabling the identity: `exp(x) = exp2(x * log2(e))`.

use crate::components::tile::{
    FragmentAccumulator, FragmentAccumulatorExpand, FragmentLayout, FragmentMask,
    FragmentMaskExpand, LOGIT_MASKED, RowWise, RowwiseFormat, RowwiseFormatExpand,
    accelerated::local_tile::{LocalTile, LocalTileLayout},
};
use cubecl::{self, prelude::*};

/// Wrapper around `LocalTile` that uses `exp2()` for softmax computation.
///
/// Used by INT8 CMMA attention which bakes `log2(e)` into Q quantization,
/// enabling the identity: `exp(x) = exp2(x * log2(e))`.
///
/// This follows SageAttention's approach where:
/// - Q is quantized with `sm_scale * log2(e)` baked in
/// - K is quantized with scale `1.0`
/// - Softmax uses `exp2()` instead of `exp()`
#[derive(CubeType)]
pub struct Exp2LocalTile<E: Numeric> {
    /// The underlying local tile.
    pub inner: LocalTile<E>,
}

#[cube]
impl<E: Numeric> Exp2LocalTile<E> {
    /// Create a new `Exp2LocalTile` with the given layout.
    pub fn new(layout: LocalTileLayout) -> Exp2LocalTile<E> {
        Exp2LocalTile::<E> {
            inner: LocalTile::new(layout),
        }
    }

    /// Zero out the tile.
    pub fn zero(&mut self) {
        self.inner.zero();
    }

    /// Load from a slice.
    pub fn load_from_slice(&mut self, smem_slice: &Slice<E>) {
        self.inner.load_from_slice(smem_slice);
    }

    /// Store to a mutable slice.
    pub fn store_to(&self, smem_slice: &mut SliceMut<E>) {
        self.inner.store_to(smem_slice);
    }
}

#[cube]
impl<E: Float> RowwiseFormat<E> for Exp2LocalTile<E> {
    type Layout = LocalTileLayout;

    fn rowwise_max(&self) -> RowWise<E> {
        self.inner.rowwise_max()
    }

    fn rowwise_sum(&self) -> RowWise<E> {
        self.inner.rowwise_sum()
    }

    fn scale_and_mask<M: FragmentMask>(this: &mut Self, scale: E, mask: &M) {
        // Inline the implementation to avoid type inference issues with the generic M
        #[unroll]
        for r in 0..this.inner.layout.unit_size.0 {
            let row_offset = r * this.inner.layout.unit_size.1;
            #[unroll]
            for c in 0..this.inner.layout.unit_size.1 {
                let index = row_offset + c;
                this.inner.array[index as usize] = this.inner.array[index as usize] * scale
                    + E::cast_from(mask.should_mask((r, c).runtime())) * E::min_value();
            }
        }
    }

    /// Compute `2^(x_ij - m_i)` for each element.
    ///
    /// This uses `exp2()` instead of `exp()` for INT8 CMMA attention,
    /// where `log2(e)` is baked into Q quantization.
    fn exp_diff(&mut self, val: &RowWise<E>) {
        let threshold = E::new(LOGIT_MASKED);

        #[unroll]
        for r in 0..self.inner.layout.unit_size.0 as usize {
            let row_offset = r as u32 * self.inner.layout.unit_size.1;

            let val = val.index(r);

            #[unroll]
            for c in 0..self.inner.layout.unit_size.1 {
                let index = row_offset + c;

                let safe_val = clamp_min(val, threshold);
                let not_masked = E::cast_from(val >= threshold);
                // Use exp2() instead of exp() for INT8 CMMA
                self.inner.array[index as usize] =
                    not_masked * (self.inner.array[index as usize] - safe_val).exp2();
            }
        }
    }

    fn num_units_per_row(&self) -> comptime_type!(u32) {
        <LocalTileLayout as FragmentLayout>::num_units_per_row(&self.inner.layout)
    }
}

#[cube]
impl<E: Float> FragmentAccumulator<E> for Exp2LocalTile<E> {
    fn rowwise_scale(&mut self, scale: &RowWise<E>) {
        self.inner.rowwise_scale(scale);
    }

    fn zero(&mut self) {
        self.inner.zero();
    }
}

#[cube]
impl<E: Numeric> FragmentMask for Exp2LocalTile<E> {
    type Layout = LocalTileLayout;

    fn should_mask(&self, local_pos: cubecl::std::tensor::layout::Coords2d) -> bool {
        self.inner.should_mask(local_pos)
    }
}
