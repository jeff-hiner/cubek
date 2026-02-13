use cubecl;
use cubecl::prelude::*;

use cubecl::std::tensor::layout::Coords2d;

use crate::components::tile::RowWise;

#[cube]
/// Describes how a fragment is fragmented across units
/// The layout is independent of the data and data types
pub trait FragmentLayout: CubeType {
    /// Maps the (row, col) of the registers of a single unit to the position within the whole tile
    ///
    /// Example: for simplicity, if we had a 4 units warp for a 4x4 tile divided as such:
    ///  0, 0, 1, 1,
    ///  2, 2, 3, 3,
    ///  0, 0, 1, 1,
    ///  2, 2, 3, 3,
    /// Then we would have:
    /// unit_0: absolute_pos((0, 0)) == (0, 0)
    /// unit_0: absolute_pos((0, 1)) == (0, 1)
    /// unit_0: absolute_pos((1, 0)) == (2, 0)
    /// unit_0: absolute_pos((1, 1)) == (2, 1)
    /// ...
    /// unit_3: absolute_pos((0, 0)) == (1, 2)
    /// unit_3: absolute_pos((0, 1)) == (1, 3)
    /// unit_3: absolute_pos((1, 0)) == (3, 2)
    /// unit_3: absolute_pos((1, 1)) == (3, 3)
    fn absolute_pos(&self, local_pos: Coords2d) -> Coords2d;

    /// Gives how many units participate in the same row
    ///
    /// Example: for simplicity, if we had a 4 units warp for a 4x4 tile divided as such:
    ///  0, 0, 1, 1,
    ///  2, 2, 3, 3,
    ///  0, 0, 1, 1,
    ///  2, 2, 3, 3,
    /// Then it would output 2, because each row is spread across two different units (0 and 1, or 2 and 3)
    /// Layouts with varying num_units_per_row are not supported
    fn num_units_per_row(&self) -> comptime_type!(u32);
}

#[cube]
pub trait FragmentSoftmax<E: Float>: CubeType {
    type Layout: FragmentLayout;
    type SoftmaxScore: CubeType;
    type SoftmaxRowFormat: RowwiseFormat<E, Layout = Self::Layout>;
    type SoftmaxVal: CubeType;

    /// Get the softmax fragment in row format
    fn rowwise_mut(&mut self) -> &mut Self::SoftmaxRowFormat;

    /// Update score/val from rowwise format
    fn update_from_rowwise(&mut self);

    /// Zeroes out the fragment
    fn zero(&mut self);

    /// Set the combined quantization scale for INT8 CMMA attention.
    /// This scale (q_scale * k_scale) is applied after CMMA to dequantize i32 results.
    /// For non-INT8 attention implementations, this is a no-op.
    fn set_combined_scale(&mut self, _scale: f32) {
        // Default no-op for non-INT8 implementations
    }
}

#[cube]
/// Trait for row-wise operations on attention score fragments.
pub trait RowwiseFormat<E: Float> {
    /// How the fragment is fragmented across units
    type Layout: FragmentLayout;

    fn num_units_per_row(&self) -> comptime_type!(u32);

    /// Return the maximum of each row
    /// Units only output values for rows they participate in
    fn rowwise_max(&self) -> RowWise<E>;

    /// Return the sum of each row
    /// Units only output values for rows they participate in
    fn rowwise_sum(&self) -> RowWise<E>;

    /// Scale every element by a constant factor, and masks values identified by the mask
    fn scale_and_mask<M: FragmentMask>(this: &mut Self, scale: E, mask: &M);

    /// Changes each value x_ij to e^(x_ij - m_i) for every row.
    fn exp_diff(&mut self, m: &RowWise<E>);
}

#[cube]
pub trait FragmentAccumulator<E: Float> {
    /// Scale each element in a row by a value for this row
    fn rowwise_scale(&mut self, val: &RowWise<E>);

    /// Zeroes out the fragment
    fn zero(&mut self);
}

#[cube]
/// Describes which elements of a fragment should be masked
pub trait FragmentMask: CubeType {
    /// How the fragment is fragmented across units
    type Layout: FragmentLayout;

    /// Returns `true` if the element at `local_pos` should be masked
    fn should_mask(&self, local_pos: Coords2d) -> bool;
}
