use crate::{
    LineMode,
    components::{
        instructions::*, level::fill_coordinate_line, partition::ReducePartition,
        precision::ReducePrecision,
    },
};
use cubecl::prelude::*;

/// Use an individual unit to reduce the `items` with the specified range.
/// That is, this will reduces `items[range.start]`, `items[range.start + range.step]`
/// until `items[range.end]` (exclusive).
///
/// This reduces using the given `line_mode` but doesn't reduce the accumulator itself.
///
/// Since each individual unit performs a reduction, this function is meant to be called
/// with either a different `items` for each unit, a different `range` or both based on ABSOLUTE_UNIT_POS.
#[cube]
pub fn reduce<P: ReducePrecision, I: List<Line<P::EI>>, R: ReduceInstruction<P>>(
    items: &I,
    partition: ReducePartition,
    inst: &R,
    #[comptime] line_size: u32,
    #[comptime] line_mode: LineMode,
) -> R::AccumulatorItem {
    let mut accumulator = R::null_accumulator(inst, line_size);

    let mut index = partition.index_start;
    for coordinate in range_stepped(
        partition.coordinate_start,
        partition.coordinate_end,
        partition.coordinate_step,
    ) {
        let requirements = R::requirements(inst);
        let coordinates = if comptime![requirements.coordinates] {
            ReduceCoordinate::new_Required(fill_coordinate_line(coordinate, line_size, line_mode))
        } else {
            ReduceCoordinate::new_NotRequired()
        };
        reduce_inplace::<P, R>(
            inst,
            &mut accumulator,
            items.read(index),
            coordinates,
            false,
        );
        index += partition.index_step;
    }

    accumulator
}
