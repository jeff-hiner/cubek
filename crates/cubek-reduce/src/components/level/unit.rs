use crate::{
    LineMode,
    components::{instructions::*, partition::ReducePartition, precision::ReducePrecision},
};
use cubecl::prelude::*;

#[derive(Clone)]
pub struct UnitReduceConfig {
    line_size: u32,
    line_mode: LineMode,
}

impl UnitReduceConfig {
    pub fn new(input_line_size: u32, line_mode: LineMode) -> Self {
        Self {
            line_size: input_line_size,
            line_mode,
        }
    }
}
/// Use an individual unit to reduce the `items` with the specified range.
/// That is, this will reduces `items[range.start]`, `items[range.start + range.step]`
/// until `items[range.end]` (exclusive).
///
/// This reduces using the given `line_mode` but doesn't reduce the accumulator itself.
///
/// Since each individual unit performs a reduction, this function is meant to be called
/// with either a different `items` for each unit, a different `range` or both based on ABSOLUTE_UNIT_POS.
///
/// # Notes
///
/// A single worker (UNIT) is reducing the full partition.
#[cube]
pub fn reduce_unit<P: ReducePrecision, I: List<Line<P::EI>>, R: ReduceInstruction<P>>(
    items: &I,
    partition: ReducePartition,
    inst: &R,
    #[comptime] config: UnitReduceConfig,
) -> R::AccumulatorItem {
    let mut accumulator = R::null_accumulator(inst, config.line_size);
    let mut index = partition.index_start;
    let requirements = R::requirements(inst);

    for coordinate in range_stepped(
        partition.coordinate_start,
        partition.coordinate_end,
        partition.coordinate_step,
    ) {
        let coordinates =
            ReduceCoordinate::new(coordinate, requirements, config.line_size, config.line_mode);

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
