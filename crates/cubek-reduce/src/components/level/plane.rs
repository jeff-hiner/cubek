use crate::{
    BoundChecksInner, LineMode,
    components::{
        instructions::*, level::fill_coordinate_line, precision::ReducePrecision,
        range::ReduceRange,
    },
};
use cubecl::prelude::*;

/// Use an individual plane  to reduce the `items` with the specified range.
/// That is, this will reduces `items[range.start]`, `items[range.start + range.step]`
/// until `items[range.end]` (exclusive).
///
/// This reduces using the given `line_mode` but doesn't reduce the accumulator itself.
///
/// This assumes that `UNIT_POS_X` provides the index of unit with a plane and that `CUBE_DIM_X` is the plane dimension.
/// That is, the cube_dim is `CubeDim::new_2d(plane_dim, plane_count)`.
///
/// Since each individual plane performs a reduction, this function is meant to be called
/// with either a different `items` for each plane, a different `range` or both based on
/// the absolute plane position (`CUBE_POS * CUBE_DIM_Y + UNIT_POS_Y`).
#[cube]
pub fn reduce<P: ReducePrecision, I: List<Line<P::EI>>, R: ReduceInstruction<P>>(
    items: &I,
    inst: &R,
    range: ReduceRange,
    #[comptime] line_size: u32,
    #[comptime] line_mode: LineMode,
    #[comptime] bound_checks: BoundChecksInner,
) -> R::AccumulatorItem {
    let plane_dim = CUBE_DIM_X;

    let mut accumulator = R::null_accumulator(inst, line_size);

    let mut first_index = range.index_start;
    for first_coordinate in range_stepped(
        range.coordinate_start,
        range.coordinate_end,
        range.coordinate_step,
    ) {
        let unit_coordinate_offset = match line_mode {
            LineMode::Parallel => UNIT_POS_X * line_size,
            LineMode::Perpendicular => UNIT_POS_X,
        };
        let unit_coordinate = first_coordinate + unit_coordinate_offset;

        let requirements = R::requirements(inst);
        let coordinates = if comptime![requirements.coordinates] {
            ReduceCoordinate::new_Required(fill_coordinate_line(
                unit_coordinate,
                line_size,
                line_mode,
            ))
        } else {
            ReduceCoordinate::new_NotRequired()
        };

        let index = first_index + UNIT_POS_X * range.index_step;
        let item = match bound_checks {
            BoundChecksInner::None => items.read(index),
            BoundChecksInner::Mask => {
                let mask = unit_coordinate < range.coordinate_end;
                let index = index * u32::cast_from(mask);
                select(mask, items.read(index), R::null_input(inst, line_size))
            }
            BoundChecksInner::Branch => {
                if unit_coordinate < range.coordinate_end {
                    items.read(index)
                } else {
                    R::null_input(inst, line_size)
                }
            }
        };

        reduce_inplace::<P, R>(inst, &mut accumulator, item, coordinates, true);

        first_index += plane_dim * range.index_step;
    }
    accumulator
}
