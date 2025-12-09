use crate::{
    BoundChecksInner, LineMode,
    components::{
        instructions::*, level::fill_coordinate_line, partition::ReducePartition,
        precision::ReducePrecision,
    },
    routines::CubeReduceBlueprint,
};
use cubecl::prelude::*;

#[derive(Clone)]
pub struct ReduceCubeConfig {
    pub accumulator_size: u32,
    pub line_size: u32,
    pub line_mode: LineMode,
    pub use_planes: bool,
    pub bound_checks: BoundChecksInner,
}

impl ReduceCubeConfig {
    pub fn new(input_line_size: u32, line_mode: LineMode, blueprint: CubeReduceBlueprint) -> Self {
        Self {
            accumulator_size: blueprint.accumulator_size,
            line_mode,
            line_size: input_line_size,
            use_planes: blueprint.use_planes,
            bound_checks: blueprint.bound_checks_inner,
        }
    }
}

#[derive(CubeType)]
struct Indexes {
    accumulator_index: u32,
    worker_index: u32,
    num_workers: u32,
}

/// Use an individual cube to reduce the `items` with the specified range.
/// That is, this will reduces `items[range.start]`, `items[range.start + range.step]`
/// until `items[range.end]` (exclusive). Inside a cube, the reduction will use plane operations
/// if `use_planes` is set to `true`.
///
/// This reduces using the given `line_mode` but doesn't reduce the accumulator itself.
///
/// When `use_planes` is `true`, this assumes that `UNIT_POS_Y` provides the relative position
/// of a plane within its cube.
///
/// Since each individual cube performs a reduction, this function is meant to be called
/// with either a different `items` for each cube, a different `range` or both based on `CUBE_POS`.
#[cube]
pub fn reduce<P: ReducePrecision, I: List<Line<P::EI>>, R: ReduceInstruction<P>>(
    items: &I,
    inst: &R,
    partition: ReducePartition,
    #[comptime] config: ReduceCubeConfig,
) -> R::AccumulatorItem {
    // The index used to read and write into the accumulator.
    let accumulator_index = match config.use_planes {
        true => UNIT_POS_Y,
        false => UNIT_POS,
    };

    let indexes = Indexes {
        accumulator_index,
        worker_index: UNIT_POS,
        num_workers: CUBE_DIM,
    };
    let mut accumulator = reduce_slice::<P, I, R>(
        items,
        inst,
        partition,
        &indexes,
        config.accumulator_size,
        config.line_size,
        config.line_mode,
        config.use_planes,
        config.bound_checks,
    );
    sync_cube();
    reduce_tree::<P, R>(inst, &mut accumulator, indexes, config.accumulator_size)
}

#[cube]
fn reduce_slice<P: ReducePrecision, I: List<Line<P::EI>>, R: ReduceInstruction<P>>(
    items: &I,
    inst: &R,
    partition: ReducePartition,
    indexes: &Indexes,
    #[comptime] accumulator_size: u32,
    #[comptime] line_size: u32,
    #[comptime] line_mode: LineMode,
    #[comptime] use_planes: bool,
    #[comptime] bound_checks: BoundChecksInner,
) -> R::SharedAccumulator {
    let requirements = R::requirements(inst);
    let mut accumulator =
        R::SharedAccumulator::allocate(accumulator_size, line_size, requirements.coordinates);

    R::SharedAccumulator::write(
        &mut accumulator,
        indexes.accumulator_index,
        R::null_accumulator(inst, line_size),
    );

    let mut first_index = partition.index_start;
    for first_coordinate in range_stepped(
        partition.coordinate_start,
        partition.coordinate_end,
        partition.coordinate_step,
    ) {
        let unit_coordinate_offset = match line_mode {
            LineMode::Parallel => indexes.worker_index * line_size,
            LineMode::Perpendicular => indexes.worker_index,
        };
        let unit_coordinate = first_coordinate + unit_coordinate_offset;
        let index = first_index + indexes.worker_index * partition.index_step;

        let item = match bound_checks {
            BoundChecksInner::None => items.read(index),
            BoundChecksInner::Mask => {
                let mask = unit_coordinate < partition.coordinate_end;
                let index = index * u32::cast_from(mask);
                select(mask, items.read(index), R::null_input(inst, line_size))
            }
            BoundChecksInner::Branch => {
                if unit_coordinate < partition.coordinate_end {
                    items.read(index)
                } else {
                    R::null_input(inst, line_size)
                }
            }
        };

        let coordinates = if comptime! {requirements.coordinates} {
            let coordinate = fill_coordinate_line(unit_coordinate, line_size, line_mode);
            let coordinate = select(
                unit_coordinate < partition.coordinate_end,
                coordinate,
                Line::empty(line_size).fill(u32::MAX),
            );

            ReduceCoordinate::new_Required(coordinate)
        } else {
            ReduceCoordinate::new_NotRequired()
        };

        reduce_shared_inplace::<P, R>(
            inst,
            &mut accumulator,
            indexes.accumulator_index,
            item,
            coordinates,
            use_planes,
        );
        first_index += partition.index_step * indexes.num_workers;
    }
    accumulator
}

/// Use all units within a cube to fuse the first `size` elements of `accumulator` inplace like this with some padding if `size` is not a power of 2.
///
///
/// ```ignored
///
///     0   1   2   3   4   5   6   7
///     |   |   |   |   |   |   |   |
///     +---+   +---+   +---+   +---+
///     |       |       |       |
///     +-------+       +-------+
///     |               |
///     +---------------+
///     |
///     *
///
/// ```
///
/// The outcome is stored in the first element of the accumulator and also returned by this function for convenience.
///
/// Since each individual cube performs a reduction, this function is meant to be called
/// with a different `accumulator` for each cube based on `CUBE_POS`.
///
/// There is no out-of-bound check, so it is the responsibility of the caller to ensure that `size` is at most the length
/// of the shared memory and that there are at least `size` units within each cube.
#[cube]
fn reduce_tree<P: ReducePrecision, Inst: ReduceInstruction<P>>(
    inst: &Inst,
    accumulator: &mut Inst::SharedAccumulator,
    indexes: Indexes,
    #[comptime] size: u32,
) -> Inst::AccumulatorItem {
    if comptime!(size.is_power_of_two()) {
        let mut num_active_units = size.runtime();
        let mut jump = 1;
        while num_active_units > 1 {
            num_active_units /= 2;
            let destination = jump * 2 * indexes.worker_index;
            let origin = jump * (2 * indexes.worker_index + 1);
            if indexes.worker_index < num_active_units {
                fuse_accumulator_inplace::<P, Inst>(inst, accumulator, destination, origin);
            }
            jump *= 2;
            sync_cube();
        }
    } else {
        let mut num_remaining_items = size.runtime();
        let mut jump = 1;
        while num_remaining_items > 1 {
            let destination = jump * 2 * indexes.worker_index;
            let origin = jump * (2 * indexes.worker_index + 1);
            if indexes.worker_index < num_remaining_items / 2 {
                fuse_accumulator_inplace::<P, Inst>(inst, accumulator, destination, origin);
            }
            num_remaining_items = num_remaining_items.div_ceil(2);
            jump *= 2;
            sync_cube();
        }
    }
    sync_cube();
    Inst::SharedAccumulator::read(accumulator, 0)
}
