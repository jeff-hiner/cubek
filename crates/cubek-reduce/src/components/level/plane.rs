use crate::{
    BoundChecksInner, LineMode,
    components::{
        instructions::*, level::fill_coordinate_line, partition::ReducePartition,
        precision::ReducePrecision,
    },
    routines::PlaneReduceBlueprint,
};
use cubecl::prelude::*;

#[derive(Clone)]
pub struct PlaneReduceConfig {
    line_size: u32,
    line_mode: LineMode,
    bound_checks: BoundChecksInner,
}

impl PlaneReduceConfig {
    pub fn new(input_line_size: u32, line_mode: LineMode, blueprint: PlaneReduceBlueprint) -> Self {
        Self {
            line_size: input_line_size,
            line_mode,
            bound_checks: blueprint.bound_checks_inner,
        }
    }
}

#[derive(CubeType)]
struct Indexes {
    worker_index: u32,
    num_workers: u32,
}

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
///
/// # Notes
///
/// Multiple workers (PLANE) are reducing the full partition.
#[cube]
pub fn reduce<P: ReducePrecision, I: List<Line<P::EI>>, R: ReduceInstruction<P>>(
    items: &I,
    inst: &R,
    partition: ReducePartition,
    #[comptime] config: PlaneReduceConfig,
) -> R::AccumulatorItem {
    let indexes = Indexes {
        worker_index: CUBE_POS_X,
        num_workers: CUBE_DIM_X,
    };

    let mut accumulator = R::null_accumulator(inst, config.line_size);

    let mut first_index = partition.index_start;
    for first_coordinate in range_stepped(
        partition.coordinate_start,
        partition.coordinate_end,
        partition.coordinate_step,
    ) {
        let unit_coordinate_offset = match config.line_mode {
            LineMode::Parallel => indexes.worker_index * config.line_size,
            LineMode::Perpendicular => indexes.worker_index,
        };
        let unit_coordinate = first_coordinate + unit_coordinate_offset;

        let requirements = R::requirements(inst);
        let coordinates = if comptime![requirements.coordinates] {
            ReduceCoordinate::new_Required(fill_coordinate_line(
                unit_coordinate,
                config.line_size,
                config.line_mode,
            ))
        } else {
            ReduceCoordinate::new_NotRequired()
        };

        let index = first_index + indexes.worker_index * partition.index_step;
        let item = match config.bound_checks {
            BoundChecksInner::None => items.read(index),
            BoundChecksInner::Mask => {
                let mask = unit_coordinate < partition.coordinate_end;
                let index = index * u32::cast_from(mask);
                select(
                    mask,
                    items.read(index),
                    R::null_input(inst, config.line_size),
                )
            }
            BoundChecksInner::Branch => {
                if unit_coordinate < partition.coordinate_end {
                    items.read(index)
                } else {
                    R::null_input(inst, config.line_size)
                }
            }
        };

        reduce_inplace::<P, R>(inst, &mut accumulator, item, coordinates, true);

        first_index += indexes.num_workers * partition.index_step;
    }
    accumulator
}
