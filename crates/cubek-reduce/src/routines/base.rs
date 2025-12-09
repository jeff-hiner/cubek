use crate::{
    LineMode,
    components::{
        instructions::*,
        level::{self, cube::ReduceCubeConfig, plane::PlaneReduceConfig, unit::UnitReduceConfig},
        partition::ReducePartition,
        precision::ReducePrecision,
        writer,
    },
    routines::{ReduceBlueprint, ReduceBlueprintKind},
};
use cubecl::{prelude::*, std::tensor::r#virtual::VirtualTensor};

#[cube]
pub fn reduce_kernel_virtual<In: Numeric, Out: Numeric, Acc: Numeric>(
    input: &VirtualTensor<In>,
    output: &mut VirtualTensor<Out, ReadWrite>,
    axis_reduce: u32,
    #[comptime] blueprint: ReduceBlueprint,
    #[comptime] config: ReduceOperationConfig,
) {
    let reduce_index = get_reduce_index(blueprint.kind);

    // #[allow(clippy::collapsible_if)]
    // if comptime![params.bound_checks] {
    //     if reduce_index
    //         >= get_reduce_count(
    //             output.len() * output.line_size(),
    //             params.line_mode,
    //             input.line_size(),
    //         )
    //     {
    //         terminate!();
    //     }
    // }

    reduce_kernel_inner::<(In, Acc), Out, ReduceOperation>(
        input,
        output,
        axis_reduce,
        reduce_index,
        blueprint,
        config,
    )
}

#[cube]
fn reduce_kernel_inner<P: ReducePrecision, Out: Numeric, R: ReduceFamily>(
    input: &VirtualTensor<P::EI>,
    output: &mut VirtualTensor<Out, ReadWrite>,
    axis_reduce: u32,
    reduce_index: u32,
    #[comptime] blueprint: ReduceBlueprint,
    #[comptime] config: R::Config,
) {
    let partition =
        ReducePartition::new::<P, Out>(reduce_index, input, output, axis_reduce, blueprint);

    let input_line_size = input.line_size();
    let inst = &R::Instruction::<P>::from_config(config);
    let accumulator = match comptime!(blueprint.kind) {
        ReduceBlueprintKind::Cube(cube) => {
            let config = comptime!(ReduceCubeConfig::new(
                input_line_size,
                blueprint.line_mode,
                cube
            ));
            level::cube::reduce::<P, VirtualTensor<P::EI>, R::Instruction<P>>(
                input, inst, partition, config,
            )
        }
        ReduceBlueprintKind::Plane(plane) => {
            let config = comptime!(PlaneReduceConfig::new(
                input_line_size,
                blueprint.line_mode,
                plane,
            ));
            level::plane::reduce::<P, VirtualTensor<P::EI>, R::Instruction<P>>(
                input, inst, partition, config,
            )
        }
        ReduceBlueprintKind::Unit => {
            let config = comptime!(UnitReduceConfig::new(input_line_size, blueprint.line_mode));

            level::unit::reduce_unit::<P, VirtualTensor<P::EI>, R::Instruction<P>>(
                input, partition, inst, config,
            )
        }
    };

    writer::write::<P, Out, R::Instruction<P>>(
        output,
        accumulator,
        reduce_index,
        input.shape(axis_reduce),
        blueprint,
        input.line_size(),
        inst,
    )
}

#[cube]
fn get_reduce_index(#[comptime] params: ReduceBlueprintKind) -> u32 {
    match params {
        ReduceBlueprintKind::Unit { .. } => ABSOLUTE_POS,
        ReduceBlueprintKind::Plane { .. } => CUBE_POS * CUBE_DIM_Y + UNIT_POS_Y,
        ReduceBlueprintKind::Cube { .. } => CUBE_POS,
    }
    // if params.shared.is_some() {
    //     CUBE_POS
    // } else if params.use_planes {
    // } else {
    //     ABSOLUTE_POS
    // }
}

#[cube]
fn get_reduce_count(
    output_size: u32,
    #[comptime] line_mode: LineMode,
    #[comptime] input_line_size: u32,
) -> u32 {
    match comptime!(line_mode) {
        LineMode::Parallel => output_size,
        LineMode::Perpendicular => output_size / input_line_size,
    }
}
