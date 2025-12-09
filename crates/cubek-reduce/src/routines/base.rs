use crate::{
    LineMode, ReduceParams,
    components::{
        instructions::*,
        level::{self},
        partition::ReducePartition,
        precision::ReducePrecision,
        writer,
    },
};
use cubecl::{prelude::*, std::tensor::r#virtual::VirtualTensor};

#[cube]
pub fn reduce_kernel_virtual<In: Numeric, Out: Numeric, Acc: Numeric>(
    input: &VirtualTensor<In>,
    output: &mut VirtualTensor<Out, ReadWrite>,
    axis_reduce: u32,
    #[comptime] params: ReduceParams,
    #[comptime] config: ReduceOperationConfig,
) {
    let reduce_index = get_reduce_index(params);

    #[allow(clippy::collapsible_if)]
    if comptime![params.bound_checks] {
        if reduce_index >= get_reduce_count(output.len() * params.line_size_output, params) {
            terminate!();
        }
    }

    reduce_kernel_inner::<(In, Acc), Out, ReduceOperation>(
        input,
        output,
        axis_reduce,
        reduce_index,
        params,
        config,
    )
}

#[cube]
fn reduce_kernel_inner<P: ReducePrecision, Out: Numeric, R: ReduceFamily>(
    input: &VirtualTensor<P::EI>,
    output: &mut VirtualTensor<Out, ReadWrite>,
    axis_reduce: u32,
    reduce_index: u32,
    #[comptime] params: ReduceParams,
    #[comptime] config: R::Config,
) {
    let partition =
        ReducePartition::new::<P, Out>(reduce_index, input, output, axis_reduce, params);

    let inst = &R::Instruction::<P>::from_config(config);
    let accumulator = match comptime!((params.shared, params.use_planes)) {
        (Some(accumulator_size), use_planes) => {
            level::cube::reduce::<P, VirtualTensor<P::EI>, R::Instruction<P>>(
                input,
                inst,
                partition,
                accumulator_size,
                params.line_size_input,
                params.line_mode,
                use_planes,
                params.bound_checks_inner,
            )
        }
        (None, true) => level::plane::reduce::<P, VirtualTensor<P::EI>, R::Instruction<P>>(
            input,
            inst,
            partition,
            params.line_size_input,
            params.line_mode,
            params.bound_checks_inner,
        ),
        (None, false) => level::unit::reduce::<P, VirtualTensor<P::EI>, R::Instruction<P>>(
            input,
            partition,
            inst,
            params.line_size_input,
            params.line_mode,
        ),
    };

    writer::write::<P, Out, R::Instruction<P>>(
        output,
        accumulator,
        reduce_index,
        input.shape(axis_reduce),
        params,
        inst,
    )
}

#[cube]
fn get_reduce_index(#[comptime] params: ReduceParams) -> u32 {
    if params.shared.is_some() {
        CUBE_POS
    } else if params.use_planes {
        CUBE_POS * CUBE_DIM_Y + UNIT_POS_Y
    } else {
        ABSOLUTE_POS
    }
}

#[cube]
fn get_reduce_count(output_size: u32, #[comptime] params: ReduceParams) -> u32 {
    match comptime!(params.line_mode) {
        LineMode::Parallel => output_size,
        LineMode::Perpendicular => output_size / params.line_size_input,
    }
}
