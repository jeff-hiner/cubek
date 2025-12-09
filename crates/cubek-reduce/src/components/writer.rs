use crate::{LineMode, ReduceInstruction, ReduceParams, ReducePrecision};
use cubecl::{prelude::*, std::tensor::r#virtual::VirtualTensor};

#[cube]
pub fn write<P: ReducePrecision, Out: Numeric, R: ReduceInstruction<P>>(
    output: &mut VirtualTensor<Out, ReadWrite>,
    accumulator: R::AccumulatorItem,
    reduce_index: u32,
    shape_axis_reduce: u32,
    #[comptime] settings: ReduceParams,
    inst: &R,
) {
    if elected_writer(settings) {
        write_accumulator::<P, Out, R>(
            output,
            accumulator,
            reduce_index,
            shape_axis_reduce,
            settings,
            inst,
        );
    }
}

#[cube]
fn write_accumulator<P: ReducePrecision, Out: Numeric, R: ReduceInstruction<P>>(
    output: &mut VirtualTensor<Out, ReadWrite>,
    accumulator: R::AccumulatorItem,
    reduce_index: u32,
    shape_axis_reduce: u32,
    #[comptime] settings: ReduceParams,
    inst: &R,
) {
    match comptime!(settings.line_mode) {
        LineMode::Parallel => {
            let result = R::merge_line::<Out>(inst, accumulator, shape_axis_reduce);
            output.write(reduce_index, Line::cast_from(result))
        }
        LineMode::Perpendicular => {
            let out = R::to_output_perpendicular(inst, accumulator, shape_axis_reduce);

            if comptime![settings.line_size_output == settings.line_size_input] {
                output.write(reduce_index, out);
            } else {
                let num_iters = comptime![settings.line_size_input / settings.line_size_output];

                #[unroll]
                for i in 0..num_iters {
                    let mut tmp = Line::empty(settings.line_size_output);

                    #[unroll]
                    for j in 0..settings.line_size_output {
                        tmp[j] = out[i * settings.line_size_output + j];
                    }

                    let index = num_iters * reduce_index + i;
                    output.write(index, tmp);
                }
            }
        }
    }
}

#[cube]
fn elected_writer(#[comptime] settings: ReduceParams) -> bool {
    if settings.shared.is_some() {
        UNIT_POS == 0
    } else if settings.use_planes {
        UNIT_POS_X == 0
    } else {
        true.runtime()
    }
}
