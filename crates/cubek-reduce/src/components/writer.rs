use crate::{
    LineMode, ReduceInstruction, ReducePrecision,
    routines::{ReduceBlueprint, ReduceBlueprintKind},
};
use cubecl::{prelude::*, std::tensor::r#virtual::VirtualTensor};

#[derive(Clone)]
pub struct WriterConfig {
    line_mode: LineMode,
    shared: bool,
    use_planes: bool,
}

impl WriterConfig {
    fn from_blueprint(blueprint: ReduceBlueprint) -> Self {
        match blueprint.kind {
            ReduceBlueprintKind::Unit => Self {
                line_mode: blueprint.line_mode,
                shared: false,
                use_planes: false,
            },
            ReduceBlueprintKind::Plane(..) => Self {
                line_mode: blueprint.line_mode,
                shared: false,
                use_planes: true,
            },
            ReduceBlueprintKind::Cube(cube) => Self {
                line_mode: blueprint.line_mode,
                shared: true,
                use_planes: cube.use_planes,
            },
        }
    }
}

#[cube]
pub fn write<P: ReducePrecision, Out: Numeric, R: ReduceInstruction<P>>(
    output: &mut VirtualTensor<Out, ReadWrite>,
    accumulator: R::AccumulatorItem,
    reduce_index: u32,
    shape_axis_reduce: u32,
    #[comptime] blueprint: ReduceBlueprint,
    #[comptime] input_line_size: u32,
    inst: &R,
) {
    let config = comptime!(WriterConfig::from_blueprint(blueprint));
    if elected_writer(comptime!(config.clone())) {
        write_accumulator::<P, Out, R>(
            output,
            accumulator,
            reduce_index,
            shape_axis_reduce,
            config,
            input_line_size,
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
    #[comptime] settings: WriterConfig,
    #[comptime] input_line_size: u32,
    inst: &R,
) {
    match comptime!(settings.line_mode) {
        LineMode::Parallel => {
            let result = R::merge_line::<Out>(inst, accumulator, shape_axis_reduce);
            output.write(reduce_index, Line::cast_from(result))
        }
        LineMode::Perpendicular => {
            let out = R::to_output_perpendicular(inst, accumulator, shape_axis_reduce);
            let output_line_size = output.line_size();

            if comptime![output_line_size == input_line_size] {
                output.write(reduce_index, out);
            } else {
                let num_iters = comptime![input_line_size / output_line_size];

                #[unroll]
                for i in 0..num_iters {
                    let mut tmp = Line::empty(output_line_size);

                    #[unroll]
                    for j in 0..output_line_size {
                        tmp[j] = out[i * output_line_size + j];
                    }

                    let index = num_iters * reduce_index + i;
                    output.write(index, tmp);
                }
            }
        }
    }
}

#[cube]
fn elected_writer(#[comptime] settings: WriterConfig) -> bool {
    if settings.shared {
        UNIT_POS == 0
    } else if settings.use_planes {
        UNIT_POS_X == 0
    } else {
        true.runtime()
    }
}
