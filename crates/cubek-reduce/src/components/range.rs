use crate::{LineMode, ReduceParams, components::precision::ReducePrecision};
use cubecl::prelude::*;
use cubecl::std::tensor::r#virtual::VirtualTensor;

/// A simple range to specify how to iterate a slice when performing a reduction.
#[derive(CubeType)]
pub struct ReduceRange {
    pub index_start: u32,
    pub index_step: u32,
    pub coordinate_start: u32,
    pub coordinate_end: u32,
    pub coordinate_step: u32,
}

#[cube]
impl ReduceRange {
    pub(crate) fn new<P: ReducePrecision, Out: Numeric>(
        reduce_index: u32,
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        axis_reduce: u32,
        #[comptime] params: ReduceParams,
    ) -> ReduceRange {
        match comptime!(params.line_mode) {
            LineMode::Parallel => {
                Self::new_parallel::<P, Out>(reduce_index, input, output, axis_reduce, params)
            }
            LineMode::Perpendicular => {
                Self::new_perpendicular::<P, Out>(reduce_index, input, output, axis_reduce, params)
            }
        }
    }

    fn new_parallel<P: ReducePrecision, Out: Numeric>(
        reduce_index: u32,
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        axis_reduce: u32,
        #[comptime] params: ReduceParams,
    ) -> ReduceRange {
        let shape_axis = input.shape(axis_reduce);

        let mut index_start = 0;
        for axis in 0..input.rank() {
            let coordinate = output.coordinate(reduce_index, axis);
            index_start += coordinate * input.stride(axis);
        }
        index_start /= params.line_size_input;

        let coordinate_end = shape_axis;

        let coordinate_step = if params.shared.is_some() {
            CUBE_DIM * params.line_size_input
        } else if params.use_planes {
            CUBE_DIM_X * params.line_size_input
        } else {
            params.line_size_input.runtime()
        };

        ReduceRange {
            index_start,
            index_step: 1,
            coordinate_start: 0,
            coordinate_end,
            coordinate_step,
        }
    }

    fn new_perpendicular<P: ReducePrecision, Out: Numeric>(
        reduce_index: u32,
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        axis_reduce: u32,
        #[comptime] params: ReduceParams,
    ) -> ReduceRange {
        let shape_axis = input.shape(axis_reduce);

        let mut index_start = 0;
        for axis in 0..input.rank() {
            let coordinate = output.coordinate(reduce_index * params.line_size_input, axis);
            index_start += coordinate * input.stride(axis);
        }
        index_start /= params.line_size_input;

        let index_step = input.stride(axis_reduce) / params.line_size_input;

        let coordinate_end = shape_axis;

        let coordinate_step = if params.shared.is_some() {
            CUBE_DIM
        } else if params.use_planes {
            CUBE_DIM_X
        } else {
            1_u32.runtime()
        };

        ReduceRange {
            index_start,
            index_step,
            coordinate_start: 0,
            coordinate_step,
            coordinate_end,
        }
    }
}
