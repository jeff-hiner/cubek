use crate::components::{
    partition::{PartitionConfig, ReducePartition},
    precision::ReducePrecision,
};
use cubecl::{prelude::*, std::tensor::r#virtual::VirtualTensor};

#[cube]
pub fn partition_perpendicular<P: ReducePrecision, Out: Numeric>(
    reduce_index: u32,
    input: &VirtualTensor<P::EI>,
    output: &mut VirtualTensor<Out, ReadWrite>,
    axis_reduce: u32,
    #[comptime] input_line_size: u32,
    #[comptime] params: PartitionConfig,
) -> ReducePartition {
    let shape_axis = input.shape(axis_reduce);

    let mut index_start = 0;
    for axis in 0..input.rank() {
        let coordinate = output.coordinate(reduce_index * input_line_size, axis);
        index_start += coordinate * input.stride(axis);
    }
    index_start /= input_line_size;

    let index_step = input.stride(axis_reduce) / input_line_size;

    let coordinate_end = shape_axis;

    let coordinate_step = if params.shared {
        CUBE_DIM
    } else if params.use_planes {
        CUBE_DIM_X
    } else {
        1_u32.runtime()
    };

    ReducePartition {
        index_start,
        index_step,
        coordinate_start: 0,
        coordinate_step,
        coordinate_end,
    }
}
