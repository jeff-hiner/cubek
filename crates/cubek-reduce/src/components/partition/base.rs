use crate::{
    LineMode,
    components::{
        partition::{parallel::partition_parallel, perpendicular::partition_perpendicular},
        precision::ReducePrecision,
    },
    routines::{ReduceBlueprint, ReduceBlueprintKind},
};
use cubecl::{prelude::*, std::tensor::r#virtual::VirtualTensor};

/// A simple range to specify how to iterate a slice when performing a reduction.
#[derive(CubeType)]
pub struct ReducePartition {
    pub index_start: u32,
    pub index_step: u32,
    pub coordinate_start: u32,
    pub coordinate_end: u32,
    pub coordinate_step: u32,
}

#[derive(Clone)]
pub struct PartitionConfig {
    pub shared: bool,
    pub use_planes: bool,
}

impl PartitionConfig {
    fn new(blueprint: ReduceBlueprint) -> Self {
        match blueprint.kind {
            ReduceBlueprintKind::Unit => Self {
                shared: false,
                use_planes: false,
            },
            ReduceBlueprintKind::Plane(..) => Self {
                shared: false,
                use_planes: true,
            },
            ReduceBlueprintKind::Cube(b) => Self {
                shared: true,
                use_planes: b.use_planes,
            },
        }
    }
}
#[cube]
impl ReducePartition {
    pub(crate) fn new<P: ReducePrecision, Out: Numeric>(
        reduce_index: u32,
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        axis_reduce: u32,
        #[comptime] blueprint: ReduceBlueprint,
    ) -> ReducePartition {
        let config = comptime!(PartitionConfig::new(blueprint));

        match comptime!(blueprint.line_mode) {
            LineMode::Parallel => partition_parallel::<P, Out>(
                reduce_index,
                input,
                output,
                axis_reduce,
                input.line_size(),
                config,
            ),
            LineMode::Perpendicular => partition_perpendicular::<P, Out>(
                reduce_index,
                input,
                output,
                axis_reduce,
                input.line_size(),
                config,
            ),
        }
    }
}
