use crate::{
    components::{
        args::{ReduceArgs, TensorArgs, init_tensors},
        instructions::*,
    },
    launch::{ReduceLaunchInfo, ReduceStrategy},
    routines::{
        CubeReduceBlueprint, PlaneReduceBlueprint, ReduceBlueprint, ReduceBlueprintKind,
        reduce_kernel_virtual,
    },
};
use cubecl::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct ReduceDtypes {
    pub input: StorageType,
    pub output: StorageType,
    pub accumulation: StorageType,
}

/// Launch a reduce kernel. This function assumes that all parameters are already validated.
/// See the main entrypoint `reduce` in `lib.rs` for an example how to call this function
/// with the appropriate assumptions.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_reduce<Run: Runtime>(
    client: &ComputeClient<Run>,
    input: TensorHandleRef<Run>,
    output: TensorHandleRef<Run>,
    axis: u32,
    info: ReduceLaunchInfo,
    strategy: ReduceStrategy,
    dtypes: ReduceDtypes,
    inst: ReduceOperationConfig,
) -> Result<(), LaunchError> {
    let kind = match (strategy.shared, strategy.use_planes) {
        (true, true) => ReduceBlueprintKind::Cube(CubeReduceBlueprint {
            accumulator_size: info.cube_dim.y,
            bound_checks_inner: info.bound_checks_inner,
            use_planes: true,
        }),
        (true, false) => ReduceBlueprintKind::Cube(CubeReduceBlueprint {
            accumulator_size: info.cube_dim.num_elems(),
            bound_checks_inner: info.bound_checks_inner,
            use_planes: false,
        }),
        (false, true) => ReduceBlueprintKind::Plane(PlaneReduceBlueprint {
            bound_checks_inner: info.bound_checks_inner,
        }),
        (false, false) => ReduceBlueprintKind::Unit,
    };

    let blueprint = ReduceBlueprint {
        line_mode: info.line_mode,
        bound_checks: info.bound_checks,
        kind,
    };

    unsafe {
        reduce_kernel::launch_unchecked::<TensorArgs, Run>(
            client,
            info.cube_count,
            info.cube_dim,
            input.as_tensor_arg(info.line_size_input as u8),
            output.as_tensor_arg(info.line_size_output as u8),
            ScalarArg::new(axis),
            blueprint,
            inst,
            dtypes.input,
            dtypes.output,
            dtypes.accumulation,
        )
    }
}

#[cube(launch_unchecked)]
pub fn reduce_kernel<In: Numeric, Out: Numeric, Acc: Numeric, RA: ReduceArgs>(
    input: &RA::Input<In>,
    output: &mut RA::Output<Out>,
    axis_reduce: u32,
    #[comptime] blueprint: ReduceBlueprint,
    #[comptime] config: ReduceOperationConfig,
    #[define(In)] _input_dtype: StorageType,
    #[define(Out)] _output_dtype: StorageType,
    #[define(Acc)] _acc_dtype: StorageType,
) {
    let (input, mut output) = init_tensors::<RA, In, Out>(input, output);
    reduce_kernel_virtual::<In, Out, Acc>(&input, &mut output, axis_reduce, blueprint, config);
}
