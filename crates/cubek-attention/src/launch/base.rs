use cubecl::{Runtime, client::ComputeClient, prelude::TensorHandleRef};

use cubecl::std::tensor::TensorHandle;

use crate::definition::AttentionSetupError;
use crate::definition::{AttentionDims, AttentionGlobalTypes, AttentionOptions, AttentionProblem};
use crate::launch::args::{TensorArgs, TensorInputsLaunch};
use crate::routines::DeviceSettings;
use crate::routines::{
    Int8CmmaRoutine, Routine, SageRoutine, blackbox_accelerated::BlackboxAcceleratedRoutine,
    unit::UnitRoutine,
};

use crate::components::batch::BatchAttentionFamily;

#[derive(Debug, Clone)]
pub enum BlueprintStrategy<R: Routine> {
    /// Use a predefined blueprint
    Forced(R::Blueprint),
    /// Allows to give limited settings information, and the rest is inferred from it
    Inferred(R::Strategy),
}

#[derive(Debug, Clone)]
pub enum Strategy {
    BlackboxAccelerated(BlueprintStrategy<BlackboxAcceleratedRoutine>),
    Unit(BlueprintStrategy<UnitRoutine>),
    /// SageAttention with proper INT8 quantization and scale handling.
    Sage(BlueprintStrategy<SageRoutine>),
    /// INT8 CMMA tensor core attention for Q·K^T using i8×i8→i32.
    Int8Cmma(BlueprintStrategy<Int8CmmaRoutine>),
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch<R: Runtime>(
    strategy: Strategy,
    client: &ComputeClient<R>,
    query: TensorHandle<R>,
    key: TensorHandle<R>,
    value: TensorHandle<R>,
    mask: Option<TensorHandle<R>>,
    out: TensorHandle<R>,
    attention_global_types: &AttentionGlobalTypes,
    attention_options: AttentionOptions,
) -> Result<(), AttentionSetupError> {
    launch_ref(
        strategy,
        client,
        &query.as_ref(),
        &key.as_ref(),
        &value.as_ref(),
        &mask.as_ref().map(|m| m.as_ref()),
        &out.as_ref(),
        attention_global_types,
        attention_options,
    )
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_ref<R: Runtime>(
    strategy: Strategy,
    client: &ComputeClient<R>,
    query: &TensorHandleRef<R>,
    key: &TensorHandleRef<R>,
    value: &TensorHandleRef<R>,
    mask: &Option<TensorHandleRef<R>>,
    out: &TensorHandleRef<R>,
    attention_global_types: &AttentionGlobalTypes,
    attention_options: AttentionOptions,
) -> Result<(), AttentionSetupError> {
    match strategy {
        Strategy::BlackboxAccelerated(strategy) => {
            launch_attention::<R, BlackboxAcceleratedRoutine>(
                client,
                query,
                key,
                value,
                mask,
                out,
                attention_global_types,
                strategy,
                attention_options,
            )
        }
        Strategy::Unit(strategy) => launch_attention::<R, UnitRoutine>(
            client,
            query,
            key,
            value,
            mask,
            out,
            attention_global_types,
            strategy,
            attention_options,
        ),
        Strategy::Sage(strategy) => launch_attention::<R, SageRoutine>(
            client,
            query,
            key,
            value,
            mask,
            out,
            attention_global_types,
            strategy,
            attention_options,
        ),
        Strategy::Int8Cmma(strategy) => launch_attention::<R, Int8CmmaRoutine>(
            client,
            query,
            key,
            value,
            mask,
            out,
            attention_global_types,
            strategy,
            attention_options,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn launch_attention<R: Runtime, A: Routine>(
    client: &ComputeClient<R>,
    query: &TensorHandleRef<R>,
    key: &TensorHandleRef<R>,
    value: &TensorHandleRef<R>,
    mask: &Option<TensorHandleRef<R>>,
    out: &TensorHandleRef<R>,
    global_dtypes: &AttentionGlobalTypes,
    strategy: BlueprintStrategy<A>,
    attention_options: AttentionOptions,
) -> Result<(), AttentionSetupError> {
    let definition = AttentionProblem {
        dims: AttentionDims {
            batch: query.shape[0],
            num_heads: query.shape[1],
            seq_q: query.shape[2],
            head_dim: query.shape[3],
            seq_kv: key.shape[2],
            val_dim: value.shape[3],
        },
        masked: mask.is_some(),
        global_dtypes: global_dtypes.clone(),
        options: attention_options,
    };

    eprintln!(
        "[LAUNCH] dims: batch={}, heads={}, seq_q={}, head_dim={}, seq_kv={}, val_dim={}",
        definition.dims.batch,
        definition.dims.num_heads,
        definition.dims.seq_q,
        definition.dims.head_dim,
        definition.dims.seq_kv,
        definition.dims.val_dim
    );
    eprintln!(
        "[LAUNCH] query shape: {:?}, key shape: {:?}, value shape: {:?}, out shape: {:?}",
        query.shape, key.shape, value.shape, out.shape
    );
    eprintln!(
        "[LAUNCH] dtypes: query={:?}, key={:?}, value={:?}, out={:?}",
        global_dtypes.query, global_dtypes.key, global_dtypes.value, global_dtypes.out
    );

    let device_settings = DeviceSettings::new(client, &definition);
    eprintln!(
        "[LAUNCH] line_sizes: query={}, key={}, value={}, mask={}, out={}",
        device_settings.line_sizes.query,
        device_settings.line_sizes.key,
        device_settings.line_sizes.value,
        device_settings.line_sizes.mask,
        device_settings.line_sizes.out
    );

    let launch_info = A::prepare(client, &definition, &device_settings, strategy)?;
    eprintln!(
        "[LAUNCH] cube_dim: {:?}, cube_count: {:?}",
        launch_info.cube_dim,
        launch_info.cube_count_plan.resolve()
    );
    eprintln!("[LAUNCH] dtypes: {:?}", launch_info.dtypes);

    let result = {
        <A as Routine>::BatchAttention::launch::<TensorArgs, R>(
            client,
            launch_info.cube_dim,
            launch_info.cube_count_plan.resolve(),
            TensorInputsLaunch::new(
                query.as_tensor_arg(device_settings.line_sizes.query),
                key.as_tensor_arg(device_settings.line_sizes.key),
                value.as_tensor_arg(device_settings.line_sizes.value),
                mask.as_ref()
                    .map(|it| it.as_tensor_arg(device_settings.line_sizes.mask))
                    .into(),
            ),
            out.as_tensor_arg(device_settings.line_sizes.out),
            launch_info.cube_count_plan.as_args(),
            &launch_info.dtypes,
            launch_info.blueprint,
        )
    };

    match result {
        Ok(_) => Ok(()),
        Err(err) => Err(AttentionSetupError::Execution(err)),
    }
}
