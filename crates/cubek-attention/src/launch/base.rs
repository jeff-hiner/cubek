use crate::{
    components::batch::BatchAttentionFamily,
    definition::{
        AttentionDims, AttentionGlobalTypes, AttentionOptions, AttentionProblem,
        AttentionSetupError,
    },
    kernels::{QuantizeConfig, launch_quantize_per_block},
    launch::args::{Int8TensorArgs, Int8TensorInputsLaunch, TensorArgs, TensorInputsLaunch},
    routines::{
        DeviceSettings, Int8CmmaRoutine, Routine, SageRoutine,
        blackbox_accelerated::BlackboxAcceleratedRoutine, unit::UnitRoutine,
    },
};
use cubecl::ir::{ElemType, FloatKind, IntKind, StorageType};
use cubecl::{Runtime, client::ComputeClient, prelude::TensorHandleRef, std::tensor::TensorHandle};

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
    original_head_dim: Option<usize>,
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
        original_head_dim,
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
    original_head_dim: Option<usize>,
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
                original_head_dim,
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
            original_head_dim,
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
            original_head_dim,
        ),
        Strategy::Int8Cmma(strategy) => launch_int8_attention(
            client,
            query,
            key,
            value,
            mask,
            out,
            attention_global_types,
            strategy,
            attention_options,
            original_head_dim,
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
    original_head_dim: Option<usize>,
) -> Result<(), AttentionSetupError> {
    let definition = AttentionProblem {
        dims: AttentionDims {
            batch: query.shape[0],
            num_heads: query.shape[1],
            seq_q: query.shape[2],
            head_dim: query.shape[3],
            seq_kv: key.shape[2],
            val_dim: value.shape[3],
            original_head_dim,
        },
        masked: mask.is_some(),
        global_dtypes: global_dtypes.clone(),
        options: attention_options,
    };

    let device_settings = DeviceSettings::new(client, &definition);

    let launch_info = A::prepare(client, &definition, &device_settings, strategy)?;

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

/// Launch INT8 CMMA attention with pre-quantized Q and K.
///
/// This function:
/// 1. Allocates temporary i8 tensors for quantized Q and K
/// 2. Allocates f32 scale tensors for per-head quantization scales
/// 3. Runs quantization kernels on Q and K
/// 4. Launches attention with the pre-quantized tensors
#[allow(clippy::too_many_arguments)]
pub fn launch_int8_attention<R: Runtime>(
    client: &ComputeClient<R>,
    query: &TensorHandleRef<R>,
    key: &TensorHandleRef<R>,
    value: &TensorHandleRef<R>,
    mask: &Option<TensorHandleRef<R>>,
    out: &TensorHandleRef<R>,
    global_dtypes: &AttentionGlobalTypes,
    strategy: BlueprintStrategy<Int8CmmaRoutine>,
    attention_options: AttentionOptions,
    original_head_dim: Option<usize>,
) -> Result<(), AttentionSetupError> {
    let definition = AttentionProblem {
        dims: AttentionDims {
            batch: query.shape[0],
            num_heads: query.shape[1],
            seq_q: query.shape[2],
            head_dim: query.shape[3],
            seq_kv: key.shape[2],
            val_dim: value.shape[3],
            original_head_dim,
        },
        masked: mask.is_some(),
        global_dtypes: global_dtypes.clone(),
        options: attention_options,
    };

    let device_settings = DeviceSettings::new(client, &definition);
    let launch_info = Int8CmmaRoutine::prepare(client, &definition, &device_settings, strategy)?;

    // Extract block sizes from the blueprint's tiling scheme.
    // These determine how many blocks per head for scale tensors.
    let tiling = &launch_info.blueprint.tiling_scheme;
    let q_block_size = tiling.tile_size.seq_q * tiling.partition_size.seq_q * tiling.stage_size.seq_q;
    let k_block_size = tiling.tile_size.seq_kv * tiling.partition_size.seq_kv;
    let num_q_blocks = (query.shape[2] as u32).div_ceil(q_block_size) as usize;
    let num_k_blocks = (key.shape[2] as u32).div_ceil(k_block_size) as usize;

    // Allocate quantized Q tensor (i8) - same shape as query
    let q_i8_shape = query.shape.to_vec();
    let q_i8_strides = compute_contiguous_strides(&q_i8_shape);
    let q_i8_num_elems: usize = q_i8_shape.iter().product();
    let q_i8_handle = client.empty(q_i8_num_elems); // i8 = 1 byte
    let q_i8 = TensorHandle::new(
        q_i8_handle,
        q_i8_shape.clone(),
        q_i8_strides,
        StorageType::Scalar(ElemType::Int(IntKind::I8)),
    );

    // Allocate Q scale tensor (f32) - shape [batch * heads * num_q_blocks] flattened (per-block scale)
    let q_scale_total = query.shape[0] * query.shape[1] * num_q_blocks;
    let q_scale_shape = vec![q_scale_total];
    let q_scale_strides = vec![1];
    let q_scale_handle = client.empty(q_scale_total * core::mem::size_of::<f32>());
    let q_scale = TensorHandle::new(
        q_scale_handle,
        q_scale_shape,
        q_scale_strides,
        StorageType::Scalar(ElemType::Float(FloatKind::F32)),
    );

    // Allocate quantized K tensor (i8) - same shape as key
    let k_i8_shape = key.shape.to_vec();
    let k_i8_strides = compute_contiguous_strides(&k_i8_shape);
    let k_i8_num_elems: usize = k_i8_shape.iter().product();
    let k_i8_handle = client.empty(k_i8_num_elems); // i8 = 1 byte
    let k_i8 = TensorHandle::new(
        k_i8_handle,
        k_i8_shape.clone(),
        k_i8_strides,
        StorageType::Scalar(ElemType::Int(IntKind::I8)),
    );

    // Allocate K scale tensor (f32) - shape [batch * heads * num_k_blocks] flattened (per-block scale)
    let k_scale_total = key.shape[0] * key.shape[1] * num_k_blocks;
    let k_scale_shape = vec![k_scale_total];
    let k_scale_strides = vec![1];
    let k_scale_handle = client.empty(k_scale_total * core::mem::size_of::<f32>());
    let k_scale = TensorHandle::new(
        k_scale_handle,
        k_scale_shape,
        k_scale_strides,
        StorageType::Scalar(ElemType::Float(FloatKind::F32)),
    );

    // Run per-block quantization kernels
    let q_config = QuantizeConfig {
        dim: query.shape[3] as u32,
        seq: query.shape[2] as u32,
        line_size: device_settings.line_sizes.query as u32,
        block_size: q_block_size,
    };
    let k_config = QuantizeConfig {
        dim: key.shape[3] as u32,
        seq: key.shape[2] as u32,
        line_size: device_settings.line_sizes.key as u32,
        block_size: k_block_size,
    };

    // Quantize Q and K based on input dtype (using per-block quantization)
    quantize_tensor_per_block(client, query, &q_i8.as_ref(), &q_scale.as_ref(), &q_config, global_dtypes.query);
    quantize_tensor_per_block(client, key, &k_i8.as_ref(), &k_scale.as_ref(), &k_config, global_dtypes.key);

    // Launch attention with pre-quantized inputs
    let result = {
        <Int8CmmaRoutine as Routine>::BatchAttention::launch::<Int8TensorArgs, R>(
            client,
            launch_info.cube_dim,
            launch_info.cube_count_plan.resolve(),
            Int8TensorInputsLaunch::new(
                q_i8.as_ref().as_tensor_arg(device_settings.line_sizes.query),
                q_scale.as_ref().as_tensor_arg(1),
                k_i8.as_ref().as_tensor_arg(device_settings.line_sizes.key),
                k_scale.as_ref().as_tensor_arg(1),
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

/// Compute contiguous strides for a given shape.
fn compute_contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Dispatch per-block quantization based on input dtype.
fn quantize_tensor_per_block<R: Runtime>(
    client: &ComputeClient<R>,
    input: &TensorHandleRef<R>,
    output: &TensorHandleRef<R>,
    scales: &TensorHandleRef<R>,
    config: &QuantizeConfig,
    dtype: StorageType,
) {
    use cubecl::ir::{ElemType, FloatKind};

    match dtype {
        StorageType::Scalar(ElemType::Float(FloatKind::F16)) => {
            launch_quantize_per_block::<R, half::f16>(client, input, output, scales, config);
        }
        StorageType::Scalar(ElemType::Float(FloatKind::BF16)) => {
            launch_quantize_per_block::<R, half::bf16>(client, input, output, scales, config);
        }
        StorageType::Scalar(ElemType::Float(FloatKind::F32)) => {
            launch_quantize_per_block::<R, f32>(client, input, output, scales, config);
        }
        _ => panic!("Unsupported input dtype for INT8 CMMA quantization: {dtype:?}"),
    }
}
