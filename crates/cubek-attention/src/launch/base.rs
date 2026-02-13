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
use cubecl::{
    Runtime,
    client::ComputeClient,
    ir::{ElemType, FloatKind, IntKind, StorageType},
    prelude::TensorHandleRef,
    std::tensor::TensorHandle,
};

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
    let q_block_size =
        tiling.tile_size.seq_q * tiling.partition_size.seq_q * tiling.stage_size.seq_q;
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

    // Compute softmax scale:
    // - Q is quantized with sm_scale = (1/sqrt(original_head_dim)) * log2(e) baked in
    // - K is quantized without sm_scale (uses 1.0)
    // - The log2(e) factor enables using exp2() in softmax: exp2(x * log2(e)) = exp(x)
    // - This matches SageAttention's approach for INT8 quantization
    let original_head_dim = definition
        .dims
        .original_head_dim
        .unwrap_or(definition.dims.head_dim);
    let sm_scale = (1.0 / (original_head_dim as f32).sqrt()) * std::f32::consts::LOG2_E;

    // Quantize Q and K based on input dtype (using per-block quantization)
    // Q gets sm_scale baked in, K uses 1.0
    quantize_tensor_per_block(
        client,
        query,
        &q_i8.as_ref(),
        &q_scale.as_ref(),
        &q_config,
        global_dtypes.query,
        sm_scale,
    );
    quantize_tensor_per_block(
        client,
        key,
        &k_i8.as_ref(),
        &k_scale.as_ref(),
        &k_config,
        global_dtypes.key,
        1.0,
    );

    // DEBUG: Print scale values to verify quantization
    let debug_enabled = std::env::var("DEBUG_INT8_CMMA").is_ok();
    if debug_enabled {
        eprintln!("\n[DEBUG Int8Cmma] === Quantization Debug ===");
        eprintln!("[DEBUG Int8Cmma] Problem dimensions:");
        eprintln!(
            "[DEBUG Int8Cmma]   batch={}, num_heads={}",
            query.shape[0], query.shape[1]
        );
        eprintln!(
            "[DEBUG Int8Cmma]   seq_q={}, seq_kv={}, head_dim={}",
            query.shape[2], key.shape[2], query.shape[3]
        );
        eprintln!(
            "[DEBUG Int8Cmma]   original_head_dim={} (for quantization)",
            original_head_dim
        );
        eprintln!(
            "[DEBUG Int8Cmma]   blueprint.original_head_dim={} (for softmax)",
            launch_info.blueprint.original_head_dim
        );
        eprintln!("[DEBUG Int8Cmma] Block sizes:");
        eprintln!(
            "[DEBUG Int8Cmma]   q_block_size={}, k_block_size={}",
            q_block_size, k_block_size
        );
        eprintln!(
            "[DEBUG Int8Cmma]   num_q_blocks_per_head={}, num_k_blocks_per_head={}",
            num_q_blocks, num_k_blocks
        );
        eprintln!(
            "[DEBUG Int8Cmma]   total_q_scales={}, total_k_scales={}",
            q_scale_total, k_scale_total
        );
        eprintln!("[DEBUG Int8Cmma] Scale computation:");
        eprintln!(
            "[DEBUG Int8Cmma]   sm_scale = 1/sqrt({}) = {}",
            original_head_dim, sm_scale
        );

        let q_scale_bytes = client.read_one(q_scale.handle.clone());
        let q_scales: &[f32] = bytemuck::cast_slice(&q_scale_bytes);
        eprintln!(
            "[DEBUG Int8Cmma] Q scales (first 10): {:?}",
            &q_scales[..10.min(q_scales.len())]
        );
        if q_scales.len() > 10 {
            eprintln!(
                "[DEBUG Int8Cmma] Q scales (last 5): {:?}",
                &q_scales[q_scales.len().saturating_sub(5)..]
            );
        }

        let k_scale_bytes = client.read_one(k_scale.handle.clone());
        let k_scales: &[f32] = bytemuck::cast_slice(&k_scale_bytes);
        eprintln!(
            "[DEBUG Int8Cmma] K scales (first 10): {:?}",
            &k_scales[..10.min(k_scales.len())]
        );
        if k_scales.len() > 10 {
            eprintln!(
                "[DEBUG Int8Cmma] K scales (last 5): {:?}",
                &k_scales[k_scales.len().saturating_sub(5)..]
            );
        }

        // Show combined scale for first few block pairs
        eprintln!("[DEBUG Int8Cmma] Combined scales (q_scale * k_scale) for first head:");
        for (q_blk, q_scale_val) in q_scales.iter().enumerate().take(num_q_blocks.min(3)) {
            for (k_blk, k_scale_val) in k_scales.iter().enumerate().take(num_k_blocks.min(3)) {
                let combined = q_scale_val * k_scale_val;
                eprintln!(
                    "[DEBUG Int8Cmma]   q_block={}, k_block={} -> combined_scale={}",
                    q_blk, k_blk, combined
                );
            }
        }

        // Verify quantized values
        let q_i8_bytes = client.read_one(q_i8.handle.clone());
        let q_i8_vals: &[i8] = bytemuck::cast_slice(&q_i8_bytes);
        let q_i8_first_row = &q_i8_vals[..query.shape[3].min(32)];
        eprintln!(
            "[DEBUG Int8Cmma] Q_i8 first row (first 32): {:?}",
            q_i8_first_row
        );

        let k_i8_bytes = client.read_one(k_i8.handle.clone());
        let k_i8_vals: &[i8] = bytemuck::cast_slice(&k_i8_bytes);
        let k_i8_first_row = &k_i8_vals[..key.shape[3].min(32)];
        eprintln!(
            "[DEBUG Int8Cmma] K_i8 first row (first 32): {:?}",
            k_i8_first_row
        );

        // Compute expected i32 dot product for first Q row and first K row
        let first_q_row: Vec<i32> = q_i8_first_row.iter().map(|&x| x as i32).collect();
        let first_k_row: Vec<i32> = k_i8_first_row.iter().map(|&x| x as i32).collect();
        let expected_i32_dot: i32 = first_q_row
            .iter()
            .zip(first_k_row.iter())
            .map(|(a, b)| a * b)
            .sum();
        let expected_f32_score = expected_i32_dot as f32 * q_scales[0] * k_scales[0];
        eprintln!("[DEBUG Int8Cmma] Expected dot product (Q[0] · K[0]):");
        eprintln!("[DEBUG Int8Cmma]   i32 sum = {}", expected_i32_dot);
        eprintln!(
            "[DEBUG Int8Cmma]   combined_scale = {} * {} = {}",
            q_scales[0],
            k_scales[0],
            q_scales[0] * k_scales[0]
        );
        eprintln!(
            "[DEBUG Int8Cmma]   f32 score = {} * {} = {}",
            expected_i32_dot,
            q_scales[0] * k_scales[0],
            expected_f32_score
        );
        eprintln!("[DEBUG Int8Cmma] === End Quantization Debug ===\n");
    }

    // Launch attention with pre-quantized inputs
    let result = {
        <Int8CmmaRoutine as Routine>::BatchAttention::launch::<Int8TensorArgs, R>(
            client,
            launch_info.cube_dim,
            launch_info.cube_count_plan.resolve(),
            Int8TensorInputsLaunch::new(
                q_i8.as_ref()
                    .as_tensor_arg(device_settings.line_sizes.query),
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

    // DEBUG: Print output values after kernel execution
    if debug_enabled && result.is_ok() {
        eprintln!("\n[DEBUG Int8Cmma] === Post-Kernel Debug ===");
        let out_bytes = client.read_one(out.handle.clone());

        // Determine output element size based on dtype
        let elem_size = match global_dtypes.out {
            StorageType::Scalar(ElemType::Float(FloatKind::F16)) => 2,
            StorageType::Scalar(ElemType::Float(FloatKind::BF16)) => 2,
            StorageType::Scalar(ElemType::Float(FloatKind::F32)) => 4,
            _ => 4,
        };

        let out_vals: Vec<f32> = if elem_size == 2 {
            // f16 or bf16 - convert to f32
            let out_f16: &[half::f16] = bytemuck::cast_slice(&out_bytes);
            out_f16.iter().map(|x| x.to_f32()).collect()
        } else {
            bytemuck::cast_slice(&out_bytes).to_vec()
        };

        // Print first row of output
        let val_dim = definition.dims.val_dim;
        let first_row = &out_vals[..val_dim.min(32)];
        eprintln!(
            "[DEBUG Int8Cmma] Output first row (first 32): {:?}",
            first_row
        );

        // Check for NaN/Inf
        let nan_count = out_vals.iter().filter(|x| x.is_nan()).count();
        let inf_count = out_vals.iter().filter(|x| x.is_infinite()).count();
        eprintln!(
            "[DEBUG Int8Cmma] Output NaN count: {}, Inf count: {}",
            nan_count, inf_count
        );

        // Compute output statistics
        let out_min = out_vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let out_max = out_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let out_mean = out_vals.iter().sum::<f32>() / out_vals.len() as f32;
        eprintln!(
            "[DEBUG Int8Cmma] Output stats: min={}, max={}, mean={}",
            out_min, out_max, out_mean
        );
        eprintln!("[DEBUG Int8Cmma] === End Post-Kernel Debug ===\n");
    }

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
///
/// Following SageAttention, `sm_scale` is baked into the quantization.
/// For Q, this is typically 1/sqrt(head_dim). For K, this is 1.0.
fn quantize_tensor_per_block<R: Runtime>(
    client: &ComputeClient<R>,
    input: &TensorHandleRef<R>,
    output: &TensorHandleRef<R>,
    scales: &TensorHandleRef<R>,
    config: &QuantizeConfig,
    dtype: StorageType,
    sm_scale: f32,
) {
    use cubecl::ir::{ElemType, FloatKind};

    match dtype {
        StorageType::Scalar(ElemType::Float(FloatKind::F16)) => {
            launch_quantize_per_block::<R, half::f16>(
                client, input, output, scales, config, sm_scale,
            );
        }
        StorageType::Scalar(ElemType::Float(FloatKind::BF16)) => {
            launch_quantize_per_block::<R, half::bf16>(
                client, input, output, scales, config, sm_scale,
            );
        }
        StorageType::Scalar(ElemType::Float(FloatKind::F32)) => {
            launch_quantize_per_block::<R, f32>(client, input, output, scales, config, sm_scale);
        }
        _ => panic!("Unsupported input dtype for INT8 CMMA quantization: {dtype:?}"),
    }
}
