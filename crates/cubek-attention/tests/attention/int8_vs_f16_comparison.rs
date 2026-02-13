//! Comparison tests between INT8 CMMA and f16 CMMA (BlackboxAccelerated) attention.
//!
//! These tests validate that INT8 quantized attention produces results within acceptable
//! tolerance of the f16 reference implementation.

use cubecl::{Runtime, TestRuntime, client::ComputeClient, frontend::CubePrimitive};
use cubek_attention::{
    definition::{
        AccumulatorPrecision, AttentionBlueprint, AttentionDims, AttentionGlobalTypes,
        AttentionIdent, AttentionOptions, AttentionPartitionSize, AttentionProblem,
        AttentionStageSize, AttentionTileSize, AttentionTilingScheme, HypercubeBlueprint,
    },
    launch::{BlueprintStrategy, Strategy, launch},
    routines::DeviceSettings,
};
use cubek_test_utils::{Distribution, StrideSpec, TestInput};

/// Configuration for INT8 CMMA tile sizes (i8×i8→i32 for Q·K^T, f16×f16→f32 for P×V).
fn int8_tile_size() -> AttentionTileSize {
    AttentionTileSize {
        seq_q: 8,
        seq_kv: 16,
        head_dim: 32,
        val_dim: 16,
    }
}

/// Configuration for BlackboxAccelerated tile sizes (f16×f16→f32 CMMA).
fn f16_tile_size() -> AttentionTileSize {
    #[cfg(target_os = "macos")]
    {
        AttentionTileSize {
            seq_q: 8,
            seq_kv: 8,
            head_dim: 8,
            val_dim: 8,
        }
    }

    #[cfg(not(target_os = "macos"))]
    AttentionTileSize {
        seq_q: 8,
        seq_kv: 16,
        head_dim: 16,
        val_dim: 16,
    }
}

/// Create an INT8 CMMA strategy with the given blueprint.
fn int8_strategy(blueprint: AttentionBlueprint) -> Strategy {
    Strategy::Int8Cmma(BlueprintStrategy::Forced(blueprint))
}

/// Create a BlackboxAccelerated (f16) strategy with the given blueprint.
fn f16_strategy(blueprint: AttentionBlueprint) -> Strategy {
    Strategy::BlackboxAccelerated(BlueprintStrategy::Forced(blueprint))
}

/// Global dtypes for f16 precision.
fn f16_global_dtypes() -> AttentionGlobalTypes {
    AttentionGlobalTypes::from_single_dtype(half::f16::as_type_native_unchecked())
}

/// Compute relative error between two tensors.
fn relative_error(expected: &[f32], actual: &[f32]) -> f32 {
    assert_eq!(expected.len(), actual.len());
    let mut max_error = 0.0f32;
    let mut max_abs = 0.0f32;

    for (e, a) in expected.iter().zip(actual.iter()) {
        let abs_err = (e - a).abs();
        max_error = max_error.max(abs_err);
        max_abs = max_abs.max(e.abs().max(a.abs()));
    }

    if max_abs < 1e-6 {
        max_error
    } else {
        max_error / max_abs
    }
}

/// Compute mean absolute error between two tensors.
fn mean_absolute_error(expected: &[f32], actual: &[f32]) -> f32 {
    assert_eq!(expected.len(), actual.len());
    let sum: f32 = expected
        .iter()
        .zip(actual.iter())
        .map(|(e, a)| (e - a).abs())
        .sum();
    sum / expected.len() as f32
}

/// Run attention with a given strategy and return the output as f32 values.
fn run_attention(
    client: &ComputeClient<TestRuntime>,
    strategy: Strategy,
    problem: &AttentionProblem,
    query_handle: cubecl::std::tensor::TensorHandle<TestRuntime>,
    key_handle: cubecl::std::tensor::TensorHandle<TestRuntime>,
    value_handle: cubecl::std::tensor::TensorHandle<TestRuntime>,
    mask_handle: Option<cubecl::std::tensor::TensorHandle<TestRuntime>>,
) -> Vec<f32> {
    let out_shape = problem.shape(AttentionIdent::Out);
    let out_handle = TestInput::zeros(
        client.clone(),
        out_shape.to_vec(),
        problem.global_dtypes.out,
        StrideSpec::RowMajor,
    )
    .generate_without_host_data();

    launch(
        strategy,
        client,
        query_handle,
        key_handle,
        value_handle,
        mask_handle,
        out_handle.clone(),
        &problem.global_dtypes,
        problem.options.clone(),
        problem.dims.original_head_dim,
    )
    .expect("Attention launch failed");

    // Read back output and convert to f32
    let out_bytes = client.read_one(out_handle.handle);
    let out_f16: &[half::f16] = bytemuck::cast_slice(&out_bytes);
    out_f16.iter().map(|x| x.to_f32()).collect()
}

/// Test comparing INT8 CMMA vs f16 attention for a single tile.
#[test]
fn test_int8_vs_f16_single_tile() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    // Check if required CMMA configs are available
    let cmma_configs = &client.properties().features.cmma;
    let i8_type =
        cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Int(cubecl::ir::IntKind::I8));
    let i32_type =
        cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Int(cubecl::ir::IntKind::I32));
    let f16_type =
        cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F16));
    let f32_type =
        cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F32));

    // Check for i8×i8→i32 CMMA
    let i8_cmma_available = [(8, 16, 32), (16, 16, 32), (16, 8, 32)]
        .iter()
        .any(|&(m, n, k)| {
            cmma_configs.contains(&cubecl::features::MmaConfig {
                a_type: i8_type,
                b_type: i8_type,
                cd_type: i32_type,
                m,
                n,
                k,
            })
        });

    // Check for f16×f16→f32 CMMA
    let f16_cmma_available = [(8, 16, 16), (8, 8, 8), (16, 16, 16)]
        .iter()
        .any(|&(m, n, k)| {
            cmma_configs.contains(&cubecl::features::MmaConfig {
                a_type: f16_type,
                b_type: f16_type,
                cd_type: f32_type,
                m,
                n,
                k,
            })
        });

    if !i8_cmma_available {
        println!("Skipping test: no i8×i8→i32 CMMA config available");
        return;
    }
    if !f16_cmma_available {
        println!("Skipping test: no f16×f16→f32 CMMA config available");
        return;
    }

    // Use a problem size that fits exactly one tile for INT8
    let int8_tile = int8_tile_size();
    let f16_tile = f16_tile_size();

    // Use dimensions compatible with both tile sizes
    let seq_q = int8_tile.seq_q as usize;
    let seq_kv = int8_tile.seq_kv as usize;
    let head_dim = int8_tile.head_dim as usize;
    let val_dim = int8_tile.val_dim as usize;

    println!(
        "Test dimensions: seq_q={}, seq_kv={}, head_dim={}, val_dim={}",
        seq_q, seq_kv, head_dim, val_dim
    );

    let global_dtypes = f16_global_dtypes();

    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q,
            seq_kv,
            head_dim,
            val_dim,
            original_head_dim: None,
        },
        masked: false,
        global_dtypes: global_dtypes.clone(),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
    };

    let device_settings = DeviceSettings::new(&client, &problem);

    // Create INT8 blueprint
    let int8_tiling = AttentionTilingScheme {
        tile_size: int8_tile,
        partition_size: AttentionPartitionSize {
            seq_q: 1,
            seq_kv: 1,
            head_dim: 1,
            val_dim: 1,
        },
        stage_size: AttentionStageSize { seq_q: 1 },
    };

    let int8_blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme: int8_tiling,
        plane_dim: device_settings.plane_dim,
        two_rows_in_array_tile: false,
        line_sizes: device_settings.line_sizes.clone(),
        masked: false,
        causal: false,
        check_bounds: int8_tiling.check_bounds(&problem.dims),
        // For INT8, sm_scale is baked into Q, so set original_head_dim=1 to avoid double scaling
        original_head_dim: 1,
    };

    // Create f16 blueprint
    let f16_tiling = AttentionTilingScheme {
        tile_size: f16_tile,
        partition_size: AttentionPartitionSize {
            seq_q: 1,
            seq_kv: 1,
            head_dim: head_dim as u32 / f16_tile.head_dim,
            val_dim: val_dim as u32 / f16_tile.val_dim,
        },
        stage_size: AttentionStageSize { seq_q: 1 },
    };

    let f16_blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme: f16_tiling,
        plane_dim: device_settings.plane_dim,
        two_rows_in_array_tile: false,
        line_sizes: device_settings.line_sizes.clone(),
        masked: false,
        causal: false,
        check_bounds: f16_tiling.check_bounds(&problem.dims),
        original_head_dim: head_dim as u32,
    };

    // Create test inputs with the same seed
    let query_shape = problem.shape(AttentionIdent::Query);
    let key_shape = problem.shape(AttentionIdent::Key);
    let value_shape = problem.shape(AttentionIdent::Value);

    let (query_handle, _) = TestInput::random(
        client.clone(),
        query_shape.to_vec(),
        global_dtypes.query,
        42, // fixed seed for reproducibility
        Distribution::Uniform(-1., 1.),
        StrideSpec::RowMajor,
    )
    .generate_with_f32_host_data();

    let (key_handle, _) = TestInput::random(
        client.clone(),
        key_shape.to_vec(),
        global_dtypes.key,
        43,
        Distribution::Uniform(-1., 1.),
        StrideSpec::RowMajor,
    )
    .generate_with_f32_host_data();

    let (value_handle, _) = TestInput::random(
        client.clone(),
        value_shape.to_vec(),
        global_dtypes.value,
        44,
        Distribution::Uniform(-1., 1.),
        StrideSpec::RowMajor,
    )
    .generate_with_f32_host_data();

    // Create duplicate handles for second run
    let (query_handle2, _) = TestInput::random(
        client.clone(),
        query_shape.to_vec(),
        global_dtypes.query,
        42, // same seed
        Distribution::Uniform(-1., 1.),
        StrideSpec::RowMajor,
    )
    .generate_with_f32_host_data();

    let (key_handle2, _) = TestInput::random(
        client.clone(),
        key_shape.to_vec(),
        global_dtypes.key,
        43,
        Distribution::Uniform(-1., 1.),
        StrideSpec::RowMajor,
    )
    .generate_with_f32_host_data();

    let (value_handle2, _) = TestInput::random(
        client.clone(),
        value_shape.to_vec(),
        global_dtypes.value,
        44,
        Distribution::Uniform(-1., 1.),
        StrideSpec::RowMajor,
    )
    .generate_with_f32_host_data();

    // Run f16 attention first (reference)
    println!("\nRunning f16 (BlackboxAccelerated) attention...");
    let f16_output = run_attention(
        &client,
        f16_strategy(f16_blueprint),
        &problem,
        query_handle,
        key_handle,
        value_handle,
        None,
    );
    println!(
        "f16 output (first 8): {:?}",
        &f16_output[..8.min(f16_output.len())]
    );

    // Run INT8 attention
    println!("\nRunning INT8 CMMA attention...");
    let int8_output = run_attention(
        &client,
        int8_strategy(int8_blueprint),
        &problem,
        query_handle2,
        key_handle2,
        value_handle2,
        None,
    );
    println!(
        "INT8 output (first 8): {:?}",
        &int8_output[..8.min(int8_output.len())]
    );

    // Compare outputs
    let rel_err = relative_error(&f16_output, &int8_output);
    let mae = mean_absolute_error(&f16_output, &int8_output);

    println!("\n===== Comparison Results =====");
    println!("Relative error: {:.4} ({:.2}%)", rel_err, rel_err * 100.0);
    println!("Mean absolute error: {:.6}", mae);

    // Check for NaN/Inf
    let f16_nan = f16_output.iter().filter(|x| x.is_nan()).count();
    let int8_nan = int8_output.iter().filter(|x| x.is_nan()).count();
    let f16_inf = f16_output.iter().filter(|x| x.is_infinite()).count();
    let int8_inf = int8_output.iter().filter(|x| x.is_infinite()).count();

    println!("f16 NaN count: {}, Inf count: {}", f16_nan, f16_inf);
    println!("INT8 NaN count: {}, Inf count: {}", int8_nan, int8_inf);

    // Per-element comparison for debugging
    println!("\nPer-element comparison (first 16):");
    for i in 0..16.min(f16_output.len()) {
        let f16_val = f16_output[i];
        let int8_val = int8_output[i];
        let err = (f16_val - int8_val).abs();
        println!(
            "  [{}] f16={:+.6}, int8={:+.6}, err={:.6}",
            i, f16_val, int8_val, err
        );
    }

    // Expected tolerance: ~10% relative error for single attention layer
    // INT8 quantization introduces ~1-2% error, but accumulates through operations
    let tolerance = 0.15; // 15% relative error
    assert!(
        rel_err < tolerance,
        "Relative error {:.4} exceeds tolerance {:.4}",
        rel_err,
        tolerance
    );
    assert_eq!(f16_nan, 0, "f16 output contains NaN");
    assert_eq!(int8_nan, 0, "INT8 output contains NaN");

    println!("\n===== TEST PASSED =====");
}

/// Test comparing INT8 CMMA vs f16 attention with larger problem size (multiple tiles).
#[test]
fn test_int8_vs_f16_multiple_tiles() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    // Check if required CMMA configs are available
    let cmma_configs = &client.properties().features.cmma;
    let i8_type =
        cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Int(cubecl::ir::IntKind::I8));
    let i32_type =
        cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Int(cubecl::ir::IntKind::I32));
    let f16_type =
        cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F16));
    let f32_type =
        cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F32));

    let i8_cmma_available = [(8, 16, 32), (16, 16, 32), (16, 8, 32)]
        .iter()
        .any(|&(m, n, k)| {
            cmma_configs.contains(&cubecl::features::MmaConfig {
                a_type: i8_type,
                b_type: i8_type,
                cd_type: i32_type,
                m,
                n,
                k,
            })
        });

    let f16_cmma_available = [(8, 16, 16), (8, 8, 8), (16, 16, 16)]
        .iter()
        .any(|&(m, n, k)| {
            cmma_configs.contains(&cubecl::features::MmaConfig {
                a_type: f16_type,
                b_type: f16_type,
                cd_type: f32_type,
                m,
                n,
                k,
            })
        });

    if !i8_cmma_available {
        println!("Skipping test: no i8×i8→i32 CMMA config available");
        return;
    }
    if !f16_cmma_available {
        println!("Skipping test: no f16×f16→f32 CMMA config available");
        return;
    }

    // Use dimensions that require multiple tiles
    let seq_q = 64;
    let seq_kv = 64;
    let head_dim = 64;
    let val_dim = 64;

    println!(
        "Test dimensions: seq_q={}, seq_kv={}, head_dim={}, val_dim={}",
        seq_q, seq_kv, head_dim, val_dim
    );

    let global_dtypes = f16_global_dtypes();

    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 4,
            seq_q,
            seq_kv,
            head_dim,
            val_dim,
            original_head_dim: None,
        },
        masked: false,
        global_dtypes: global_dtypes.clone(),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
    };

    let device_settings = DeviceSettings::new(&client, &problem);

    // Create INT8 blueprint
    let int8_tile = int8_tile_size();
    let int8_tiling = AttentionTilingScheme {
        tile_size: int8_tile,
        partition_size: AttentionPartitionSize {
            seq_q: 1,
            seq_kv: 1,
            head_dim: head_dim as u32 / int8_tile.head_dim,
            val_dim: val_dim as u32 / int8_tile.val_dim,
        },
        stage_size: AttentionStageSize {
            seq_q: seq_q as u32 / int8_tile.seq_q,
        },
    };

    let int8_blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme: int8_tiling,
        plane_dim: device_settings.plane_dim,
        two_rows_in_array_tile: false,
        line_sizes: device_settings.line_sizes.clone(),
        masked: false,
        causal: false,
        check_bounds: int8_tiling.check_bounds(&problem.dims),
        original_head_dim: 1, // INT8 bakes sm_scale into Q
    };

    // Create f16 blueprint
    let f16_tile = f16_tile_size();
    let f16_tiling = AttentionTilingScheme {
        tile_size: f16_tile,
        partition_size: AttentionPartitionSize {
            seq_q: 1,
            seq_kv: 1,
            head_dim: head_dim as u32 / f16_tile.head_dim,
            val_dim: val_dim as u32 / f16_tile.val_dim,
        },
        stage_size: AttentionStageSize {
            seq_q: seq_q as u32 / f16_tile.seq_q,
        },
    };

    let f16_blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme: f16_tiling,
        plane_dim: device_settings.plane_dim,
        two_rows_in_array_tile: false,
        line_sizes: device_settings.line_sizes.clone(),
        masked: false,
        causal: false,
        check_bounds: f16_tiling.check_bounds(&problem.dims),
        original_head_dim: head_dim as u32,
    };

    // Create test inputs
    let query_shape = problem.shape(AttentionIdent::Query);
    let key_shape = problem.shape(AttentionIdent::Key);
    let value_shape = problem.shape(AttentionIdent::Value);

    let (query_handle, _) = TestInput::random(
        client.clone(),
        query_shape.to_vec(),
        global_dtypes.query,
        42,
        Distribution::Uniform(-1., 1.),
        StrideSpec::RowMajor,
    )
    .generate_with_f32_host_data();

    let (key_handle, _) = TestInput::random(
        client.clone(),
        key_shape.to_vec(),
        global_dtypes.key,
        43,
        Distribution::Uniform(-1., 1.),
        StrideSpec::RowMajor,
    )
    .generate_with_f32_host_data();

    let (value_handle, _) = TestInput::random(
        client.clone(),
        value_shape.to_vec(),
        global_dtypes.value,
        44,
        Distribution::Uniform(-1., 1.),
        StrideSpec::RowMajor,
    )
    .generate_with_f32_host_data();

    // Create duplicate handles
    let (query_handle2, _) = TestInput::random(
        client.clone(),
        query_shape.to_vec(),
        global_dtypes.query,
        42,
        Distribution::Uniform(-1., 1.),
        StrideSpec::RowMajor,
    )
    .generate_with_f32_host_data();

    let (key_handle2, _) = TestInput::random(
        client.clone(),
        key_shape.to_vec(),
        global_dtypes.key,
        43,
        Distribution::Uniform(-1., 1.),
        StrideSpec::RowMajor,
    )
    .generate_with_f32_host_data();

    let (value_handle2, _) = TestInput::random(
        client.clone(),
        value_shape.to_vec(),
        global_dtypes.value,
        44,
        Distribution::Uniform(-1., 1.),
        StrideSpec::RowMajor,
    )
    .generate_with_f32_host_data();

    // Run f16 attention
    println!("\nRunning f16 (BlackboxAccelerated) attention...");
    let f16_output = run_attention(
        &client,
        f16_strategy(f16_blueprint),
        &problem,
        query_handle,
        key_handle,
        value_handle,
        None,
    );
    println!(
        "f16 output (first 8): {:?}",
        &f16_output[..8.min(f16_output.len())]
    );

    // Run INT8 attention
    println!("\nRunning INT8 CMMA attention...");
    let int8_output = run_attention(
        &client,
        int8_strategy(int8_blueprint),
        &problem,
        query_handle2,
        key_handle2,
        value_handle2,
        None,
    );
    println!(
        "INT8 output (first 8): {:?}",
        &int8_output[..8.min(int8_output.len())]
    );

    // Compare outputs
    let rel_err = relative_error(&f16_output, &int8_output);
    let mae = mean_absolute_error(&f16_output, &int8_output);

    println!("\n===== Comparison Results =====");
    println!("Relative error: {:.4} ({:.2}%)", rel_err, rel_err * 100.0);
    println!("Mean absolute error: {:.6}", mae);

    // Statistics by head
    let num_heads = problem.dims.num_heads;
    let elements_per_head = seq_q * val_dim;
    println!("\nPer-head relative errors:");
    for h in 0..num_heads {
        let start = h * elements_per_head;
        let end = start + elements_per_head;
        let head_f16 = &f16_output[start..end];
        let head_int8 = &int8_output[start..end];
        let head_err = relative_error(head_f16, head_int8);
        println!("  Head {}: {:.4} ({:.2}%)", h, head_err, head_err * 100.0);
    }

    // Expected tolerance: ~15% for multiple-tile case (errors accumulate)
    let tolerance = 0.20;
    assert!(
        rel_err < tolerance,
        "Relative error {:.4} exceeds tolerance {:.4}",
        rel_err,
        tolerance
    );

    println!("\n===== TEST PASSED =====");
}
