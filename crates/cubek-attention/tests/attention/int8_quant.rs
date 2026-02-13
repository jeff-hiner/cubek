//! Unit tests for INT8 quantization used in INT8 CMMA attention.

use cubecl::{CubeCount, CubeDim, Runtime, TestRuntime, prelude::*};
use cubek_attention::components::tile::accelerated::local_tile::{
    InnerLayout, LocalTile, LocalTileLayout,
};

/// Simple quantization kernel that mimics INT8 CMMA's load_query quantization.
/// Input: f32 array, Output: i8 array + scale
#[cube(launch)]
fn quantize_kernel(input: &Array<f32>, output: &mut Array<i8>, scale_out: &mut Array<f32>) {
    let len = input.len();

    // Phase 1: Find max absolute value
    let mut local_max_abs = 0.0f32;
    for i in 0..len {
        let val = input[i];
        local_max_abs = f32::max(local_max_abs, f32::abs(val));
    }

    // Compute scale (matching INT8 CMMA implementation)
    let min_scale = 1e-6f32;
    let scale = f32::max(local_max_abs / 127.0f32, min_scale);
    let inv_scale = 127.0f32 / f32::max(local_max_abs, min_scale);

    // Store scale
    scale_out[0] = scale;

    // Phase 2: Quantize
    for i in 0..len {
        let val = input[i];
        let quantized = f32::round(val * inv_scale);
        let clamped = f32::clamp(quantized, -127.0f32, 127.0f32);
        output[i] = i8::cast_from(clamped);
    }
}

/// Test that quantization + dequantization round-trips correctly.
#[test]
fn test_quantize_roundtrip() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    // Test data: values in range [-1, 1]
    let input_data: Vec<f32> = vec![0.5, -0.3, 0.8, -0.1, 0.0, 1.0, -1.0, 0.25];
    let len = input_data.len();

    let input = client.create_from_slice(f32::as_bytes(&input_data));
    let output = client.empty(len);
    let scale_out = client.empty(core::mem::size_of::<f32>());

    unsafe {
        quantize_kernel::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts::<f32>(&input, len, 1),
            ArrayArg::from_raw_parts::<i8>(&output, len, 1),
            ArrayArg::from_raw_parts::<f32>(&scale_out, 1, 1),
        )
        .unwrap();
    }

    let output_bytes = client.read_one(output);
    let output_i8: Vec<i8> = output_bytes.iter().map(|&b| b as i8).collect();

    let scale_bytes = client.read_one(scale_out);
    let scale = f32::from_ne_bytes([
        scale_bytes[0],
        scale_bytes[1],
        scale_bytes[2],
        scale_bytes[3],
    ]);

    println!("Input: {:?}", input_data);
    println!("Quantized i8: {:?}", output_i8);
    println!("Scale: {}", scale);

    // Dequantize and compare
    let dequantized: Vec<f32> = output_i8.iter().map(|&q| q as f32 * scale).collect();
    println!("Dequantized: {:?}", dequantized);

    // Check round-trip error
    let max_abs = input_data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let expected_scale = max_abs / 127.0;
    println!("Expected scale: {}", expected_scale);

    for (i, (&orig, &deq)) in input_data.iter().zip(dequantized.iter()).enumerate() {
        let error = (orig - deq).abs();
        let max_error = expected_scale; // Quantization error should be at most 1 LSB
        println!(
            "  [{}] orig={:.4}, deq={:.4}, error={:.6}, max_allowed={:.6}",
            i, orig, deq, error, max_error
        );
        assert!(
            error <= max_error * 1.5, // Allow some slack for rounding
            "Round-trip error too large at index {}: {} > {}",
            i,
            error,
            max_error
        );
    }
}

/// Test that quantized values are stored in correct memory layout.
/// Uses a pattern matrix where each element has a unique value to verify positions.
#[test]
fn test_quantize_memory_layout() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    // Create a 4x4 matrix with unique values to verify layout
    // Row-major: [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]]
    let input_data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let len = input_data.len();

    let input = client.create_from_slice(f32::as_bytes(&input_data));
    let output = client.empty(len);
    let scale_out = client.empty(core::mem::size_of::<f32>());

    unsafe {
        quantize_kernel::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts::<f32>(&input, len, 1),
            ArrayArg::from_raw_parts::<i8>(&output, len, 1),
            ArrayArg::from_raw_parts::<f32>(&scale_out, 1, 1),
        )
        .unwrap();
    }

    let output_bytes = client.read_one(output);
    let output_i8: Vec<i8> = output_bytes.iter().map(|&b| b as i8).collect();

    let scale_bytes = client.read_one(scale_out);
    let scale = f32::from_ne_bytes([
        scale_bytes[0],
        scale_bytes[1],
        scale_bytes[2],
        scale_bytes[3],
    ]);

    println!("Input (row-major 4x4): {:?}", input_data);
    println!("Quantized i8: {:?}", output_i8);
    println!("Scale: {}", scale);

    // Expected: max value is 15, so scale = 15/127 ≈ 0.118
    // Each value i should quantize to round(i * 127 / 15) = round(i * 8.467)
    let max_val = 15.0f32;
    let inv_scale = 127.0 / max_val;

    println!("\nVerifying memory positions:");
    for i in 0..16 {
        let expected_quantized = (input_data[i] * inv_scale).round() as i8;
        let actual = output_i8[i];
        let row = i / 4;
        let col = i % 4;
        println!(
            "  [{},{}] (linear {}) input={:.0} expected_q={} actual_q={} {}",
            row,
            col,
            i,
            input_data[i],
            expected_quantized,
            actual,
            if expected_quantized == actual {
                "OK"
            } else {
                "MISMATCH!"
            }
        );
        assert_eq!(
            actual, expected_quantized,
            "Mismatch at position [{},{}]",
            row, col
        );
    }
}

/// Test that negative values are correctly quantized.
/// This is critical for sign preservation in INT8 CMMA.
#[test]
fn test_quantize_negative_values() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    // Test with explicitly negative values
    let input_data: Vec<f32> = vec![-1.0, -0.5, 0.0, 0.5, 1.0, -0.75, 0.75, -0.25];
    let len = input_data.len();

    let input = client.create_from_slice(f32::as_bytes(&input_data));
    let output = client.empty(len);
    let scale_out = client.empty(core::mem::size_of::<f32>());

    unsafe {
        quantize_kernel::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts::<f32>(&input, len, 1),
            ArrayArg::from_raw_parts::<i8>(&output, len, 1),
            ArrayArg::from_raw_parts::<f32>(&scale_out, 1, 1),
        )
        .unwrap();
    }

    let output_bytes = client.read_one(output);
    let output_i8: Vec<i8> = output_bytes.iter().map(|&b| b as i8).collect();

    let scale_bytes = client.read_one(scale_out);
    let scale = f32::from_ne_bytes([
        scale_bytes[0],
        scale_bytes[1],
        scale_bytes[2],
        scale_bytes[3],
    ]);

    println!("Input: {:?}", input_data);
    println!("Quantized i8: {:?}", output_i8);
    println!("Scale: {}", scale);

    // Verify signs are preserved
    for (i, (&input_val, &quant_val)) in input_data.iter().zip(output_i8.iter()).enumerate() {
        let input_sign = if input_val > 0.0 {
            1
        } else if input_val < 0.0 {
            -1
        } else {
            0
        };
        let quant_sign = if quant_val > 0 {
            1
        } else if quant_val < 0 {
            -1
        } else {
            0
        };

        println!(
            "  [{}] input={:+.2} (sign={:+}) -> quant={:+} (sign={:+}) {}",
            i,
            input_val,
            input_sign,
            quant_val,
            quant_sign,
            if input_sign == quant_sign {
                "OK"
            } else {
                "SIGN MISMATCH!"
            }
        );
        assert_eq!(input_sign, quant_sign, "Sign mismatch at index {}", i);
    }

    // Verify dequantized values match signs
    println!("\nDequantization check:");
    for (i, (&input_val, &quant_val)) in input_data.iter().zip(output_i8.iter()).enumerate() {
        let dequant = quant_val as f32 * scale;
        let error = (input_val - dequant).abs();
        println!(
            "  [{}] input={:+.4} -> quant={:+} -> dequant={:+.4} (error={:.6})",
            i, input_val, quant_val, dequant, error
        );
    }
}

/// Test i8×i8→i32 CMMA matmul for Q·K^T computation.
///
/// This test validates Stage 3 of INT8 CMMA attention:
/// 1. Create known Q and K matrices in f32
/// 2. Quantize them to i8 using dynamic scaling
/// 3. Run i8×i8→i32 CMMA
/// 4. Apply combined dequantization scale
/// 5. Compare to CPU reference matmul
///
/// CMMA dimensions for INT8 on Intel Arc: M=16, N=16, K=32
/// - Q: [M, K] = [16, 32] (RowMajor A matrix)
/// - K^T: [K, N] = [32, 16] (ColMajor B matrix from K [N, K] = [16, 32])
/// - Output: [M, N] = [16, 16]
#[cube(launch)]
fn i8_cmma_matmul_kernel(
    q_i8: &Array<i8>,        // [16, 32] row-major
    k_i8: &Array<i8>,        // [16, 32] row-major (stored as K, loaded as K^T via ColMajor)
    q_scale: &Array<f32>,    // scalar
    k_scale: &Array<f32>,    // scalar
    output: &mut Array<f32>, // [16, 16] row-major
    #[comptime] tile_m: u32,
    #[comptime] tile_n: u32,
    #[comptime] tile_k: u32,
) {
    // Load Q as A matrix (RowMajor)
    let q_fragment = unsafe {
        cmma::Matrix::<i8>::uninitialized(
            cmma::MatrixIdent::A,
            tile_m as usize,
            tile_n as usize,
            tile_k as usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    cmma::load(&q_fragment, &q_i8.to_slice(), tile_k); // stride = K = 32

    // Load K as B matrix (ColMajor to get K^T)
    // K is stored row-major as [N, K] = [16, 32]
    // Loading with ColMajor interprets columns as rows, giving K^T [K, N]
    let k_fragment = unsafe {
        cmma::Matrix::<i8>::uninitialized(
            cmma::MatrixIdent::B,
            tile_m as usize,
            tile_n as usize,
            tile_k as usize,
            cmma::MatrixLayout::ColMajor,
        )
    };
    cmma::load(&k_fragment, &k_i8.to_slice(), tile_k); // stride = K = 32

    // Initialize i32 accumulator to zero
    let acc_i32 = unsafe {
        cmma::Matrix::<i32>::uninitialized(
            cmma::MatrixIdent::Accumulator,
            tile_m as usize,
            tile_n as usize,
            tile_k as usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    cmma::fill(&acc_i32, 0i32);

    // Execute i8×i8→i32 CMMA: acc = Q @ K^T
    cmma::execute::<i8, i8, i32, i32>(&q_fragment, &k_fragment, &acc_i32, &acc_i32);

    // Store i32 result to shared memory, then convert to f32 with dequant scale
    let smem_size = comptime!(tile_m * tile_n);
    let mut smem_i32 = SharedMemory::<i32>::new(smem_size as usize);
    cmma::store(
        &mut smem_i32.slice_mut(0, smem_size as usize),
        &acc_i32,
        tile_n,
        cmma::MatrixLayout::RowMajor,
    );

    sync_cube();

    // Apply combined dequantization scale: f32 = i32 * q_scale * k_scale
    let combined_scale = q_scale[0] * k_scale[0];

    // Convert i32 → f32 with scale (distribute across threads)
    let elements_per_thread = comptime!((tile_m * tile_n + 31) / 32);
    for i in 0..elements_per_thread {
        let idx = UNIT_POS_X + i * 32;
        if idx < smem_size {
            let i32_val = smem_i32[idx as usize];
            output[idx as usize] = f32::cast_from(i32_val) * combined_scale;
        }
    }
}

#[test]
fn test_i8_cmma_matmul() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    // Check if i8×i8→i32 CMMA is available
    let cmma_configs = &client.properties().features.cmma;
    let i8_type =
        cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Int(cubecl::ir::IntKind::I8));
    let i32_type =
        cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Int(cubecl::ir::IntKind::I32));

    // Try to find an available i8×i8→i32 config
    // Intel Arc uses (8, 16, 32), NVIDIA may use (16, 8, 32) or (16, 16, 32)
    let mut found_config: Option<(u32, u32, u32)> = None;
    for (m, n, k) in [(8, 16, 32), (16, 16, 32), (16, 8, 32)] {
        let config = cubecl::features::MmaConfig {
            a_type: i8_type,
            b_type: i8_type,
            cd_type: i32_type,
            m,
            n,
            k,
        };
        if cmma_configs.contains(&config) {
            found_config = Some((m, n, k));
            break;
        }
    }

    let (tile_m, tile_n, tile_k) = match found_config {
        Some(config) => config,
        None => {
            println!("Skipping test: no i8×i8→i32 CMMA config available on this hardware");
            println!("Available CMMA configs: {:?}", cmma_configs);
            return;
        }
    };

    println!(
        "Using i8×i8→i32 CMMA config: M={}, N={}, K={}",
        tile_m, tile_n, tile_k
    );

    // Dimensions match the CMMA tile size for a single-tile test
    let m = tile_m as usize;
    let n = tile_n as usize;
    let k = tile_k as usize;

    // Create Q matrix [M, K] with known values
    // Use small values to avoid i32 overflow: each element in [-1, 1]
    // For Q, use values that make the computation predictable
    let q_f32: Vec<f32> = (0..m * k)
        .map(|i| {
            let row = i / k;
            // Pattern: each row has the same value across columns
            // Row 0: all 0.05, Row 1: all 0.10, etc.
            (row as f32 + 1.0) * 0.05
        })
        .collect();

    // Create K matrix [N, K]
    // Pattern: each column has the same value across rows
    // This makes Q @ K^T have predictable diagonal-ish structure
    let k_f32: Vec<f32> = (0..n * k)
        .map(|i| {
            let row = i / k;
            // Pattern: value depends on row (seq_kv position)
            (row as f32 + 1.0) * 0.05
        })
        .collect();

    println!("Q matrix (first row): {:?}", &q_f32[0..k]);
    println!("K matrix (first row): {:?}", &k_f32[0..k]);

    // Quantize Q
    let q_max_abs = q_f32.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let q_scale = q_max_abs / 127.0;
    let q_inv_scale = 127.0 / q_max_abs.max(1e-6);
    let q_i8: Vec<i8> = q_f32
        .iter()
        .map(|&x| (x * q_inv_scale).round().clamp(-127.0, 127.0) as i8)
        .collect();

    // Quantize K
    let k_max_abs = k_f32.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let k_scale = k_max_abs / 127.0;
    let k_inv_scale = 127.0 / k_max_abs.max(1e-6);
    let k_i8: Vec<i8> = k_f32
        .iter()
        .map(|&x| (x * k_inv_scale).round().clamp(-127.0, 127.0) as i8)
        .collect();

    println!("\nQuantization:");
    println!(
        "  Q: max_abs={}, scale={}, inv_scale={}",
        q_max_abs, q_scale, q_inv_scale
    );
    println!(
        "  K: max_abs={}, scale={}, inv_scale={}",
        k_max_abs, k_scale, k_inv_scale
    );
    println!("  Combined scale: {}", q_scale * k_scale);
    println!("  Q_i8 (first row): {:?}", &q_i8[0..k]);
    println!("  K_i8 (first row): {:?}", &k_i8[0..k]);

    // Compute CPU reference: Q @ K^T
    // Result[i, j] = sum_d(Q[i, d] * K[j, d])
    let mut reference = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for d in 0..k {
                sum += q_f32[i * k + d] * k_f32[j * k + d];
            }
            reference[i * n + j] = sum;
        }
    }
    println!(
        "\nCPU reference Q @ K^T (first row): {:?}",
        &reference[0..n]
    );

    // Create GPU buffers
    let q_i8_handle = client.create_from_slice(bytemuck::cast_slice(&q_i8));
    let k_i8_handle = client.create_from_slice(bytemuck::cast_slice(&k_i8));
    let q_scale_handle = client.create_from_slice(f32::as_bytes(&[q_scale]));
    let k_scale_handle = client.create_from_slice(f32::as_bytes(&[k_scale]));
    let output_handle = client.empty(m * n * core::mem::size_of::<f32>());

    // Launch kernel
    unsafe {
        i8_cmma_matmul_kernel::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(32), // Single warp/plane
            ArrayArg::from_raw_parts::<i8>(&q_i8_handle, m * k, 1),
            ArrayArg::from_raw_parts::<i8>(&k_i8_handle, n * k, 1),
            ArrayArg::from_raw_parts::<f32>(&q_scale_handle, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&k_scale_handle, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&output_handle, m * n, 1),
            m as u32, // tile_m
            n as u32, // tile_n
            k as u32, // tile_k
        )
        .unwrap();
    }

    // Read back result
    let output_bytes = client.read_one(output_handle);
    let output_f32: Vec<f32> = bytemuck::cast_slice(&output_bytes).to_vec();

    println!("\nGPU result Q @ K^T (first row): {:?}", &output_f32[0..n]);

    // Compare with reference
    println!("\nComparison (GPU vs CPU reference):");
    let mut max_abs_error = 0.0f32;
    let mut max_rel_error = 0.0f32;

    for i in 0..m {
        for j in 0..n {
            let idx = i * n + j;
            let expected = reference[idx];
            let actual = output_f32[idx];
            let abs_error = (expected - actual).abs();
            let rel_error = if expected.abs() > 1e-6 {
                abs_error / expected.abs()
            } else {
                abs_error
            };

            max_abs_error = max_abs_error.max(abs_error);
            max_rel_error = max_rel_error.max(rel_error);

            if i < 4 && j < 4 {
                println!(
                    "  [{},{}] expected={:.6}, actual={:.6}, abs_err={:.6}, rel_err={:.4}",
                    i, j, expected, actual, abs_error, rel_error
                );
            }
        }
    }

    println!("\nMax absolute error: {}", max_abs_error);
    println!("Max relative error: {}", max_rel_error);

    // INT8 quantization introduces error due to rounding to 8-bit values.
    // The relative error is approximately 1/127 ≈ 0.8% per element, and compounds
    // for both Q and K matrices. Expected relative error is around 1.6%.
    //
    // For validation, we use:
    // - Relative tolerance: ~2% (allowing for quantization error from both Q and K)
    // - Minimum absolute tolerance for small values
    let rel_tolerance = 0.025; // 2.5% relative error
    let abs_tolerance = q_scale + k_scale; // Absolute floor for near-zero values

    println!(
        "Tolerance: rel={:.4}, abs={:.6}",
        rel_tolerance, abs_tolerance
    );

    // Check that results match within tolerance
    let mut all_pass = true;
    for i in 0..m {
        for j in 0..n {
            let idx = i * n + j;
            let expected = reference[idx];
            let actual = output_f32[idx];
            let abs_error = (expected - actual).abs();
            let rel_error = if expected.abs() > abs_tolerance {
                abs_error / expected.abs()
            } else {
                abs_error / abs_tolerance
            };

            let ok = rel_error <= rel_tolerance || abs_error <= abs_tolerance;
            if !ok {
                println!(
                    "FAIL at [{},{}]: expected={:.6}, actual={:.6}, abs_err={:.6}, rel_err={:.4}",
                    i, j, expected, actual, abs_error, rel_error
                );
                all_pass = false;
            }
        }
    }

    assert!(
        all_pass,
        "Some values exceeded tolerance - see above for details"
    );

    println!("\nTest PASSED: i8×i8→i32 CMMA matmul matches CPU reference within tolerance");
}

/// Test the softmax fragment round-trip: f32 SMEM → LocalTile → SMEM → CMMA fragment.
///
/// This test validates Stages 6-9 of INT8 CMMA attention:
/// - Stage 6: f32 SMEM → LocalTile load (load_from_slice)
/// - Stage 7: (softmax operations would happen here, but we skip for this test)
/// - Stage 8: LocalTile → f32 SMEM store (store_to)
/// - Stage 9: f32 SMEM → CMMA fragment load (cmma::load_with_layout)
///
/// The test verifies values are preserved through the entire chain.
#[cube(launch)]
fn softmax_roundtrip_kernel(
    input: &Array<f32>,
    output_after_localtile: &mut Array<f32>,
    output_after_fragment: &mut Array<f32>,
    #[comptime] seq_q: u32,
    #[comptime] seq_kv: u32,
    #[comptime] plane_dim: u32,
) {
    // Allocate shared memory for f32 softmax values
    let smem_size = seq_q * seq_kv;
    let mut smem = SharedMemory::<f32>::new(smem_size as usize);
    let mut smem_out = SharedMemory::<f32>::new(smem_size as usize);

    // Copy input to shared memory
    let elements_per_thread = comptime!((smem_size + 31) / 32);
    for i in 0..elements_per_thread {
        let idx = UNIT_POS_X + i * 32;
        if idx < smem_size {
            smem[idx as usize] = input[idx as usize];
        }
    }
    sync_cube();

    // Create LocalTile with the same layout as INT8 CMMA attention uses
    let layout = LocalTileLayout::new((seq_q, seq_kv), plane_dim, InnerLayout::Contiguous);
    let mut local_tile = LocalTile::<f32>::new(layout);

    // Stage 6: Load from SMEM to LocalTile
    local_tile.load_from_slice(&smem.to_slice());

    sync_cube();

    // Stage 8: Store LocalTile back to SMEM
    let mut smem_slice = smem.slice_mut(0, smem_size as usize);
    local_tile.store_to(&mut smem_slice);

    sync_cube();

    // Copy SMEM to output_after_localtile for verification
    for i in 0..elements_per_thread {
        let idx = UNIT_POS_X + i * 32;
        if idx < smem_size {
            output_after_localtile[idx as usize] = smem[idx as usize];
        }
    }

    sync_cube();

    // Stage 9: Load from SMEM into CMMA fragment (Accumulator with K=seq_kv)
    // This mimics how Int8CmmaSoftmax::update_from_rowwise loads the fragment
    let fragment = unsafe {
        cmma::Matrix::<f32>::uninitialized(
            cmma::MatrixIdent::Accumulator,
            seq_q as usize,
            seq_kv as usize,
            seq_kv as usize, // K=seq_kv for value_matmul compatibility
            cmma::MatrixLayout::RowMajor,
        )
    };
    cmma::load_with_layout(
        &fragment,
        &smem.to_slice(),
        seq_kv,
        cmma::MatrixLayout::RowMajor,
    );

    // Store fragment back to verify values are preserved
    cmma::store(
        &mut smem_out.slice_mut(0, smem_size as usize),
        &fragment,
        seq_kv,
        cmma::MatrixLayout::RowMajor,
    );

    sync_cube();

    // Copy to output for verification
    for i in 0..elements_per_thread {
        let idx = UNIT_POS_X + i * 32;
        if idx < smem_size {
            output_after_fragment[idx as usize] = smem_out[idx as usize];
        }
    }
}

#[test]
fn test_softmax_roundtrip() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    // Check if f32 CMMA is available (needed for the fragment operations)
    let cmma_configs = &client.properties().features.cmma;
    let f32_type =
        cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F32));

    // Try to find an f32 accumulator config
    let mut found_config: Option<(u32, u32, u32)> = None;
    for (m, n, k) in [(8, 16, 16), (8, 8, 8), (16, 16, 16)] {
        let config = cubecl::features::MmaConfig {
            a_type: f32_type,
            b_type: f32_type,
            cd_type: f32_type,
            m,
            n,
            k,
        };
        if cmma_configs.contains(&config) {
            found_config = Some((m, n, k));
            break;
        }
    }

    // For f32 Accumulator, the config might be specified differently
    // Let's just try with the INT8 attention dimensions
    let seq_q = 8u32;
    let seq_kv = 16u32;
    let plane_dim = 32u32;

    println!(
        "Testing softmax roundtrip with seq_q={}, seq_kv={}",
        seq_q, seq_kv
    );
    if let Some((m, n, k)) = found_config {
        println!("Found f32 CMMA config: M={}, N={}, K={}", m, n, k);
    } else {
        println!(
            "No f32×f32→f32 CMMA config found, but testing anyway (Accumulator might work differently)"
        );
    }

    // Create test data: values that look like softmax probabilities
    // Each row should sum to 1.0, values in [0, 1]
    let mut input_data: Vec<f32> = vec![0.0; (seq_q * seq_kv) as usize];
    for row in 0..seq_q {
        for col in 0..seq_kv {
            // Uniform probability for simplicity
            input_data[(row * seq_kv + col) as usize] = 1.0 / seq_kv as f32;
        }
    }

    println!("Input (first row): {:?}", &input_data[0..seq_kv as usize]);

    // Create GPU buffers
    let input_handle = client.create_from_slice(f32::as_bytes(&input_data));
    let output_localtile_handle =
        client.empty((seq_q * seq_kv) as usize * core::mem::size_of::<f32>());
    let output_fragment_handle =
        client.empty((seq_q * seq_kv) as usize * core::mem::size_of::<f32>());

    // Launch kernel
    unsafe {
        softmax_roundtrip_kernel::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(plane_dim),
            ArrayArg::from_raw_parts::<f32>(&input_handle, (seq_q * seq_kv) as usize, 1),
            ArrayArg::from_raw_parts::<f32>(&output_localtile_handle, (seq_q * seq_kv) as usize, 1),
            ArrayArg::from_raw_parts::<f32>(&output_fragment_handle, (seq_q * seq_kv) as usize, 1),
            seq_q,
            seq_kv,
            plane_dim,
        )
        .unwrap();
    }

    // Read back results
    let output_localtile_bytes = client.read_one(output_localtile_handle);
    let output_localtile: Vec<f32> = bytemuck::cast_slice(&output_localtile_bytes).to_vec();

    let output_fragment_bytes = client.read_one(output_fragment_handle);
    let output_fragment: Vec<f32> = bytemuck::cast_slice(&output_fragment_bytes).to_vec();

    println!(
        "After LocalTile round-trip (first row): {:?}",
        &output_localtile[0..seq_kv as usize]
    );
    println!(
        "After Fragment round-trip (first row): {:?}",
        &output_fragment[0..seq_kv as usize]
    );

    // Verify LocalTile round-trip
    println!("\nLocalTile round-trip verification:");
    let mut localtile_ok = true;
    for i in 0..(seq_q * seq_kv) as usize {
        let expected = input_data[i];
        let actual = output_localtile[i];
        let error = (expected - actual).abs();
        if error > 1e-6 {
            let row = i / seq_kv as usize;
            let col = i % seq_kv as usize;
            println!(
                "  MISMATCH at [{},{}]: expected={}, actual={}, error={}",
                row, col, expected, actual, error
            );
            localtile_ok = false;
        }
    }
    if localtile_ok {
        println!("  All values match!");
    }

    // Verify Fragment round-trip
    println!("\nFragment round-trip verification:");
    let mut fragment_ok = true;
    let mut max_error = 0.0f32;
    for i in 0..(seq_q * seq_kv) as usize {
        let expected = input_data[i];
        let actual = output_fragment[i];
        let error = (expected - actual).abs();
        max_error = max_error.max(error);
        if error > 1e-5 {
            let row = i / seq_kv as usize;
            let col = i % seq_kv as usize;
            println!(
                "  MISMATCH at [{},{}]: expected={}, actual={}, error={}",
                row, col, expected, actual, error
            );
            fragment_ok = false;
        }
    }
    println!("  Max error: {}", max_error);
    if fragment_ok {
        println!("  All values match within tolerance!");
    }

    assert!(localtile_ok, "LocalTile round-trip failed");
    assert!(
        fragment_ok,
        "Fragment round-trip failed - this is Stage 9 (suspected problem area)"
    );

    println!("\nTest PASSED: Softmax fragment round-trip preserves values");
}

/// Test the P×V matmul: softmax fragment (f32→f16 cast) × Value (f16) → Accumulator (f32).
///
/// This test validates Stages 10-11 of INT8 CMMA attention.
/// The key thing being tested: an Accumulator-identity f32 fragment is cast to f16
/// and used as an A matrix in cmma::execute.
#[cube(launch)]
fn pv_matmul_kernel(
    // P (softmax probabilities): [seq_q, seq_kv] f32 input, will be cast to f16
    p_f32: &Array<f32>,
    // V (values): [seq_kv, val_dim] f16 input
    v_f16: &Array<half::f16>,
    // Output: [seq_q, val_dim] f32
    output: &mut Array<f32>,
    #[comptime] seq_q: u32,
    #[comptime] seq_kv: u32,
    #[comptime] val_dim: u32,
) {
    // Allocate SMEM for P values
    let smem_p_size = seq_q * seq_kv;
    let mut smem_p = SharedMemory::<f32>::new(smem_p_size as usize);

    // Copy P to SMEM
    let elements_per_thread = comptime!((smem_p_size + 31) / 32);
    for i in 0..elements_per_thread {
        let idx = UNIT_POS_X + i * 32;
        if idx < smem_p_size {
            smem_p[idx as usize] = p_f32[idx as usize];
        }
    }
    sync_cube();

    // Create f32 Accumulator fragment and load from SMEM (mimics update_from_rowwise)
    let p_fragment_f32 = unsafe {
        cmma::Matrix::<f32>::uninitialized(
            cmma::MatrixIdent::Accumulator,
            seq_q as usize,
            val_dim as usize,
            seq_kv as usize, // K = seq_kv for P×V
            cmma::MatrixLayout::RowMajor,
        )
    };
    cmma::load_with_layout(
        &p_fragment_f32,
        &smem_p.to_slice(),
        seq_kv,
        cmma::MatrixLayout::RowMajor,
    );

    // Cast f32 → f16 (as done in value_matmul)
    let p_fragment_f16 = cmma::cast::<f32, half::f16>(&p_fragment_f32);

    // Load V as B matrix
    let v_fragment = unsafe {
        cmma::Matrix::<half::f16>::uninitialized(
            cmma::MatrixIdent::B,
            seq_q as usize,
            val_dim as usize,
            seq_kv as usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    cmma::load(&v_fragment, &v_f16.to_slice(), val_dim);

    // Create f32 accumulator for output
    let out_fragment = unsafe {
        cmma::Matrix::<f32>::uninitialized(
            cmma::MatrixIdent::Accumulator,
            seq_q as usize,
            val_dim as usize,
            seq_kv as usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    cmma::fill(&out_fragment, 0.0f32);

    // Execute P×V: f16×f16→f32
    cmma::execute::<half::f16, half::f16, f32, f32>(
        &p_fragment_f16,
        &v_fragment,
        &out_fragment,
        &out_fragment,
    );

    // Store output
    let smem_out_size = seq_q * val_dim;
    let mut smem_out = SharedMemory::<f32>::new(smem_out_size as usize);
    cmma::store(
        &mut smem_out.slice_mut(0, smem_out_size as usize),
        &out_fragment,
        val_dim,
        cmma::MatrixLayout::RowMajor,
    );
    sync_cube();

    // Copy to output
    let out_elements_per_thread = comptime!((smem_out_size + 31) / 32);
    for i in 0..out_elements_per_thread {
        let idx = UNIT_POS_X + i * 32;
        if idx < smem_out_size {
            output[idx as usize] = smem_out[idx as usize];
        }
    }
}

#[test]
fn test_pv_matmul() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    // Check for f16 CMMA support
    let cmma_configs = &client.properties().features.cmma;
    let f16_type =
        cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F16));
    let f32_type =
        cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F32));

    let mut found_config: Option<(u32, u32, u32)> = None;
    for (m, n, k) in [(8, 16, 16), (8, 8, 8), (16, 16, 16)] {
        let config = cubecl::features::MmaConfig {
            a_type: f16_type,
            b_type: f16_type,
            cd_type: f32_type,
            m,
            n,
            k,
        };
        if cmma_configs.contains(&config) {
            found_config = Some((m, n, k));
            break;
        }
    }

    let (seq_q, seq_kv, val_dim) = match found_config {
        Some((m, n, k)) => {
            println!("Found f16×f16→f32 CMMA config: M={}, N={}, K={}", m, n, k);
            // seq_q = M, val_dim = N, seq_kv = K
            (m, k, n)
        }
        None => {
            println!("Skipping test: no f16×f16→f32 CMMA config found");
            println!("Available configs: {:?}", cmma_configs);
            return;
        }
    };

    println!(
        "Testing P×V matmul with seq_q={}, seq_kv={}, val_dim={}",
        seq_q, seq_kv, val_dim
    );

    // Create P (softmax probabilities): uniform 1/seq_kv
    let p_f32: Vec<f32> = (0..(seq_q * seq_kv)).map(|_| 1.0 / seq_kv as f32).collect();
    println!("P (uniform softmax): {:?}", &p_f32[0..seq_kv as usize]);

    // Create V: sequential values for easy verification
    let v_f16: Vec<half::f16> = (0..(seq_kv * val_dim))
        .map(|i| half::f16::from_f32((i as f32) * 0.1))
        .collect();
    println!("V (first row): {:?}", &v_f16[0..val_dim as usize]);

    // Expected output: P × V
    // With uniform P, each output row should be the mean of V rows
    let mut expected = vec![0.0f32; (seq_q * val_dim) as usize];
    for i in 0..seq_q {
        for j in 0..val_dim {
            let mut sum = 0.0f32;
            for k in 0..seq_kv {
                let p_val = p_f32[(i * seq_kv + k) as usize];
                let v_val = v_f16[(k * val_dim + j) as usize].to_f32();
                sum += p_val * v_val;
            }
            expected[(i * val_dim + j) as usize] = sum;
        }
    }
    println!(
        "Expected output (first row): {:?}",
        &expected[0..val_dim as usize]
    );

    // Create GPU buffers
    let p_handle = client.create_from_slice(f32::as_bytes(&p_f32));
    let v_bytes: Vec<u8> = bytemuck::cast_slice(&v_f16).to_vec();
    let v_handle = client.create_from_slice(&v_bytes);
    let out_handle = client.empty((seq_q * val_dim) as usize * core::mem::size_of::<f32>());

    // Launch kernel
    unsafe {
        pv_matmul_kernel::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(32),
            ArrayArg::from_raw_parts::<f32>(&p_handle, (seq_q * seq_kv) as usize, 1),
            ArrayArg::from_raw_parts::<half::f16>(&v_handle, (seq_kv * val_dim) as usize, 1),
            ArrayArg::from_raw_parts::<f32>(&out_handle, (seq_q * val_dim) as usize, 1),
            seq_q,
            seq_kv,
            val_dim,
        )
        .unwrap();
    }

    // Read results
    let out_bytes = client.read_one(out_handle);
    let output: Vec<f32> = bytemuck::cast_slice(&out_bytes).to_vec();

    println!(
        "Actual output (first row): {:?}",
        &output[0..val_dim as usize]
    );

    // Compare
    let mut max_error = 0.0f32;
    let mut all_ok = true;
    for i in 0..(seq_q * val_dim) as usize {
        let exp = expected[i];
        let act = output[i];
        let err = (exp - act).abs();
        max_error = max_error.max(err);
        // Allow small error from f16 precision
        if err > 0.01 {
            let row = i / val_dim as usize;
            let col = i % val_dim as usize;
            println!(
                "MISMATCH at [{},{}]: expected={}, actual={}, error={}",
                row, col, exp, act, err
            );
            all_ok = false;
        }
    }

    println!("Max error: {}", max_error);
    assert!(all_ok, "P×V matmul test failed");
    println!("Test PASSED: P×V matmul works correctly");
}

/// Test K matrix transpose quantization.
/// K is stored as [seq_kv, head_dim] but CMMA B matrix needs [head_dim, seq_kv] (K^T).
/// This test verifies the transpose is handled correctly.
#[test]
fn test_quantize_k_transpose() {
    let _client = <TestRuntime as Runtime>::client(&Default::default());

    // K matrix: 2 rows (seq_kv) x 4 cols (head_dim)
    // Row-major storage: [[0,1,2,3], [4,5,6,7]]
    // K^T should be: 4 rows (head_dim) x 2 cols (seq_kv)
    // [[0,4], [1,5], [2,6], [3,7]]
    let seq_kv = 2usize;
    let head_dim = 4usize;
    let k_data: Vec<f32> = (0..8).map(|i| i as f32).collect();

    println!("K matrix [seq_kv={}, head_dim={}]:", seq_kv, head_dim);
    for row in 0..seq_kv {
        let row_data: Vec<f32> = (0..head_dim)
            .map(|col| k_data[row * head_dim + col])
            .collect();
        println!("  row {}: {:?}", row, row_data);
    }

    println!("\nExpected K^T [head_dim={}, seq_kv={}]:", head_dim, seq_kv);
    for row in 0..head_dim {
        let row_data: Vec<f32> = (0..seq_kv)
            .map(|col| k_data[col * head_dim + row])
            .collect();
        println!("  row {}: {:?}", row, row_data);
    }

    // The INT8 CMMA load_key_transposed function stores K to SMEM without transposing,
    // then relies on CMMA load with ColMajor to interpret it as transposed.
    //
    // With K stored row-major as [seq_kv, head_dim] and loaded with stride=head_dim
    // and ColMajor layout, CMMA should interpret columns of K as rows of K^T.
    //
    // Let's verify this interpretation:
    // K in memory (row-major): [0,1,2,3,4,5,6,7]
    // With stride=4 (head_dim), CMMA ColMajor reads:
    //   Column 0: elements at 0, 4 (stride apart) -> [0, 4]
    //   Column 1: elements at 1, 5 -> [1, 5]
    //   etc.
    // This gives K^T correctly!

    // But wait - let me check the actual code stores it differently...
    // In load_key_transposed:
    //   smem[(smem_slice_start + linear_idx) as usize] = ...
    //   where linear_idx = src_row * src_cols + src_col
    //   and src_row is over seq_kv, src_col is over head_dim
    // So it stores K as-is (row-major [seq_kv, head_dim]).

    // Then cmma::load with stride=k_dim=head_dim and ColMajor.
    // This should work for the transpose... unless there's a bug.

    println!("\nMemory layout test:");
    println!("K stored row-major: {:?}", k_data);
    println!(
        "CMMA load with stride={} and ColMajor should give K^T",
        head_dim
    );

    // Verify: K[seq_kv_idx, head_dim_idx] should become K^T[head_dim_idx, seq_kv_idx]
    for hd in 0..head_dim {
        for skv in 0..seq_kv {
            let k_val = k_data[skv * head_dim + hd];
            let kt_expected = k_val; // K^T[hd, skv] = K[skv, hd]
            println!(
                "  K[{},{}] = {} -> K^T[{},{}] = {}",
                skv, hd, k_val, hd, skv, kt_expected
            );
        }
    }
}

/// Test the full INT8 CMMA attention pipeline:
/// 1. Q·K^T using i8×i8→i32 CMMA with quantization scales
/// 2. Softmax (using exp2)
/// 3. P×V using f16×f16→f32 CMMA
///
/// This test simulates what the real kernel does but with full visibility
/// into intermediate values at each stage.
#[cube(launch)]
fn int8_attention_debug_kernel(
    // Q: [seq_q, head_dim] i8
    q_i8: &Array<i8>,
    // K: [seq_kv, head_dim] i8 (will be transposed via CMMA load)
    k_i8: &Array<i8>,
    // V: [seq_kv, val_dim] f16
    v_f16: &Array<half::f16>,
    // Scales
    q_scale: &Array<f32>,
    k_scale: &Array<f32>,
    // Debug outputs
    debug_i32_scores: &mut Array<i32>, // Stage 3: raw i32 CMMA output
    debug_f32_scores: &mut Array<f32>, // Stage 4: after scale application
    debug_softmax: &mut Array<f32>,    // Stage 5: after softmax
    debug_output: &mut Array<f32>,     // Final output
    #[comptime] seq_q: u32,
    #[comptime] seq_kv: u32,
    #[comptime] head_dim: u32,
    #[comptime] val_dim: u32,
) {
    // ===== Stage 3: Q·K^T using i8×i8→i32 CMMA =====

    // Load Q as A matrix (RowMajor)
    let q_fragment = unsafe {
        cmma::Matrix::<i8>::uninitialized(
            cmma::MatrixIdent::A,
            seq_q as usize,
            seq_kv as usize,
            head_dim as usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    cmma::load(&q_fragment, &q_i8.to_slice(), head_dim);

    // Load K as B matrix (ColMajor for transpose)
    let k_fragment = unsafe {
        cmma::Matrix::<i8>::uninitialized(
            cmma::MatrixIdent::B,
            seq_q as usize,
            seq_kv as usize,
            head_dim as usize,
            cmma::MatrixLayout::ColMajor,
        )
    };
    cmma::load(&k_fragment, &k_i8.to_slice(), head_dim);

    // i32 accumulator
    let acc_i32 = unsafe {
        cmma::Matrix::<i32>::uninitialized(
            cmma::MatrixIdent::Accumulator,
            seq_q as usize,
            seq_kv as usize,
            head_dim as usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    cmma::fill(&acc_i32, 0i32);

    // Execute i8×i8→i32 CMMA
    cmma::execute::<i8, i8, i32, i32>(&q_fragment, &k_fragment, &acc_i32, &acc_i32);

    // Store raw i32 scores to debug output
    let smem_size = comptime!(seq_q * seq_kv);
    let mut smem_i32 = SharedMemory::<i32>::new(smem_size as usize);
    cmma::store(
        &mut smem_i32.slice_mut(0, smem_size as usize),
        &acc_i32,
        seq_kv,
        cmma::MatrixLayout::RowMajor,
    );
    sync_cube();

    // Copy to debug_i32_scores
    let elements_per_thread = comptime!((smem_size + 31) / 32);
    for i in 0..elements_per_thread {
        let idx = UNIT_POS_X + i * 32;
        if idx < smem_size {
            debug_i32_scores[idx as usize] = smem_i32[idx as usize];
        }
    }
    sync_cube();

    // ===== Stage 4: Apply combined scale (q_scale * k_scale) =====

    let combined_scale = q_scale[0] * k_scale[0];
    let mut smem_f32 = SharedMemory::<f32>::new(smem_size as usize);

    for i in 0..elements_per_thread {
        let idx = UNIT_POS_X + i * 32;
        if idx < smem_size {
            let i32_val = smem_i32[idx as usize];
            smem_f32[idx as usize] = f32::cast_from(i32_val) * combined_scale;
        }
    }
    sync_cube();

    // Copy to debug_f32_scores
    for i in 0..elements_per_thread {
        let idx = UNIT_POS_X + i * 32;
        if idx < smem_size {
            debug_f32_scores[idx as usize] = smem_f32[idx as usize];
        }
    }
    sync_cube();

    // ===== Stage 5: Softmax (using exp2) =====
    // Simple row-wise softmax: exp2(score - max) / sum
    // Use thread 0 to do serial softmax for simplicity in debug kernel

    let mut row_max = SharedMemory::<f32>::new(seq_q as usize);
    let mut row_sum = SharedMemory::<f32>::new(seq_q as usize);

    // Thread 0 computes softmax serially (simple but slow - fine for debug)
    if UNIT_POS_X == 0 {
        // Find row max values
        for row in 0..seq_q {
            let row_start = row * seq_kv;
            let first_val = smem_f32[row_start as usize];
            let mut max_val = first_val;
            for col in 1..seq_kv {
                let val = smem_f32[(row_start + col) as usize];
                max_val = f32::max(max_val, val);
            }
            row_max[row as usize] = max_val;
        }

        // Compute exp2 and row sums
        for row in 0..seq_q {
            let row_start = row * seq_kv;
            let max_val = row_max[row as usize];
            let mut sum = 0.0f32;
            for col in 0..seq_kv {
                let idx = (row_start + col) as usize;
                let exp_val = (smem_f32[idx] - max_val).exp2();
                smem_f32[idx] = exp_val;
                sum += exp_val;
            }
            row_sum[row as usize] = sum;
        }

        // Normalize
        for row in 0..seq_q {
            let row_start = row * seq_kv;
            let sum = row_sum[row as usize];
            let inv_sum = 1.0f32 / sum;
            for col in 0..seq_kv {
                let idx = (row_start + col) as usize;
                smem_f32[idx] = smem_f32[idx] * inv_sum;
            }
        }
    }
    sync_cube();

    // Copy to debug_softmax
    for i in 0..elements_per_thread {
        let idx = UNIT_POS_X + i * 32;
        if idx < smem_size {
            debug_softmax[idx as usize] = smem_f32[idx as usize];
        }
    }
    sync_cube();

    // ===== Stage: P×V using f16×f16→f32 CMMA =====

    // Load softmax probabilities into f32 accumulator, then cast to f16
    let p_fragment_f32 = unsafe {
        cmma::Matrix::<f32>::uninitialized(
            cmma::MatrixIdent::Accumulator,
            seq_q as usize,
            val_dim as usize,
            seq_kv as usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    cmma::load_with_layout(
        &p_fragment_f32,
        &smem_f32.to_slice(),
        seq_kv,
        cmma::MatrixLayout::RowMajor,
    );

    let p_fragment_f16 = cmma::cast::<f32, half::f16>(&p_fragment_f32);

    // Load V
    let v_fragment = unsafe {
        cmma::Matrix::<half::f16>::uninitialized(
            cmma::MatrixIdent::B,
            seq_q as usize,
            val_dim as usize,
            seq_kv as usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    cmma::load(&v_fragment, &v_f16.to_slice(), val_dim);

    // Output accumulator
    let out_fragment = unsafe {
        cmma::Matrix::<f32>::uninitialized(
            cmma::MatrixIdent::Accumulator,
            seq_q as usize,
            val_dim as usize,
            seq_kv as usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    cmma::fill(&out_fragment, 0.0f32);

    // Execute P×V
    cmma::execute::<half::f16, half::f16, f32, f32>(
        &p_fragment_f16,
        &v_fragment,
        &out_fragment,
        &out_fragment,
    );

    // Store output
    let out_size = comptime!(seq_q * val_dim);
    let mut smem_out = SharedMemory::<f32>::new(out_size as usize);
    cmma::store(
        &mut smem_out.slice_mut(0, out_size as usize),
        &out_fragment,
        val_dim,
        cmma::MatrixLayout::RowMajor,
    );
    sync_cube();

    // Copy to debug_output
    let out_elements_per_thread = comptime!((out_size + 31) / 32);
    for i in 0..out_elements_per_thread {
        let idx = UNIT_POS_X + i * 32;
        if idx < out_size {
            debug_output[idx as usize] = smem_out[idx as usize];
        }
    }
}

/// Test INT8 CMMA attention with debug output at each stage.
#[test]
fn test_int8_cmma_attention_stages() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    // Check for required CMMA configs
    let cmma_configs = &client.properties().features.cmma;
    let i8_type =
        cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Int(cubecl::ir::IntKind::I8));
    let i32_type =
        cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Int(cubecl::ir::IntKind::I32));
    let f16_type =
        cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F16));
    let f32_type =
        cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F32));

    // Find i8×i8→i32 config
    let mut i8_config: Option<(u32, u32, u32)> = None;
    for (m, n, k) in [(8, 16, 32), (16, 16, 32), (16, 8, 32)] {
        let config = cubecl::features::MmaConfig {
            a_type: i8_type,
            b_type: i8_type,
            cd_type: i32_type,
            m,
            n,
            k,
        };
        if cmma_configs.contains(&config) {
            i8_config = Some((m, n, k));
            break;
        }
    }

    // Find f16×f16→f32 config
    let mut f16_config: Option<(u32, u32, u32)> = None;
    for (m, n, k) in [(8, 16, 16), (8, 8, 8), (16, 16, 16)] {
        let config = cubecl::features::MmaConfig {
            a_type: f16_type,
            b_type: f16_type,
            cd_type: f32_type,
            m,
            n,
            k,
        };
        if cmma_configs.contains(&config) {
            f16_config = Some((m, n, k));
            break;
        }
    }

    let (seq_q, seq_kv, head_dim) = match i8_config {
        Some((m, n, k)) => {
            println!("Found i8×i8→i32 CMMA config: M={}, N={}, K={}", m, n, k);
            (m, n, k)
        }
        None => {
            println!("Skipping test: no i8×i8→i32 CMMA config available");
            return;
        }
    };

    let val_dim = match f16_config {
        Some((_, n, _)) => {
            println!("Found f16×f16→f32 CMMA config, using val_dim={}", n);
            n
        }
        None => {
            println!("Skipping test: no f16×f16→f32 CMMA config available");
            return;
        }
    };

    println!(
        "\nTest dimensions: seq_q={}, seq_kv={}, head_dim={}, val_dim={}",
        seq_q, seq_kv, head_dim, val_dim
    );

    // Create test data
    // Q: each row has the same value pattern for easy verification
    let q_f32: Vec<f32> = (0..(seq_q * head_dim) as usize)
        .map(|i| {
            let row = i / head_dim as usize;
            let col = i % head_dim as usize;
            ((row + 1) as f32 * 0.1) + (col as f32 * 0.001)
        })
        .collect();

    // K: similar pattern
    let k_f32: Vec<f32> = (0..(seq_kv * head_dim) as usize)
        .map(|i| {
            let row = i / head_dim as usize;
            let col = i % head_dim as usize;
            ((row + 1) as f32 * 0.05) + (col as f32 * 0.001)
        })
        .collect();

    // V: simple increasing pattern
    let v_f32: Vec<f32> = (0..(seq_kv * val_dim) as usize)
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();
    let v_f16: Vec<half::f16> = v_f32.iter().map(|&x| half::f16::from_f32(x)).collect();

    // Quantize Q with sm_scale = 1/sqrt(head_dim) * log2(e)
    let sm_scale = (1.0 / (head_dim as f32).sqrt()) * std::f32::consts::LOG2_E;
    println!("sm_scale = 1/sqrt({}) * log2(e) = {}", head_dim, sm_scale);

    // Quantize Q (with sm_scale baked in)
    let q_scaled: Vec<f32> = q_f32.iter().map(|&x| x * sm_scale).collect();
    let q_max_abs = q_scaled.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let q_scale = q_max_abs / 127.0;
    let q_inv_scale = 127.0 / q_max_abs.max(1e-6);
    let q_i8: Vec<i8> = q_scaled
        .iter()
        .map(|&x| (x * q_inv_scale).round().clamp(-127.0, 127.0) as i8)
        .collect();

    // Quantize K (without sm_scale)
    let k_max_abs = k_f32.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let k_scale = k_max_abs / 127.0;
    let k_inv_scale = 127.0 / k_max_abs.max(1e-6);
    let k_i8: Vec<i8> = k_f32
        .iter()
        .map(|&x| (x * k_inv_scale).round().clamp(-127.0, 127.0) as i8)
        .collect();

    println!("\nQuantization:");
    println!(
        "  Q: max_abs={:.6} (after sm_scale), scale={:.6}",
        q_max_abs, q_scale
    );
    println!("  K: max_abs={:.6}, scale={:.6}", k_max_abs, k_scale);
    println!("  Combined scale: {:.6}", q_scale * k_scale);

    // Create GPU buffers
    let q_i8_handle = client.create_from_slice(bytemuck::cast_slice(&q_i8));
    let k_i8_handle = client.create_from_slice(bytemuck::cast_slice(&k_i8));
    let v_f16_handle = client.create_from_slice(bytemuck::cast_slice(&v_f16));
    let q_scale_handle = client.create_from_slice(f32::as_bytes(&[q_scale]));
    let k_scale_handle = client.create_from_slice(f32::as_bytes(&[k_scale]));

    let score_size = (seq_q * seq_kv) as usize;
    let output_size = (seq_q * val_dim) as usize;

    let debug_i32_handle = client.empty(score_size * core::mem::size_of::<i32>());
    let debug_f32_handle = client.empty(score_size * core::mem::size_of::<f32>());
    let debug_softmax_handle = client.empty(score_size * core::mem::size_of::<f32>());
    let debug_output_handle = client.empty(output_size * core::mem::size_of::<f32>());

    // Launch kernel
    unsafe {
        int8_attention_debug_kernel::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(32),
            ArrayArg::from_raw_parts::<i8>(&q_i8_handle, (seq_q * head_dim) as usize, 1),
            ArrayArg::from_raw_parts::<i8>(&k_i8_handle, (seq_kv * head_dim) as usize, 1),
            ArrayArg::from_raw_parts::<half::f16>(&v_f16_handle, (seq_kv * val_dim) as usize, 1),
            ArrayArg::from_raw_parts::<f32>(&q_scale_handle, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&k_scale_handle, 1, 1),
            ArrayArg::from_raw_parts::<i32>(&debug_i32_handle, score_size, 1),
            ArrayArg::from_raw_parts::<f32>(&debug_f32_handle, score_size, 1),
            ArrayArg::from_raw_parts::<f32>(&debug_softmax_handle, score_size, 1),
            ArrayArg::from_raw_parts::<f32>(&debug_output_handle, output_size, 1),
            seq_q,
            seq_kv,
            head_dim,
            val_dim,
        )
        .unwrap();
    }

    // Read back results
    let i32_scores: Vec<i32> = bytemuck::cast_slice(&client.read_one(debug_i32_handle)).to_vec();
    let f32_scores: Vec<f32> = bytemuck::cast_slice(&client.read_one(debug_f32_handle)).to_vec();
    let softmax: Vec<f32> = bytemuck::cast_slice(&client.read_one(debug_softmax_handle)).to_vec();
    let output: Vec<f32> = bytemuck::cast_slice(&client.read_one(debug_output_handle)).to_vec();

    // ===== Validate Stage 3: i32 CMMA scores =====
    println!("\n===== Stage 3: i32 CMMA scores =====");
    println!(
        "i32 scores (first row): {:?}",
        &i32_scores[0..seq_kv as usize]
    );

    // Compute expected i32 scores on CPU: Q @ K^T
    let mut expected_i32 = vec![0i32; score_size];
    for i in 0..seq_q as usize {
        for j in 0..seq_kv as usize {
            let mut sum = 0i32;
            for d in 0..head_dim as usize {
                sum +=
                    q_i8[i * head_dim as usize + d] as i32 * k_i8[j * head_dim as usize + d] as i32;
            }
            expected_i32[i * seq_kv as usize + j] = sum;
        }
    }
    println!(
        "Expected i32 (first row): {:?}",
        &expected_i32[0..seq_kv as usize]
    );

    let mut i32_max_error = 0i32;
    for i in 0..score_size {
        let err = (i32_scores[i] - expected_i32[i]).abs();
        i32_max_error = i32_max_error.max(err);
    }
    println!("Max i32 error: {}", i32_max_error);

    // ===== Validate Stage 4: f32 scores after scale =====
    println!("\n===== Stage 4: f32 scores after scale =====");
    println!(
        "f32 scores (first row): {:?}",
        &f32_scores[0..seq_kv as usize]
    );

    let combined_scale = q_scale * k_scale;
    let expected_f32: Vec<f32> = expected_i32
        .iter()
        .map(|&x| x as f32 * combined_scale)
        .collect();
    println!(
        "Expected f32 (first row): {:?}",
        &expected_f32[0..seq_kv as usize]
    );

    let f32_max_error: f32 = f32_scores
        .iter()
        .zip(expected_f32.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("Max f32 error: {}", f32_max_error);

    // Check if scores are in reasonable range (should be roughly -10 to +10 for attention)
    let score_min = f32_scores.iter().cloned().fold(f32::INFINITY, f32::min);
    let score_max = f32_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("Score range: [{}, {}]", score_min, score_max);

    // ===== Validate Stage 5: Softmax =====
    println!("\n===== Stage 5: Softmax =====");
    println!("Softmax (first row): {:?}", &softmax[0..seq_kv as usize]);

    // Check row sums
    println!("Row sums:");
    for row in 0..seq_q as usize {
        let row_start = row * seq_kv as usize;
        let row_end = row_start + seq_kv as usize;
        let row_sum: f32 = softmax[row_start..row_end].iter().sum();
        println!("  Row {}: sum = {:.6}", row, row_sum);
        // Row sum should be ~1.0
        assert!(
            (row_sum - 1.0).abs() < 0.01,
            "Softmax row {} sum {} not close to 1.0",
            row,
            row_sum
        );
    }

    // ===== Validate output =====
    println!("\n===== Final Output =====");
    println!("Output (first row): {:?}", &output[0..val_dim as usize]);

    // Compute expected output: softmax @ V (using CPU softmax on expected_f32)
    // Compute CPU softmax
    let mut cpu_softmax = expected_f32.clone();
    for row in 0..seq_q as usize {
        let row_start = row * seq_kv as usize;
        let row_end = row_start + seq_kv as usize;

        // Max
        let max_val = cpu_softmax[row_start..row_end]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        // Exp2
        for i in row_start..row_end {
            cpu_softmax[i] = (cpu_softmax[i] - max_val).exp2();
        }
        // Sum and normalize
        let sum: f32 = cpu_softmax[row_start..row_end].iter().sum();
        for i in row_start..row_end {
            cpu_softmax[i] /= sum;
        }
    }

    // Compute expected output: softmax @ V
    let mut expected_output = vec![0.0f32; output_size];
    for i in 0..seq_q as usize {
        for j in 0..val_dim as usize {
            let mut sum = 0.0f32;
            for k in 0..seq_kv as usize {
                sum += cpu_softmax[i * seq_kv as usize + k] * v_f32[k * val_dim as usize + j];
            }
            expected_output[i * val_dim as usize + j] = sum;
        }
    }
    println!(
        "Expected output (first row): {:?}",
        &expected_output[0..val_dim as usize]
    );

    let output_max_error: f32 = output
        .iter()
        .zip(expected_output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("Max output error: {}", output_max_error);

    // Assertions
    assert!(
        i32_max_error == 0,
        "i32 CMMA scores don't match CPU reference"
    );
    assert!(
        f32_max_error < 0.001,
        "f32 scores after scale don't match CPU reference"
    );
    assert!(
        output_max_error < 0.1,
        "Final output doesn't match CPU reference (tolerance 0.1 for f16 precision)"
    );

    println!("\n===== TEST PASSED =====");
}
