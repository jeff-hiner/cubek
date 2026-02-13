//! INT8 CMMA routine for hardware-accelerated attention with tensor cores.
//!
//! Uses i8×i8→i32 CMMA for Q·K^T and f16×f16→f32 CMMA for P×V.

use crate::{
    components::{
        batch::simple::SimpleBatchAttentionFamily, global::simple::SimpleGlobalAttentionFamily,
        stage::plane::PlanePartitionStageAttentionFamily, tile::int8_cmma::Int8CmmaTileAttention,
    },
    definition::{
        AttentionAvailabilityError, AttentionBlueprint, AttentionElems, AttentionPartitionSize,
        AttentionProblem, AttentionSetupError, AttentionStageSize, AttentionTileSize,
        AttentionTilingScheme, HypercubeBlueprint,
    },
    launch::BlueprintStrategy,
    routines::{DeviceSettings, LaunchInfo, Routine},
};
use cubecl::{
    CubeDim, Runtime,
    client::ComputeClient,
    features::MmaConfig,
    ir::{ElemType, IntKind, StorageType},
};
use cubek_matmul::components::{global::PartitionedStageFamily, stage::StridedStageFamily};

/// INT8 CMMA routine using tensor cores for Q·K^T computation.
#[derive(Debug, Clone)]
pub struct Int8CmmaRoutine {}

impl Routine for Int8CmmaRoutine {
    type TileAttention = Int8CmmaTileAttention;
    type StageAttention = PlanePartitionStageAttentionFamily<
        Self::TileAttention,
        StridedStageFamily,
        StridedStageFamily,
        PartitionedStageFamily,
    >;
    type GlobalAttention = SimpleGlobalAttentionFamily<Self::StageAttention>;
    type BatchAttention = SimpleBatchAttentionFamily<Self::GlobalAttention>;

    type Strategy = ();
    type Blueprint = AttentionBlueprint;

    fn prepare<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &AttentionProblem,
        device_settings: &DeviceSettings,
        strategy: BlueprintStrategy<Self>,
    ) -> Result<LaunchInfo<Self::Blueprint>, AttentionSetupError> {
        let blueprint = blueprint(client, problem, device_settings, strategy)?;

        // INT8 CMMA always uses specific element types
        let dtypes = AttentionElems::for_int8_cmma(&problem.global_dtypes);

        let num_planes = blueprint.tiling_scheme.stage_size.seq_q;
        let cube_dim = CubeDim::new_2d(blueprint.plane_dim, num_planes);

        let cube_count_plan = blueprint
            .hypercube_blueprint
            .cube_count_plan(&problem.dims, &blueprint);

        Ok(LaunchInfo {
            blueprint,
            dtypes,
            cube_dim,
            cube_count_plan,
        })
    }
}

/// Find a supported INT8 CMMA tile size for the hardware.
///
/// INT8 CMMA computes i8×i8→i32 for Q·K^T. Different hardware has different
/// supported configurations:
/// - Intel Arc/140V: m=8, n=16, k=32
/// - NVIDIA Ampere+: typically m=16, n=8, k=32 or m=16, n=16, k=32
///
/// Returns AttentionTileSize with:
/// - seq_q = m (query rows per tile)
/// - head_dim = k (inner dimension)
/// - seq_kv = n (KV elements per tile)
/// - val_dim = n (same as seq_kv for simplicity)
fn find_int8_cmma_tile_size<R: Runtime>(client: &ComputeClient<R>) -> Option<AttentionTileSize> {
    let cmma_configs = &client.properties().features.cmma;

    let i8_type = StorageType::Scalar(ElemType::Int(IntKind::I8));
    let i32_type = StorageType::Scalar(ElemType::Int(IntKind::I32));

    // Try sizes in preference order. Intel uses (8, 16, 32), NVIDIA may use (16, 8, 32).
    for (m, n, k) in [(16, 16, 32), (8, 16, 32), (16, 8, 32)] {
        let config = MmaConfig {
            a_type: i8_type,
            b_type: i8_type,
            cd_type: i32_type,
            m,
            n,
            k,
        };
        if cmma_configs.contains(&config) {
            return Some(AttentionTileSize {
                seq_q: m,    // M dimension (query rows)
                head_dim: k, // K dimension (inner/head dim)
                seq_kv: n,   // N dimension (KV elements)
                val_dim: n,  // Use same N for value matmul
            });
        }
    }
    None
}

fn blueprint<R: Runtime>(
    client: &ComputeClient<R>,
    problem: &AttentionProblem,
    launch_settings: &DeviceSettings,
    strategy: BlueprintStrategy<Int8CmmaRoutine>,
) -> Result<AttentionBlueprint, AttentionSetupError> {
    match strategy {
        BlueprintStrategy::Forced(attention_blueprint) => validate(problem, attention_blueprint),
        BlueprintStrategy::Inferred(_) => {
            // Query hardware for supported INT8 CMMA tile sizes
            let i8_type = StorageType::Scalar(ElemType::Int(IntKind::I8));
            let i32_type = StorageType::Scalar(ElemType::Int(IntKind::I32));

            let tile_size = find_int8_cmma_tile_size(client).ok_or(
                AttentionAvailabilityError::CmmaInstructionUnavailable {
                    a_type: i8_type,
                    b_type: i8_type,
                    cd_type: i32_type,
                },
            )?;

            let partition_head_dim = problem.dims.head_dim as u32 / tile_size.head_dim;
            let partition_val_dim = problem.dims.val_dim as u32 / tile_size.val_dim;

            // Use simple single-plane approach like f16 CMMA (BlackboxAccelerated)
            // This simplifies coordination and matches the working implementation
            let tiling_scheme = AttentionTilingScheme {
                tile_size,
                partition_size: AttentionPartitionSize {
                    seq_q: 1,
                    head_dim: partition_head_dim,
                    seq_kv: 1,
                    val_dim: partition_val_dim,
                },
                stage_size: AttentionStageSize { seq_q: 1 },
            };

            // For INT8 CMMA, sm_scale (1/sqrt(head_dim)) is baked into Q quantization
            // following SageAttention. Setting original_head_dim=1 makes the softmax
            // scale = 1/sqrt(1) = 1.0, avoiding double-scaling.
            // The actual original_head_dim is still used in the quantization step.
            let original_head_dim = 1;

            let blueprint = AttentionBlueprint {
                hypercube_blueprint: HypercubeBlueprint {},
                plane_dim: launch_settings.plane_dim,
                two_rows_in_array_tile: false,
                line_sizes: launch_settings.line_sizes.clone(),
                masked: problem.masked,
                causal: problem.options.causal,
                tiling_scheme,
                check_bounds: tiling_scheme.check_bounds(&problem.dims),
                original_head_dim,
            };

            validate(problem, blueprint)
        }
    }
}

fn validate(
    problem: &AttentionProblem,
    blueprint: AttentionBlueprint,
) -> Result<AttentionBlueprint, AttentionSetupError> {
    // INT8 CMMA requires head_dim divisible by 16
    if problem.dims.head_dim as u32 % 16 != 0 {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "INT8 CMMA requires head_dim divisible by 16",
        )));
    }

    if problem.dims.head_dim as u32 % blueprint.tiling_scheme.tile_size.head_dim != 0 {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "Tile size head dim must divide problem head dim",
        )));
    }

    if blueprint.tiling_scheme.partition_size.head_dim * blueprint.tiling_scheme.tile_size.head_dim
        != problem.dims.head_dim as u32
    {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "Tiling scheme's total head dim must equal problem's head dim",
        )));
    }

    Ok(blueprint)
}
