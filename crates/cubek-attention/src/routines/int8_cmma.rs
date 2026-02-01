//! INT8 CMMA routine for hardware-accelerated attention with tensor cores.
//!
//! Uses i8x8->i32 CMMA for Q·K^T and f16xf16->f16 CMMA for P×V.

use cubecl::CubeDim;
use cubek_matmul::components::{global::PartitionedStageFamily, stage::StridedStageFamily};

use crate::components::batch::simple::SimpleBatchAttentionFamily;
use crate::components::global::simple::SimpleGlobalAttentionFamily;
use crate::components::stage::plane::PlanePartitionStageAttentionFamily;
use crate::components::tile::int8_cmma::Int8CmmaTileAttention;
use crate::definition::{
    AttentionBlueprint, AttentionElems, AttentionPartitionSize, AttentionProblem,
    AttentionSetupError, AttentionStageSize, AttentionTileSize, AttentionTilingScheme,
    HypercubeBlueprint,
};
use crate::launch::BlueprintStrategy;
use crate::routines::{DeviceSettings, LaunchInfo, Routine};

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

    fn prepare(
        problem: &AttentionProblem,
        device_settings: &DeviceSettings,
        strategy: BlueprintStrategy<Self>,
    ) -> Result<LaunchInfo<Self::Blueprint>, AttentionSetupError> {
        let blueprint = blueprint(problem, device_settings, strategy)?;

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

fn blueprint(
    problem: &AttentionProblem,
    launch_settings: &DeviceSettings,
    strategy: BlueprintStrategy<Int8CmmaRoutine>,
) -> Result<AttentionBlueprint, AttentionSetupError> {
    match strategy {
        BlueprintStrategy::Forced(attention_blueprint) => validate(problem, attention_blueprint),
        BlueprintStrategy::Inferred(_) => {
            // INT8 CMMA tile sizes optimized for tensor core operations
            // For i8x8->i32 CMMA: m=16, n=16, k=16 or m=8, n=32, k=16
            // We use seq_q=16, head_dim=32, seq_kv=8 to match common attention patterns
            let tile_size = AttentionTileSize {
                seq_q: 16,    // m for Q·K^T
                head_dim: 32, // k for Q·K^T (must be divisible by 16 for CMMA)
                seq_kv: 8,    // n for Q·K^T
                val_dim: 8,   // n for P×V
            };

            let partition_head_dim = problem.dims.head_dim as u32 / tile_size.head_dim;
            let partition_val_dim = problem.dims.val_dim as u32 / tile_size.val_dim;

            // Match reference SageAttention block sizes:
            // - BLOCK_M = 128 query rows per workgroup (8 tiles of 16 rows each)
            // - BLOCK_N = 64 KV elements per inner loop iteration (8 tiles of 8 KV each)
            let stage_seq_q = 8;
            let partition_seq_kv = 8;

            let tiling_scheme = AttentionTilingScheme {
                tile_size,
                partition_size: AttentionPartitionSize {
                    seq_q: 1,
                    head_dim: partition_head_dim,
                    seq_kv: partition_seq_kv,
                    val_dim: partition_val_dim,
                },
                stage_size: AttentionStageSize { seq_q: stage_seq_q },
            };

            let blueprint = AttentionBlueprint {
                hypercube_blueprint: HypercubeBlueprint {},
                plane_dim: launch_settings.plane_dim,
                two_rows_in_array_tile: false,
                line_sizes: launch_settings.line_sizes.clone(),
                masked: problem.masked,
                causal: problem.options.causal,
                tiling_scheme,
                check_bounds: tiling_scheme.check_bounds(&problem.dims),
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
