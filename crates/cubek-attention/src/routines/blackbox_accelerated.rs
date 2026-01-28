use cubecl::CubeDim;
use cubek_matmul::components::{global::PartitionedStageFamily, stage::StridedStageFamily};

use crate::components::stage::plane::PlanePartitionStageAttentionFamily;
use crate::components::tile::accelerated::BlackboxAcceleratedTileAttention;
use crate::definition::AttentionTileSize;
use crate::definition::{
    AttentionBlueprint, AttentionElems, AttentionPartitionSize, AttentionProblem,
    AttentionSetupError, AttentionStageSize, AttentionTilingScheme, HypercubeBlueprint,
};
use crate::launch::BlueprintStrategy;
use crate::routines::{DeviceSettings, LaunchInfo};
use crate::{
    components::{
        batch::simple::SimpleBatchAttentionFamily, global::simple::SimpleGlobalAttentionFamily,
    },
    routines::Routine,
};

#[derive(Debug, Clone)]
pub struct BlackboxAcceleratedRoutine {}

impl Routine for BlackboxAcceleratedRoutine {
    type TileAttention = BlackboxAcceleratedTileAttention;
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

        // Use INT8 CMMA element types when enabled, otherwise standard float types
        let dtypes = if problem.options.int8_cmma {
            AttentionElems::for_int8_cmma(&problem.global_dtypes)
        } else {
            AttentionElems::from_global_types(
                &problem.global_dtypes,
                &problem.options.accumulator_precision,
            )
        };

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
    strategy: BlueprintStrategy<BlackboxAcceleratedRoutine>,
) -> Result<AttentionBlueprint, AttentionSetupError> {
    match strategy {
        BlueprintStrategy::Forced(attention_blueprint) => validate(problem, attention_blueprint),
        BlueprintStrategy::Inferred(_) => {
            // Select tile size based on platform and INT8 CMMA mode
            let tile_size = if problem.options.int8_cmma {
                // INT8 CMMA tile size: m=16, n=8, k=32
                // For Q·K^T: (seq_q, head_dim) × (head_dim, seq_kv) → (seq_q, seq_kv)
                // So: m=seq_q=16, k=head_dim=32, n=seq_kv=8
                // For softmax×V: (seq_q, seq_kv) × (seq_kv, val_dim) → (seq_q, val_dim)
                // This uses f32 CMMA, different tile sizes may apply
                AttentionTileSize {
                    seq_q: 16,      // m for Q·K^T
                    head_dim: 32,   // k for Q·K^T (shared dim)
                    seq_kv: 8,      // n for Q·K^T
                    val_dim: 8,     // n for softmax×V (can use same as seq_kv for f32 CMMA)
                }
            } else {
                #[cfg(target_os = "macos")]
                {
                    AttentionTileSize {
                        seq_q: 8,
                        head_dim: 8,
                        seq_kv: 8,
                        val_dim: 8,
                    }
                }
                #[cfg(not(target_os = "macos"))]
                {
                    AttentionTileSize {
                        seq_q: 16,
                        head_dim: 16,
                        seq_kv: 16,
                        val_dim: 16,
                    }
                }
            };

            let partition_head_dim = problem.dims.head_dim as u32 / tile_size.head_dim;
            let partition_val_dim = problem.dims.val_dim as u32 / tile_size.val_dim;

            // Match reference SageAttention block sizes:
            // - BLOCK_M = 128 query rows per workgroup
            // - BLOCK_N = 64 KV elements per inner loop iteration
            // - num_warps = 8 (256 threads per workgroup)
            let (stage_seq_q, partition_seq_kv) = if problem.options.int8_cmma {
                // INT8 CMMA: tile is 16×8, need 8 planes for 128 rows, 8 tiles for 64 KV
                (8, 8)
            } else {
                (1, 1)
            };

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
    if problem.dims.head_dim as u32 % blueprint.tiling_scheme.tile_size.head_dim != 0 {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "Tile size head dim must divide problem head dim".to_string(),
        )));
    }

    if blueprint.tiling_scheme.partition_size.head_dim * blueprint.tiling_scheme.tile_size.head_dim
        != problem.dims.head_dim as u32
    {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "Tiling scheme's total head dim must equal problem's head dim".to_string(),
        )));
    }

    Ok(blueprint)
}
