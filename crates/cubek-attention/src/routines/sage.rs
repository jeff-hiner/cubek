//! SageAttention routine - INT8 quantized attention for improved performance
//!
//! This module provides a SageAttention implementation that:
//! 1. Takes FP16 Q, K, V tensors
//! 2. Quantizes Q and K to INT8 with per-block scaling
//! 3. Computes attention scores using INT8 matmul with proper dequantization
//! 4. Applies softmax and value multiplication in FP32
//! 5. Returns FP16 output
//!
//! The key difference from BlackboxAccelerated is proper scale handling:
//! score_f32 = score_i32 * q_scale * k_scale * sm_scale

use cubecl::client::ComputeClient;
use cubecl::{CubeDim, Runtime};
use cubek_matmul::components::CubeDimResource;
use cubek_matmul::components::{global::PartitionedStageFamily, stage::StridedStageFamily};

use crate::components::stage::unit::UnitPartitionStageAttentionFamily;
use crate::components::tile::TileAttentionFamily;
use crate::components::tile::sage::SageTileAttention;
use crate::definition::{
    AttentionBlueprint, AttentionElems, AttentionPartitionSize, AttentionProblem,
    AttentionSetupError, AttentionStageSize, AttentionTileSize, AttentionTilingScheme,
    HypercubeBlueprint,
};
use crate::launch::BlueprintStrategy;
use crate::routines::{DeviceSettings, LaunchInfo};
use crate::{
    components::{
        batch::simple::SimpleBatchAttentionFamily, global::simple::SimpleGlobalAttentionFamily,
    },
    routines::Routine,
};

/// SageAttention routine marker type
#[derive(Debug, Clone)]
pub struct SageRoutine {}

impl Routine for SageRoutine {
    type TileAttention = SageTileAttention;
    type StageAttention = UnitPartitionStageAttentionFamily<
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
        _client: &ComputeClient<R>,
        problem: &AttentionProblem,
        device_settings: &DeviceSettings,
        strategy: BlueprintStrategy<Self>,
    ) -> Result<LaunchInfo<Self::Blueprint>, AttentionSetupError> {
        let blueprint = blueprint(problem, device_settings, strategy)?;

        // Use standard float element types for sage (INT8 quantization is internal)
        let dtypes = AttentionElems::from_global_types(
            &problem.global_dtypes,
            &problem.options.accumulator_precision,
        );

        let compute_resources = match Self::TileAttention::computation_resources()? {
            CubeDimResource::Units(units) => {
                CubeDimResource::Units(units * blueprint.tiling_scheme.stage_size.seq_q)
            }
            _ => {
                return Err(AttentionSetupError::InvalidConfig(Box::new(
                    "Error: Expected unit tile attention, got a plane tile attention".to_string(),
                )));
            }
        };

        let num_planes = compute_resources.num_planes(blueprint.plane_dim)?;
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
    strategy: BlueprintStrategy<SageRoutine>,
) -> Result<AttentionBlueprint, AttentionSetupError> {
    match strategy {
        BlueprintStrategy::Forced(attention_blueprint) => validate(problem, attention_blueprint),
        BlueprintStrategy::Inferred(_) => {
            // Use small tile sizes suitable for scalar INT8 computation.
            // These are similar to UnitRoutine but can be tuned for INT8 performance.
            let tile_size = AttentionTileSize {
                seq_q: 4,
                head_dim: 4,
                seq_kv: 4,
                val_dim: 4,
            };

            let partition_head_dim = problem.dims.head_dim as u32 / tile_size.head_dim;
            let partition_val_dim = partition_head_dim;

            let plane_dim = launch_settings.plane_dim;

            let tiling_scheme = AttentionTilingScheme {
                tile_size,
                partition_size: AttentionPartitionSize {
                    seq_q: 1,
                    head_dim: partition_head_dim,
                    seq_kv: 1,
                    val_dim: partition_val_dim,
                },
                stage_size: AttentionStageSize { seq_q: plane_dim },
            };

            let blueprint = AttentionBlueprint {
                hypercube_blueprint: HypercubeBlueprint {},
                tiling_scheme,
                plane_dim,
                two_rows_in_array_tile: false,
                line_sizes: launch_settings.line_sizes.clone(),
                masked: problem.masked,
                causal: problem.options.causal,
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
