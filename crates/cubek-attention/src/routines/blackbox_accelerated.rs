use cubecl::client::ComputeClient;
use cubecl::features::MmaConfig;
use cubecl::ir::StorageType;
use cubecl::{CubeDim, Runtime};
use cubek_matmul::components::{global::PartitionedStageFamily, stage::StridedStageFamily};

use crate::components::stage::plane::PlanePartitionStageAttentionFamily;
use crate::components::tile::accelerated::BlackboxAcceleratedTileAttention;
use crate::definition::AttentionAvailabilityError;
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

    fn prepare<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &AttentionProblem,
        device_settings: &DeviceSettings,
        strategy: BlueprintStrategy<Self>,
    ) -> Result<LaunchInfo<Self::Blueprint>, AttentionSetupError> {
        let blueprint = blueprint(client, problem, device_settings, strategy)?;

        let dtypes = AttentionElems::from_global_types(
            &problem.global_dtypes,
            &problem.options.accumulator_precision,
        );

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

/// Find a supported CMMA tile size for the given element types.
///
/// Both matmuls in attention use the same types after the softmax fix:
/// - score_matmul: QT × KT → ACC (then softmax is cast to VT)
/// - value_matmul: VT × VT → ACC
///
/// So we need a CMMA config with a=value_tile, b=value_tile, cd=accumulator.
fn find_supported_tile_size<R: Runtime>(
    client: &ComputeClient<R>,
    value_tile_type: StorageType,
    accumulator_type: StorageType,
) -> Option<AttentionTileSize> {
    let cmma_configs = &client.properties().features.cmma;

    // Try sizes in preference order (larger = better throughput)
    // Common configurations: 16×16×16, 8×16×16 (Intel Arc), 8×8×8 (Metal)
    for (m, n, k) in [(16, 16, 16), (8, 16, 16), (8, 8, 8)] {
        let config = MmaConfig {
            a_type: value_tile_type,
            b_type: value_tile_type,
            cd_type: accumulator_type,
            m,
            n,
            k,
        };
        if cmma_configs.contains(&config) {
            return Some(AttentionTileSize {
                seq_q: m,     // M dimension
                head_dim: k,  // K dimension for score matmul
                seq_kv: n,    // N for score, K for value matmul
                val_dim: n,   // N dimension for value matmul
            });
        }
    }
    None
}

fn blueprint<R: Runtime>(
    client: &ComputeClient<R>,
    problem: &AttentionProblem,
    launch_settings: &DeviceSettings,
    strategy: BlueprintStrategy<BlackboxAcceleratedRoutine>,
) -> Result<AttentionBlueprint, AttentionSetupError> {
    match strategy {
        BlueprintStrategy::Forced(attention_blueprint) => validate(problem, attention_blueprint),
        BlueprintStrategy::Inferred(_) => {
            // Determine the element types used by CMMA
            let value_tile_type = problem.global_dtypes.value;
            let accumulator_type = match &problem.options.accumulator_precision {
                crate::definition::AccumulatorPrecision::Strict(ty) => *ty,
                crate::definition::AccumulatorPrecision::Loose => {
                    crate::definition::AccumulatorPrecision::default_accumulator_type()
                }
            };

            let tile_size = find_supported_tile_size(client, value_tile_type, accumulator_type)
                .ok_or_else(|| {
                    AttentionAvailabilityError::CmmaInstructionUnavailable {
                        a_type: value_tile_type,
                        b_type: value_tile_type,
                        cd_type: accumulator_type,
                    }
                })?;

            let partition_head_dim = problem.dims.head_dim as u32 / tile_size.head_dim;
            let partition_val_dim = partition_head_dim;

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

            // Use original_head_dim if provided (for padded tensors), otherwise use head_dim
            let original_head_dim = problem
                .dims
                .original_head_dim
                .unwrap_or(problem.dims.head_dim) as u32;

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
