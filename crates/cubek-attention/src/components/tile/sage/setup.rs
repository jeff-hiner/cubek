//! SageAttention tile configuration

use cubek_matmul::components::CubeDimResource;

use crate::components::tile::sage::SageTileAttention;
use crate::components::tile::{SharedTileAttentionConfig, TileAttentionConfig, TileAttentionFamily};
use crate::definition::{
    AttentionBlueprint, AttentionPrecision, AttentionSetupError, AttentionTileSize,
    InvalidConfigError,
};

/// Configuration for SageTileAttention (INT8 quantized Q·K^T).
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct SageTileAttentionConfig {
    /// Shared configuration inherited from base tile attention.
    pub shared: SharedTileAttentionConfig,
}

impl TileAttentionConfig for SageTileAttentionConfig {
    fn plane_dim(&self) -> u32 {
        self.shared.plane_dim
    }

    fn num_planes(&self) -> u32 {
        self.shared.num_planes
    }

    fn attention_tile_size(&self) -> AttentionTileSize {
        self.shared.attention_tile_size
    }

    fn num_rows_per_unit(&self) -> u32 {
        self.shared.attention_tile_size.seq_q
    }

    fn causal_mask(&self) -> bool {
        self.shared.causal_mask
    }

    fn materialized_mask(&self) -> bool {
        self.shared.materialized_mask
    }
}

impl TileAttentionFamily for SageTileAttention {
    type TileAttention<F: AttentionPrecision> = SageTileAttention;

    type Config = SageTileAttentionConfig;

    fn requires_accelerator() -> bool {
        false
    }

    fn computation_resources() -> Result<CubeDimResource, InvalidConfigError> {
        Ok(CubeDimResource::Units(1))
    }

    fn expand_config(blueprint: &AttentionBlueprint) -> Result<Self::Config, AttentionSetupError> {
        Ok(SageTileAttentionConfig {
            shared: SharedTileAttentionConfig {
                plane_dim: blueprint.plane_dim,
                attention_tile_size: blueprint.tiling_scheme.tile_size,
                num_planes: blueprint.tiling_scheme.stage_size.seq_q,
                causal_mask: blueprint.causal,
                materialized_mask: blueprint.masked,
            },
        })
    }
}
