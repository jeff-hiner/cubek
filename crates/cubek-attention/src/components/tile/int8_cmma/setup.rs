//! INT8 CMMA tile attention configuration.

use cubecl::ir::LineSize;
use cubek_matmul::components::CubeDimResource;

use crate::components::tile::accelerated::local_tile::InnerLayout;
use crate::components::tile::int8_cmma::Int8CmmaTileAttention;
use crate::components::tile::{SharedTileAttentionConfig, TileAttentionConfig, TileAttentionFamily};
use crate::definition::{
    AttentionBlueprint, AttentionPrecision, AttentionSetupError, AttentionTileSize,
    InvalidConfigError,
};

/// Configuration for Int8CmmaTileAttention.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Int8CmmaAttentionConfig {
    /// Shared configuration inherited from base tile attention.
    pub shared: SharedTileAttentionConfig,
    /// Inner layout for LocalTile operations.
    pub inner_layout: InnerLayout,
}

impl TileAttentionConfig for Int8CmmaAttentionConfig {
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
        match self.inner_layout {
            InnerLayout::Contiguous => 1u32,
            InnerLayout::SplitRows => 2u32,
        }
    }

    fn causal_mask(&self) -> bool {
        self.shared.causal_mask
    }

    fn materialized_mask(&self) -> bool {
        self.shared.materialized_mask
    }
}

impl TileAttentionFamily for Int8CmmaTileAttention {
    type TileAttention<F: AttentionPrecision> = Int8CmmaTileAttention;

    type Config = Int8CmmaAttentionConfig;

    fn requires_accelerator() -> bool {
        true // Requires tensor cores for CMMA
    }

    fn computation_resources() -> Result<CubeDimResource, InvalidConfigError> {
        Ok(CubeDimResource::Planes(1))
    }

    fn expand_config(blueprint: &AttentionBlueprint) -> Result<Self::Config, AttentionSetupError> {
        validate(
            Int8CmmaAttentionConfig {
                shared: SharedTileAttentionConfig {
                    plane_dim: blueprint.plane_dim,
                    num_planes: blueprint.tiling_scheme.stage_size.seq_q,
                    attention_tile_size: blueprint.tiling_scheme.tile_size,
                    causal_mask: blueprint.causal,
                    materialized_mask: blueprint.masked,
                },
                inner_layout: if blueprint.two_rows_in_array_tile {
                    InnerLayout::SplitRows
                } else {
                    InnerLayout::Contiguous
                },
            },
            blueprint.line_sizes.mask,
        )
    }
}

fn validate(
    config: Int8CmmaAttentionConfig,
    line_sizes_mask: LineSize,
) -> Result<Int8CmmaAttentionConfig, AttentionSetupError> {
    if line_sizes_mask > 1 {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "Line size mask > 1 not supported yet on INT8 CMMA tile attention",
        )));
    }

    let softmax_num_rows = config.shared.attention_tile_size.seq_q;
    let softmax_num_cols = config.shared.attention_tile_size.seq_kv;
    let softmax_total = softmax_num_rows * softmax_num_cols;

    if softmax_total % config.shared.plane_dim != 0 {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "Softmax size should be divisible by plane dim",
        )));
    }

    if config.inner_layout == InnerLayout::Contiguous && softmax_num_rows > config.shared.plane_dim
    {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "More than one row per unit not supported with this inner layout",
        )));
    }

    if config.inner_layout == InnerLayout::SplitRows
        && softmax_total % (2 * config.shared.plane_dim) != 0
    {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "With split rows, units must have two elements each",
        )));
    }

    // INT8 CMMA has specific tile size requirements
    // NVIDIA wmma supports: m=16, n=16, k=16 for i8×i8→i32
    // Or m=8, n=32, k=16 and other variants
    let tile = &config.shared.attention_tile_size;
    if tile.head_dim % 16 != 0 {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "INT8 CMMA requires head_dim divisible by 16",
        )));
    }

    Ok(config)
}
