use crate::{
    components::tile::{
        FragmentAccumulator, FragmentLayout, FragmentMask, FragmentSoftmax, RowwiseFormat,
    },
    definition::{
        AttentionBlueprint, AttentionPrecision, AttentionSetupError, AttentionTileSize,
        InvalidConfigError,
        attention_types::{ACC, KS, SM},
    },
};
use cubecl::{self, prelude::*};
use cubek_matmul::components::{CubeDimResource, tile::StridedTile};
use std::{fmt::Debug, hash::Hash};

/// Logits below this are considered masked (effectively -inf)
/// Value chosen to fit within f16 range (~-65,504 max)
pub(crate) const LOGIT_MASKED: f32 = -6e4;

/// Any value smaller than this is considered numerically zero
/// (used for fully-masked rows or tiny contributions)
/// Value chosen to be above f16 smallest normal (~6.1e-5)
pub(crate) const FULLY_MASKED_ROW_THRESHOLD: f32 = 1e-4;

#[cube]
pub trait TileAttention<AP: AttentionPrecision>: Send + Sync + 'static {
    type Config: TileAttentionConfig;
    type Query: CubeType;
    /// Key fragment type for Q·K^T CMMA. For float: f16. For INT8: i8.
    type Key: CubeType;
    /// Value fragment type for P×V CMMA. For float: f16. For INT8: f16.
    type Value: CubeType;
    type Mask: FragmentMask<Layout = Self::FragmentLayout>;

    type Softmax: FragmentSoftmax<SM<AP>, Layout = Self::FragmentLayout, SoftmaxRowFormat = Self::SoftmaxRow>;
    type SoftmaxRow: RowwiseFormat<SM<AP>, Layout = Self::FragmentLayout>;

    type Accumulator: FragmentAccumulator<ACC<AP>>;
    type FragmentLayout: FragmentLayout;

    fn softmax_layout(#[comptime] config: Self::Config) -> Self::FragmentLayout;

    /// Compute Q·K^T score matrix.
    ///
    /// For float attention: directly outputs to Softmax type using CMMA.
    /// For INT8 CMMA: CMMA outputs i32, which is converted to f32 internally.
    ///
    /// The `key_tile` parameter provides raw access to the key data from stage SMEM,
    /// enabling scalar computation paths that bypass CMMA for reduced sync overhead.
    fn score_matmul(
        lhs: &Self::Query,
        rhs: &Self::Key,
        key_tile: &StridedTile<KS<AP>>,
        out: &mut Self::Softmax,
        #[comptime] config: Self::Config,
    );

    /// Compute Q·K^T using scalar operations with direct LocalTile output.
    ///
    /// For INT8 path: reads Q from scalar storage, K from StridedTile,
    /// writes directly to LocalTile, avoiding CMMA→SMEM→LocalTile conversion.
    /// This reduces sync count from 2 to 1 per KV tile.
    ///
    /// Default implementation calls score_matmul (CMMA path).
    fn score_matmul_scalar<QE: Numeric>(
        query_scalar: &Slice<QE>,
        key_tile: &StridedTile<KS<AP>>,
        out: &mut Self::Softmax,
        #[comptime] config: Self::Config,
    );

    /// Compute P×V (softmax × Value) accumulation.
    ///
    /// For float attention: f16 × f16 → f32.
    /// For INT8 CMMA: f16 × f16 → f32 (V stays f16, not quantized).
    fn value_matmul(
        lhs: &Self::Softmax,
        rhs: &Self::Value,
        out: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    );

    fn allocate_query(#[comptime] config: Self::Config) -> Self::Query;
    fn allocate_mask(#[comptime] config: Self::Config) -> Self::Mask;

    fn allocate_key(#[comptime] config: Self::Config) -> Self::Key;
    fn allocate_value(#[comptime] config: Self::Config) -> Self::Value;

    fn allocate_softmax(#[comptime] config: Self::Config) -> Self::Softmax;
    fn allocate_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator;

    fn load_query<E: Numeric>(tile: &StridedTile<E>, fragment: &mut Self::Query);

    fn load_key_transposed<E: Numeric>(
        tile: &StridedTile<E>,
        fragment: &mut Self::Key,
        #[comptime] config: Self::Config,
    );
    fn load_value<E: Numeric>(
        tile: &StridedTile<E>,
        fragment: &mut Self::Value,
        #[comptime] config: Self::Config,
    );
    fn load_mask<E: Numeric>(
        tile: &StridedTile<E>,
        fragment: &mut Self::Mask,
        #[comptime] config: Self::Config,
    );

    fn write_results<E: Float>(
        out: &Self::Accumulator,
        slice: &mut SliceMut<Line<E>>,
        #[comptime] config: Self::Config,
    );
}

/// Configuration for the Tile Attention level
pub trait TileAttentionConfig:
    Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static
{
    fn plane_dim(&self) -> u32;
    fn num_planes(&self) -> u32;
    fn attention_tile_size(&self) -> AttentionTileSize;
    fn num_rows_per_unit(&self) -> u32;
    fn causal_mask(&self) -> bool;
    fn materialized_mask(&self) -> bool;
}

pub trait TileAttentionFamily: Send + Sync + 'static {
    /// The specific TileMatmul implementation associated with this family.
    type TileAttention<AP: AttentionPrecision>: TileAttention<AP, Config = Self::Config>;

    /// The configuration type associated with this matmul family.
    type Config: TileAttentionConfig;

    /// Returns whether this tile matmul requires specialized hardware accelerators (e.g., tensor cores).
    fn requires_accelerator() -> bool;

    /// Returns the compute resources required to run this tile matmul.
    fn computation_resources() -> Result<CubeDimResource, InvalidConfigError>;

    /// Constructs the configuration based on the algorithm's blueprint.
    ///
    /// This function may return an error if the configuration cannot be supported.
    fn expand_config(blueprint: &AttentionBlueprint) -> Result<Self::Config, AttentionSetupError>;
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct SharedTileAttentionConfig {
    pub plane_dim: u32,
    pub num_planes: u32,
    pub attention_tile_size: AttentionTileSize,
    pub causal_mask: bool,
    pub materialized_mask: bool,
}
