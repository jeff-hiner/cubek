//! Accelerated tile attention using CMMA (tensor core) instructions.

use cubecl;
use cubecl::prelude::*;
use cubek_matmul::components::tile::StridedTile;

use crate::components::tile::accelerated::hybrid_fragment::HybridFragment;
use crate::components::tile::accelerated::local_tile::LocalTile;
use crate::components::tile::accelerated::local_tile::LocalTileLayout;
use crate::components::tile::accelerated::setup::BlackboxAcceleratedAttentionMatmulConfig;
use crate::components::tile::{TileAttention, TileAttentionConfig as _};
use crate::definition::AttentionPrecision;
use crate::definition::attention_types::*;

/// Uses accelerated instruction, but relies on shared memory for row-dependent computations
/// because the fragment layout is blackbox.
///
/// This implementation uses f16×f16→f32 CMMA for both Q·K^T and P×V matmuls.
pub struct BlackboxAcceleratedTileAttention;

#[cube]
impl<AP: AttentionPrecision> TileAttention<AP> for BlackboxAcceleratedTileAttention {
    type Config = BlackboxAcceleratedAttentionMatmulConfig;

    type Query = cmma::Matrix<QT<AP>>;
    /// Key fragment for Q·K^T CMMA. Uses VT (f16 for float precisions).
    type Key = cmma::Matrix<VT<AP>>;
    /// Value fragment for P×V CMMA. Uses VT (f16 for float precisions).
    type Value = cmma::Matrix<VT<AP>>;
    type Mask = LocalTile<SM<AP>>;
    type Softmax = HybridFragment<SM<AP>>;
    type SoftmaxRow = LocalTile<SM<AP>>;
    type Accumulator = HybridFragment<ACC<AP>>;

    type FragmentLayout = LocalTileLayout;

    fn softmax_layout(#[comptime] config: Self::Config) -> LocalTileLayout {
        LocalTileLayout::new(
            (
                config.attention_tile_size().seq_q,
                config.attention_tile_size().seq_kv,
            ),
            config.shared.plane_dim,
            config.inner_layout,
        )
    }

    fn score_matmul(
        lhs: &Self::Query,
        rhs: &Self::Key,
        _key_tile: &StridedTile<KS<AP>>,
        out: &mut Self::Softmax,
        #[comptime] _config: Self::Config,
    ) {
        // Float path: f16×f16→f32 CMMA for Q·K^T
        let out = &out.fragment;
        cmma::execute::<QT<AP>, VT<AP>, SM<AP>, SM<AP>>(lhs, rhs, out, out);
    }

    fn value_matmul(
        lhs: &Self::Softmax,
        rhs: &Self::Value,
        out: &mut Self::Accumulator,
        #[comptime] _config: Self::Config,
    ) {
        // Cast f32 softmax → f16 for P×V matmul.
        // SM<AP> is f32 for numerical precision during softmax computation,
        // but hardware CMMA only supports f16×f16 inputs on most GPUs.
        let lhs_f16 = cmma::cast::<SM<AP>, VT<AP>>(&lhs.fragment);
        let out = &out.fragment;
        // f16×f16→f32 CMMA for P×V
        cmma::execute::<VT<AP>, VT<AP>, ACC<AP>, ACC<AP>>(&lhs_f16, rhs, out, out);
    }

    fn allocate_query(#[comptime] config: Self::Config) -> Self::Query {
        let size = config.attention_tile_size().to_score_matmul_tile_size();

        unsafe {
            cmma::Matrix::<QT<AP>>::uninitialized(
                cmma::MatrixIdent::A,
                size.m() as usize,
                size.n() as usize,
                size.k() as usize,
                cmma::MatrixLayout::RowMajor,
            )
        }
    }

    fn allocate_key(#[comptime] config: Self::Config) -> Self::Key {
        let size = config.attention_tile_size();
        unsafe {
            cmma::Matrix::<VT<AP>>::uninitialized(
                cmma::MatrixIdent::B,
                size.seq_q as usize,
                size.seq_kv as usize,
                size.head_dim as usize,
                cmma::MatrixLayout::ColMajor,
            )
        }
    }

    fn allocate_value(#[comptime] config: Self::Config) -> Self::Value {
        let size = config.attention_tile_size();
        unsafe {
            cmma::Matrix::<VT<AP>>::uninitialized(
                cmma::MatrixIdent::B,
                size.seq_q as usize,
                size.val_dim as usize,
                size.seq_kv as usize,
                cmma::MatrixLayout::RowMajor,
            )
        }
    }

    fn allocate_mask(#[comptime] config: Self::Config) -> Self::Mask {
        let size = config.attention_tile_size();
        LocalTile::new(LocalTileLayout::new(
            (size.seq_q, size.seq_kv),
            config.shared.plane_dim,
            config.inner_layout,
        ))
    }

    fn allocate_softmax(#[comptime] config: Self::Config) -> Self::Softmax {
        let size = config.attention_tile_size().to_score_matmul_tile_size();
        HybridFragment::new(size, config)
    }

    fn allocate_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        let size = config.attention_tile_size().to_value_matmul_tile_size();
        HybridFragment::new(size, config)
    }

    fn load_query<E: Numeric>(tile: &StridedTile<E>, fragment: &mut Self::Query) {
        let (slice, stride) = tile.as_unlined();
        cmma::load(fragment, &slice, stride);
    }

    fn load_key_transposed<E: Numeric>(
        tile: &StridedTile<E>,
        rhs: &mut Self::Key,
        #[comptime] _config: Self::Config,
    ) {
        let (slice, stride) = tile.as_unlined();
        cmma::load(rhs, &slice, stride);
    }

    fn load_value<E: Numeric>(
        tile: &StridedTile<E>,
        rhs: &mut Self::Value,
        #[comptime] _config: Self::Config,
    ) {
        let (slice, stride) = tile.as_unlined();
        cmma::load(rhs, &slice, stride);
    }

    fn load_mask<E: Numeric>(
        tile: &StridedTile<E>,
        mask: &mut Self::Mask,
        #[comptime] _config: Self::Config,
    ) {
        mask.load_from_strided_tile(tile)
    }

    fn write_results<E: Float>(
        out: &Self::Accumulator,
        slice: &mut SliceMut<Line<E>>,
        #[comptime] config: Self::Config,
    ) {
        let acc = cmma::cast::<ACC<AP>, E>(&out.fragment);
        cmma::store(
            slice,
            &acc,
            config.attention_tile_size().val_dim,
            cmma::MatrixLayout::RowMajor,
        );
    }
}
