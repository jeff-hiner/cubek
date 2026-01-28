use crate::{
    components::tile::{
        TileAttention, TileAttentionConfig as _,
        accelerated::{
            hybrid_fragment::HybridFragment,
            local_tile::{LocalTile, LocalTileLayout},
            setup::BlackboxAcceleratedAttentionMatmulConfig,
        },
    },
    definition::{AttentionPrecision, attention_types::*},
};
use cubecl::{self, prelude::*};
use cubek_matmul::components::tile::StridedTile;

/// Uses accelerated instruction, but relies on shared memory for row-dependent computations
/// because the fragment layout is blackbox
pub struct BlackboxAcceleratedTileAttention;

#[cube]
impl<AP: AttentionPrecision> TileAttention<AP> for BlackboxAcceleratedTileAttention {
    type Config = BlackboxAcceleratedAttentionMatmulConfig;

    type Query = cmma::Matrix<QT<AP>>;
    /// Key fragment for Q·K^T CMMA. Uses KT (i8 for INT8, f16 for float).
    type Key = cmma::Matrix<KT<AP>>;
    /// Value fragment for P×V CMMA. Uses VT (f16 for both INT8 and float).
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
        out: &mut Self::Softmax,
        #[comptime] config: Self::Config,
    ) {
        // Check if score accumulator type differs from softmax type (INT8 path)
        // For INT8: SACC=i32, SM=f32 → needs i32 CMMA then cast to f32
        // For Float: SACC=f32, SM=f32 → direct f32 CMMA
        let needs_conversion = comptime!(AP::REQUIRES_SCORE_CONVERSION);

        if needs_conversion {
            // INT8 path: CMMA outputs i32, then cast to f32
            let size = config.attention_tile_size().to_score_matmul_tile_size();
            let temp_acc = unsafe {
                cmma::Matrix::<SACC<AP>>::uninitialized(
                    cmma::MatrixIdent::Accumulator,
                    size.m as usize,
                    size.n as usize,
                    size.k as usize,
                    cmma::MatrixLayout::RowMajor,
                )
            };
            cmma::fill(&temp_acc, SACC::<AP>::from_int(0));
            cmma::execute::<QT<AP>, KT<AP>, SACC<AP>, SACC<AP>>(lhs, rhs, &temp_acc, &temp_acc);

            // Cast i32 → f32 and store to HybridFragment
            out.store_from_cmma_matrix(&temp_acc);
        } else {
            // Float path: direct CMMA to f32
            let out_fragment = &out.fragment;
            cmma::execute::<QT<AP>, KT<AP>, SM<AP>, SM<AP>>(lhs, rhs, out_fragment, out_fragment);
        }
    }

    fn value_matmul(
        lhs: &Self::Softmax,
        rhs: &Self::Value,
        out: &mut Self::Accumulator,
        #[comptime] _config: Self::Config,
    ) {
        // Cast softmax from SM to VT for tensor core CMMA.
        // On non-macOS: SM=f32, VT=f16 → need cast f32→f16
        // On macOS: SM=f16, VT=f16 → cast is identity
        // Reference: p = p.to(tl.float16); acc += tl.dot(p, v, out_dtype=tl.float16)
        let lhs_cast = cmma::cast::<SM<AP>, VT<AP>>(&lhs.fragment);
        let out = &out.fragment;
        cmma::execute::<VT<AP>, VT<AP>, ACC<AP>, ACC<AP>>(&lhs_cast, rhs, out, out);
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
            cmma::Matrix::<KT<AP>>::uninitialized(
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
