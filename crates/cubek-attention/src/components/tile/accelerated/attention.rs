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
        _key_tile: &StridedTile<KS<AP>>,
        out: &mut Self::Softmax,
        #[comptime] config: Self::Config,
    ) {
        // CMMA path - uses lhs (Query CMMA) and rhs (Key CMMA)
        // key_tile is ignored here; it's used by score_matmul_scalar instead
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

    fn score_matmul_scalar<QE: Numeric>(
        query_scalar: &Slice<QE>,
        key_tile: &StridedTile<KS<AP>>,
        out: &mut Self::Softmax,
        #[comptime] config: Self::Config,
    ) {
        // DP4a path for INT8: reads Q from scalar storage, K from StridedTile,
        // uses vectorized INT8 dot products (4 multiply-adds per instruction),
        // writes directly to LocalTile. No sync needed before rowwise_mut!
        let size = config.attention_tile_size();
        let seq_kv = size.seq_kv;
        let head_dim = size.head_dim;

        // Get the local_tile for direct output
        let local_tile = out.local_tile_mut();

        // Layout parameters for computing absolute positions
        let unit_size_rows = local_tile.layout.unit_size.0;
        let unit_size_cols = local_tile.layout.unit_size.1;
        let num_units_per_row = comptime!(seq_kv / unit_size_cols);
        let row_jump = comptime!(config.shared.plane_dim / num_units_per_row);

        // DP4a processes 4 INT8 elements at a time
        let dp4a_size = 4u32;
        let head_dim_iters = comptime!(head_dim / dp4a_size);

        // Precompute K tile access parameters
        let k_line_size = key_tile.line_size;

        // Each unit computes its assigned output elements
        #[unroll]
        for r in 0..unit_size_rows {
            #[unroll]
            for c in 0..unit_size_cols {
                // Compute absolute position (same as LocalTileLayout::absolute_pos)
                let row_0 = UNIT_POS_X / num_units_per_row;
                let out_row = r * row_jump + row_0;
                let out_col = unit_size_cols * (UNIT_POS_X % num_units_per_row) + c;

                // K column access (same for all k iterations)
                let k_col_line = out_col / k_line_size;
                let k_col_offset = out_col % k_line_size;

                // Compute Q[out_row, :] · K[:, out_col] using DP4a
                let mut sum_i32 = 0i32;

                for k_iter in 0..head_dim_iters {
                    let k_base = k_iter * dp4a_size;

                    // Load 4 Q elements (contiguous in row-major storage)
                    let q_base_idx = out_row * head_dim + k_base;
                    let mut q_line = Line::<i8>::empty(4usize);
                    #[unroll]
                    for i in 0..4usize {
                        q_line[i] = i8::cast_from(query_scalar[(q_base_idx + i as u32) as usize]);
                    }

                    // Load 4 K elements (from consecutive rows, same column)
                    // K is stored transposed: (head_dim rows, seq_kv cols)
                    let mut k_line = Line::<i8>::empty(4usize);
                    #[unroll]
                    for i in 0..4usize {
                        let k_row = k_base + i as u32;
                        let k_val = key_tile.get_line(k_row, k_col_line)[k_col_offset as usize];
                        k_line[i] = i8::cast_from(k_val);
                    }

                    // DP4a: 4 INT8 multiply-adds in one instruction
                    sum_i32 += q_line.dot_i32(k_line);
                }

                // Cast i32 result to f32 and accumulate into local_tile
                let idx = (r * unit_size_cols + c) as usize;
                local_tile.array[idx] += SM::<AP>::cast_from(sum_i32);
            }
        }

        // Mark that local_tile has valid data - rowwise_mut() will skip SMEM operations
        out.mark_local_valid();
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
