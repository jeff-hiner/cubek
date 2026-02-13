//! INT8 CMMA tile attention implementation.
//!
//! Uses i8×i8→i32 CMMA for Q·K^T (with per-tile quantization) and f16×f16→f32 CMMA for P×V.

use cubecl::{self, prelude::*};
use cubek_matmul::components::tile::StridedTile;

use crate::components::tile::accelerated::local_tile::{LocalTile, LocalTileLayout};
use crate::components::tile::int8_cmma::setup::Int8CmmaAttentionConfig;
use crate::components::tile::{
    FragmentAccumulator, FragmentAccumulatorExpand, FragmentSoftmax, FragmentSoftmaxExpand,
    RowWise, TileAttention, TileAttentionConfig as _,
};
use crate::definition::AttentionPrecision;
use crate::definition::attention_types::{ACC, KS, MSK, SM};

/// log2(e) constant for converting exp to exp2 in softmax
#[expect(dead_code, reason = "will be used when proper quantization is implemented")]
const LOG2_E: f32 = std::f32::consts::LOG2_E;

/// Minimum scale value to avoid division by zero during quantization.
/// Using 1e-6 as a small but non-zero value.
const MIN_QUANT_SCALE: f32 = 1e-6;

/// INT8 CMMA tile attention implementation.
///
/// Uses tensor core CMMA operations:
/// - Q·K^T: i8×i8→i32 CMMA with per-tile quantization
/// - P×V: f16×f16→f32 CMMA (V stays float)
pub struct Int8CmmaTileAttention;

/// Tile layout info for INT8 tiles
#[derive(CubeType, Copy, Clone)]
pub struct Int8TileLayout {
    #[cube(comptime)]
    num_rows: u32,
    #[cube(comptime)]
    num_cols: u32,
}

#[cube]
impl Int8TileLayout {
    pub fn new(#[comptime] num_rows: u32, #[comptime] num_cols: u32) -> Int8TileLayout {
        Int8TileLayout { num_rows, num_cols }
    }
}

/// Query tile with INT8 quantization for CMMA.
#[derive(CubeType)]
pub struct Int8QueryTile {
    /// INT8 CMMA matrix for Q·K^T computation.
    pub fragment: cmma::Matrix<i8>,
    /// Quantization scale: original = quantized * scale.
    /// Includes sm_scale = (1/√head_dim) * log2(e) baked in.
    pub scale: f32,
    /// Layout info for loading
    pub layout: Int8TileLayout,
    /// Number of planes for per-plane SharedMemory allocation.
    #[cube(comptime)]
    pub num_planes: u32,
}

/// Key tile with INT8 quantization for CMMA.
#[derive(CubeType)]
pub struct Int8KeyTile {
    /// INT8 CMMA matrix for Q·K^T computation.
    pub fragment: cmma::Matrix<i8>,
    /// Quantization scale: original = quantized * scale.
    pub scale: f32,
    /// Layout info for loading
    pub layout: Int8TileLayout,
    /// Number of planes for per-plane SharedMemory allocation.
    #[cube(comptime)]
    pub num_planes: u32,
}

/// Softmax fragment that handles i32→E conversion after INT8 CMMA.
/// Generic over E: Float to satisfy trait bounds for all precisions.
#[derive(CubeType)]
pub struct Int8CmmaSoftmax<E: Float> {
    /// CMMA fragment for softmax computation and P×V matmul input (f32).
    pub fragment: cmma::Matrix<E>,
    /// i32 accumulator for Q·K^T CMMA. Persists across head_dim partitions.
    pub acc_i32: cmma::Matrix<i32>,
    /// Combined quantization scale (q_scale * k_scale) to apply after all partitions.
    pub combined_scale: f32,
    /// Shared memory slice for this plane's softmax tile (f32).
    smem_slice: SliceMut<E>,
    /// Shared memory slice for i32 accumulator storage.
    /// Needed because acc_i32 has K=head_dim layout, but fragment needs K=seq_kv.
    /// We store i32 to SMEM, then convert element-wise to avoid CMMA layout mismatch.
    smem_i32_slice: SliceMut<i32>,
    /// Local tile for row-wise operations.
    local_tile: LocalTile<E>,
    /// Number of rows in the softmax tile (seq_q).
    #[cube(comptime)]
    seq_q: u32,
    /// Stride for SMEM layout (= seq_kv).
    #[cube(comptime)]
    stride: u32,
}

#[cube]
impl<E: Float> Int8CmmaSoftmax<E> {
    pub fn new(
        #[comptime] seq_q: u32,
        #[comptime] seq_kv: u32,
        #[comptime] config: Int8CmmaAttentionConfig,
    ) -> Self {
        // f32 fragment used for softmax probabilities and as A matrix in value_matmul (P×V)
        // CRITICAL: Must use K=head_dim to match acc_i32's K dimension.
        // When cmma::store writes acc_i32 to SMEM and cmma::load_with_layout reads into
        // fragment, they must have the SAME K to ensure thread mappings match.
        // For value_matmul, cmma::cast handles converting Accumulator to matrix A.
        let fragment = unsafe {
            cmma::Matrix::<E>::uninitialized(
                cmma::MatrixIdent::Accumulator,
                seq_q as usize,
                seq_kv as usize,
                config.shared.attention_tile_size.head_dim as usize, // K=head_dim to match acc_i32
                cmma::MatrixLayout::RowMajor,
            )
        };

        // i32 accumulator for Q·K^T CMMA - K=head_dim
        let acc_i32 = unsafe {
            cmma::Matrix::<i32>::uninitialized(
                cmma::MatrixIdent::Accumulator,
                seq_q as usize,
                seq_kv as usize,
                config.shared.attention_tile_size.head_dim as usize,
                cmma::MatrixLayout::RowMajor,
            )
        };

        let array_tile_layout = LocalTileLayout::new(
            (seq_q, seq_kv),
            config.shared.plane_dim,
            config.inner_layout,
        );

        let local_tile = LocalTile::new(array_tile_layout);

        let smem_slot_size = seq_q * seq_kv;
        let smem_slice_start = UNIT_POS_Y * smem_slot_size;

        // f32 shared memory for softmax values
        let mut shared_memory =
            SharedMemory::new(config.shared.num_planes as usize * smem_slot_size as usize);
        let smem_slice = shared_memory.slice_mut(
            smem_slice_start as usize,
            (smem_slice_start + smem_slot_size) as usize,
        );

        // i32 shared memory for acc_i32 storage.
        // We need separate i32 SMEM because acc_i32 has K=head_dim CMMA layout,
        // but self.fragment needs K=seq_kv. Storing i32 to SMEM then converting
        // element-wise to f32 avoids the CMMA fragment layout mismatch.
        let mut shared_memory_i32 =
            SharedMemory::<i32>::new(config.shared.num_planes as usize * smem_slot_size as usize);
        let smem_i32_slice = shared_memory_i32.slice_mut(
            smem_slice_start as usize,
            (smem_slice_start + smem_slot_size) as usize,
        );

        Int8CmmaSoftmax::<E> {
            fragment,
            acc_i32,
            combined_scale: 1.0f32,
            smem_slice,
            smem_i32_slice,
            local_tile,
            seq_q,
            stride: seq_kv,
        }
    }

    fn zero(&mut self) {
        // Zero both the f32 fragment and the i32 accumulator
        cmma::fill(&self.fragment, E::from_int(0));
        cmma::fill(&self.acc_i32, 0i32);
        self.combined_scale = 1.0f32;
    }
}

#[cube]
impl<E: Float> FragmentSoftmax<E> for Int8CmmaSoftmax<E> {
    type Layout = LocalTileLayout;
    type SoftmaxScore = cmma::Matrix<E>;
    type SoftmaxRowFormat = LocalTile<E>;
    type SoftmaxVal = cmma::Matrix<E>;

    fn set_combined_scale(&mut self, scale: f32) {
        self.combined_scale = scale;
    }

    fn rowwise_mut(&mut self) -> &mut Self::SoftmaxRowFormat {
        // Store acc_i32 directly to i32 SMEM.
        // This uses acc_i32's native K=head_dim CMMA layout correctly.
        cmma::store(
            &mut self.smem_i32_slice,
            &self.acc_i32,
            self.stride,
            cmma::MatrixLayout::RowMajor,
        );

        sync_cube();

        // Convert i32 → f32 element-wise in SMEM and apply quantization scale.
        // This avoids CMMA layout mismatch: acc_i32 (K=head_dim) vs fragment (K=seq_kv).
        // Element-wise conversion has no CMMA layout dependencies.
        let seq_q = comptime!(self.seq_q);
        let stride = comptime!(self.stride);
        let num_elements = seq_q * stride;
        let elements_per_thread = (num_elements + 31) / 32;
        let scale = E::cast_from(self.combined_scale);
        for i in 0..elements_per_thread {
            let idx = UNIT_POS_X + i * 32;
            if idx < num_elements {
                // Read i32 from i32 SMEM, convert to f32, apply scale, write to f32 SMEM
                let i32_val = self.smem_i32_slice[idx as usize];
                self.smem_slice[idx as usize] = E::cast_from(i32_val) * scale;
            }
        }

        sync_cube();

        self.local_tile.load_from_slice(&self.smem_slice.to_slice());

        sync_cube();

        &mut self.local_tile
    }

    fn update_from_rowwise(&mut self) {
        self.local_tile.store_to(&mut self.smem_slice);

        sync_cube();

        cmma::load_with_layout(
            &self.fragment,
            &self.smem_slice.to_slice(),
            self.stride,
            cmma::MatrixLayout::RowMajor,
        )
    }

    fn zero(&mut self) {
        self.zero();
    }
}

/// Accumulator fragment for INT8 CMMA attention.
/// Generic over E: Float to satisfy trait bounds for all precisions.
#[derive(CubeType)]
pub struct Int8CmmaAccumulator<E: Float> {
    /// CMMA fragment for P×V accumulation.
    pub fragment: cmma::Matrix<E>,
    /// Shared memory slice for this plane.
    pub smem_slice: SliceMut<E>,
    /// Local tile for row-wise scaling.
    local_tile: LocalTile<E>,
    /// Stride for SMEM layout.
    #[cube(comptime)]
    pub stride: u32,
}

#[cube]
impl<E: Float> Int8CmmaAccumulator<E> {
    pub fn new(
        #[comptime] seq_q: u32,
        #[comptime] val_dim: u32,
        #[comptime] seq_kv: u32,
        #[comptime] config: Int8CmmaAttentionConfig,
    ) -> Self {
        let fragment = unsafe {
            cmma::Matrix::<E>::uninitialized(
                cmma::MatrixIdent::Accumulator,
                seq_q as usize,
                val_dim as usize,
                seq_kv as usize,
                cmma::MatrixLayout::RowMajor,
            )
        };

        let array_tile_layout = LocalTileLayout::new(
            (seq_q, val_dim),
            config.shared.plane_dim,
            config.inner_layout,
        );

        let local_tile = LocalTile::new(array_tile_layout);

        let smem_slot_size = seq_q * val_dim;
        let smem_slice_start = UNIT_POS_Y * smem_slot_size;
        let mut shared_memory =
            SharedMemory::new(config.shared.num_planes as usize * smem_slot_size as usize);
        let smem_slice = shared_memory.slice_mut(
            smem_slice_start as usize,
            (smem_slice_start + smem_slot_size) as usize,
        );

        Int8CmmaAccumulator::<E> {
            fragment,
            smem_slice,
            local_tile,
            stride: val_dim,
        }
    }

    fn zero(&mut self) {
        cmma::fill(&self.fragment, E::from_int(0));
    }

    fn rowwise_mut(&mut self) -> &mut LocalTile<E> {
        cmma::store(
            &mut self.smem_slice,
            &self.fragment,
            self.stride,
            cmma::MatrixLayout::RowMajor,
        );

        sync_cube();

        self.local_tile.load_from_slice(&self.smem_slice.to_slice());

        sync_cube();

        &mut self.local_tile
    }

    fn update_from_rowwise(&mut self) {
        self.local_tile.store_to(&mut self.smem_slice);

        sync_cube();

        cmma::load_with_layout(
            &self.fragment,
            &self.smem_slice.to_slice(),
            self.stride,
            cmma::MatrixLayout::RowMajor,
        )
    }
}

#[cube]
impl<E: Float> FragmentAccumulator<E> for Int8CmmaAccumulator<E> {
    fn rowwise_scale(&mut self, val: &RowWise<E>) {
        let local_tile = self.rowwise_mut();
        local_tile.rowwise_scale(val);
        self.update_from_rowwise();
    }

    fn zero(&mut self) {
        self.zero();
    }
}

#[cube]
impl<AP: AttentionPrecision> TileAttention<AP> for Int8CmmaTileAttention {
    type Config = Int8CmmaAttentionConfig;

    type Query = Int8QueryTile;
    type Key = Int8KeyTile;
    type Value = cmma::Matrix<half::f16>;
    /// Mask uses the precision's mask type
    type Mask = LocalTile<MSK<AP>>;
    /// Softmax fragment using the precision's softmax type
    type Softmax = Int8CmmaSoftmax<SM<AP>>;
    /// Softmax row format using the precision's softmax type
    type SoftmaxRow = LocalTile<SM<AP>>;
    /// Accumulator using the precision's accumulator type
    type Accumulator = Int8CmmaAccumulator<ACC<AP>>;

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
        // Accumulate Q·K^T using i8×i8→i32 CMMA into the persistent accumulator.
        // This is called multiple times for head_dim partitions; results accumulate.
        cmma::execute::<i8, i8, i32, i32>(
            &lhs.fragment,
            &rhs.fragment,
            &out.acc_i32,
            &out.acc_i32,
        );

        // Store the quantization scale (same for all partitions with fixed scale)
        // Scale is applied once in rowwise_mut after all partitions complete
        out.combined_scale = lhs.scale * rhs.scale;
    }

    fn value_matmul(
        lhs: &Self::Softmax,
        rhs: &Self::Value,
        out: &mut Self::Accumulator,
        #[comptime] _config: Self::Config,
    ) {
        // P×V uses f16×f16→f32 CMMA (same as BlackboxAccelerated)
        // Cast softmax (SM<AP>, likely f32) to f16 for CMMA input
        let lhs_f16 = cmma::cast::<SM<AP>, half::f16>(&lhs.fragment);

        // Execute f16×f16→f32 CMMA for P×V
        cmma::execute::<half::f16, half::f16, ACC<AP>, ACC<AP>>(
            &lhs_f16,
            rhs,
            &out.fragment,
            &out.fragment,
        );
    }

    fn allocate_query(#[comptime] config: Self::Config) -> Self::Query {
        let size = config.attention_tile_size();
        let fragment = unsafe {
            cmma::Matrix::<i8>::uninitialized(
                cmma::MatrixIdent::A,
                size.seq_q as usize,
                size.seq_kv as usize,
                size.head_dim as usize,
                cmma::MatrixLayout::RowMajor,
            )
        };
        Int8QueryTile {
            fragment,
            scale: 1.0f32,
            layout: Int8TileLayout::new(size.seq_q, size.head_dim),
            num_planes: config.shared.num_planes,
        }
    }

    fn allocate_key(#[comptime] config: Self::Config) -> Self::Key {
        let size = config.attention_tile_size();
        let fragment = unsafe {
            cmma::Matrix::<i8>::uninitialized(
                cmma::MatrixIdent::B,
                size.seq_q as usize,
                size.seq_kv as usize,
                size.head_dim as usize,
                cmma::MatrixLayout::ColMajor,
            )
        };
        Int8KeyTile {
            fragment,
            scale: 1.0f32,
            layout: Int8TileLayout::new(size.head_dim, size.seq_kv),
            num_planes: config.shared.num_planes,
        }
    }

    fn allocate_value(#[comptime] config: Self::Config) -> Self::Value {
        let size = config.attention_tile_size();
        unsafe {
            cmma::Matrix::<half::f16>::uninitialized(
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
        let size = config.attention_tile_size();
        Int8CmmaSoftmax::new(size.seq_q, size.seq_kv, config)
    }

    fn allocate_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        let size = config.attention_tile_size();
        Int8CmmaAccumulator::new(size.seq_q, size.val_dim, size.seq_kv, config)
    }

    fn load_query<E: Numeric>(tile: &StridedTile<E>, fragment: &mut Self::Query) {
        // With pre-quantization, the input is already i8.
        // We load it directly into SMEM then into the CMMA fragment.
        // The scale is stored per-row in a separate scale tensor.
        // For simplicity, we use scale=1.0 here and apply actual scales during dequantization.
        let num_rows = comptime!(fragment.layout.num_rows); // seq_q
        let num_cols = comptime!(fragment.layout.num_cols); // head_dim
        let num_planes = comptime!(fragment.num_planes);
        let line_size = comptime!(tile.line_size);

        let smem_slot_size = num_rows * num_cols;
        let smem_slice_start = UNIT_POS_Y * smem_slot_size;
        let mut smem = SharedMemory::<i8>::new(num_planes as usize * smem_slot_size as usize);

        let elements_per_thread = comptime!((smem_slot_size + 31) / 32);

        // Read the tile's stride value - this is set by QueryReader::get_tile
        let tile_stride = tile.stride;
        let tile_start = tile.start;

        // Pre-quantized: data is already i8, just copy to SMEM
        // Since E should be i8 for pre-quantized path, we can cast directly
        #[unroll]
        for i in 0..elements_per_thread {
            let linear_idx = UNIT_POS_X + i * 32;
            if linear_idx < smem_slot_size {
                let row = linear_idx / num_cols;
                let col = linear_idx % num_cols;
                let line_offset = tile_start + row * tile_stride + col / line_size;
                let elem_in_line = col % line_size;
                let line = tile.stage[line_offset as usize];
                // For pre-quantized i8 data, E=i8, so cast_from is identity
                let val = i8::cast_from(line[elem_in_line as usize]);
                smem[(smem_slice_start + linear_idx) as usize] = val;
            }
        }
        sync_cube();

        // Scale is stored in the scale tensor, accessible via Int8ScaleAccess.
        // For now, use scale=1.0 here. The actual scale will be applied during dequantization.
        // This works because: actual_score = i32_result * q_scale * k_scale
        // We pass scales separately and apply them in rowwise_mut().
        fragment.scale = 1.0f32;

        let smem_slice =
            smem.slice(smem_slice_start as usize, (smem_slice_start + smem_slot_size) as usize);
        cmma::load(&fragment.fragment, &smem_slice, num_cols);
    }

    fn load_key_transposed<E: Numeric>(
        tile: &StridedTile<E>,
        fragment: &mut Self::Key,
        #[comptime] _config: Self::Config,
    ) {
        // With pre-quantization, the input is already i8.
        // We load it directly into SMEM then into the CMMA fragment.
        // Fragment expects K^T: [head_dim, seq_kv] = [K, N]
        let k_dim = comptime!(fragment.layout.num_rows); // head_dim = 32
        let n_dim = comptime!(fragment.layout.num_cols); // seq_kv = 16
        let num_planes = comptime!(fragment.num_planes);
        let line_size = comptime!(tile.line_size);

        // Source K tile is [seq_kv, head_dim] = [N, K] = [16, 32]
        let src_cols = k_dim; // head_dim (contiguous dimension)
        let src_size = n_dim * k_dim;

        let smem_slot_size = k_dim * n_dim;
        let smem_slice_start = UNIT_POS_Y * smem_slot_size;
        let mut smem = SharedMemory::<i8>::new(num_planes as usize * smem_slot_size as usize);

        let elements_per_thread = comptime!((src_size + 31) / 32);

        // Pre-quantized: data is already i8, just copy to SMEM
        #[unroll]
        for i in 0..elements_per_thread {
            let linear_idx = UNIT_POS_X + i * 32;
            if linear_idx < src_size {
                let src_row = linear_idx / src_cols;
                let src_col = linear_idx % src_cols;
                // Use get_line for proper strided access
                let line = tile.get_line(src_row, src_col / line_size);
                // For pre-quantized i8 data, E=i8, so cast_from is identity
                let val = i8::cast_from(line[(src_col % line_size) as usize]);
                smem[(smem_slice_start + linear_idx) as usize] = val;
            }
        }
        sync_cube();

        // Scale is stored in the scale tensor, accessible via Int8ScaleAccess.
        // For now, use scale=1.0 here. The actual scale will be applied during dequantization.
        fragment.scale = 1.0f32;

        let smem_slice =
            smem.slice(smem_slice_start as usize, (smem_slice_start + smem_slot_size) as usize);
        cmma::load(&fragment.fragment, &smem_slice, k_dim);
    }

    fn load_value<E: Numeric>(
        tile: &StridedTile<E>,
        fragment: &mut Self::Value,
        #[comptime] _config: Self::Config,
    ) {
        // Value stays as f16 for P×V matmul (no quantization)
        let (slice, stride) = tile.as_unlined();
        cmma::load(fragment, &slice, stride);
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
        // Cast from ACC<AP> (expected to be f32) to output type E
        let acc = cmma::cast::<ACC<AP>, E>(&out.fragment);
        cmma::store(
            slice,
            &acc,
            config.attention_tile_size().val_dim,
            cmma::MatrixLayout::RowMajor,
        );
    }
}
