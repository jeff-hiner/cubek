//! INT8 CMMA tile attention implementation.
//!
//! Uses i8×i8→i32 CMMA for Q·K^T and f16×f16→f32 CMMA for P×V.

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
const LOG2_E: f32 = std::f32::consts::LOG2_E;

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
}

/// Softmax fragment that handles i32→E conversion after INT8 CMMA.
/// Generic over E: Float to satisfy trait bounds for all precisions.
#[derive(CubeType)]
pub struct Int8CmmaSoftmax<E: Float> {
    /// CMMA fragment for softmax computation and P×V matmul input.
    pub fragment: cmma::Matrix<E>,
    /// Shared memory slice for this plane's softmax tile.
    smem_slice: SliceMut<E>,
    /// Local tile for row-wise operations.
    local_tile: LocalTile<E>,
    /// Stride for SMEM layout.
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
        let fragment = unsafe {
            cmma::Matrix::<E>::uninitialized(
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
        let mut shared_memory =
            SharedMemory::new(config.shared.num_planes as usize * smem_slot_size as usize);
        let smem_slice = shared_memory.slice_mut(
            smem_slice_start as usize,
            (smem_slice_start + smem_slot_size) as usize,
        );

        Int8CmmaSoftmax::<E> {
            fragment,
            smem_slice,
            local_tile,
            stride: seq_kv,
        }
    }

    fn zero(&mut self) {
        cmma::fill(&self.fragment, E::from_int(0));
    }
}

#[cube]
impl<E: Float> FragmentSoftmax<E> for Int8CmmaSoftmax<E> {
    type Layout = LocalTileLayout;
    type SoftmaxScore = cmma::Matrix<E>;
    type SoftmaxRowFormat = LocalTile<E>;
    type SoftmaxVal = cmma::Matrix<E>;

    fn rowwise_mut(&mut self) -> &mut Self::SoftmaxRowFormat {
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
        #[comptime] config: Self::Config,
    ) {
        // INT8 CMMA path: i8×i8→i32, then dequantize to SM<AP>
        let size = config.attention_tile_size();

        // Create temporary i32 accumulator for CMMA output
        let i32_acc = unsafe {
            cmma::Matrix::<i32>::uninitialized(
                cmma::MatrixIdent::Accumulator,
                size.seq_q as usize,
                size.seq_kv as usize,
                size.head_dim as usize,
                cmma::MatrixLayout::RowMajor,
            )
        };
        cmma::fill(&i32_acc, 0i32);

        // Execute i8×i8→i32 CMMA
        cmma::execute::<i8, i8, i32, i32>(&lhs.fragment, &rhs.fragment, &i32_acc, &i32_acc);

        // Combined scale for dequantization: q_scale * k_scale
        let combined_scale = lhs.scale * rhs.scale;

        // Store i32 results to SMEM, dequantize to softmax type
        // We need to go through SMEM because we can't directly convert CMMA fragments
        let smem_slot_size = size.seq_q * size.seq_kv;
        let smem_slice_start = UNIT_POS_Y * smem_slot_size;
        let mut i32_smem = SharedMemory::<i32>::new(
            config.shared.num_planes as usize * smem_slot_size as usize,
        );
        let mut i32_slice = i32_smem.slice_mut(
            smem_slice_start as usize,
            (smem_slice_start + smem_slot_size) as usize,
        );

        cmma::store(&mut i32_slice, &i32_acc, size.seq_kv, cmma::MatrixLayout::RowMajor);

        sync_cube();

        // Dequantize: read i32 from SMEM, apply scale, write to output SMEM
        let num_elements = smem_slot_size / config.shared.plane_dim;
        let start_idx = UNIT_POS_X * num_elements;

        for i in 0..num_elements {
            let idx = start_idx + i;
            let i32_val = i32_slice.to_slice()[idx as usize];
            // Dequantize: i32 -> f32 -> SM<AP>
            let f32_val = f32::cast_from(i32_val) * combined_scale;
            out.smem_slice[idx as usize] = SM::<AP>::cast_from(f32_val);
        }

        sync_cube();

        // Load dequantized values into CMMA fragment
        cmma::load_with_layout(
            &out.fragment,
            &out.smem_slice.to_slice(),
            size.seq_kv,
            cmma::MatrixLayout::RowMajor,
        );
    }

    fn value_matmul(
        lhs: &Self::Softmax,
        rhs: &Self::Value,
        out: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        // Reference pattern: p = p.to(tl.float16); acc += tl.dot(p, v, out_dtype=tl.float16)
        // Use f16×f16→f16 CMMA (less register pressure), then add to ACC<AP> accumulator
        //
        // TODO: Minimize SMEM sync overhead. Currently we:
        //   1. Store ACC<AP> acc to SMEM
        //   2. Store f16 result to SMEM
        //   3. Add in registers (via f32 intermediate)
        //   4. Reload ACC<AP> acc
        // Could potentially pipeline or restructure to reduce syncs.
        let size = config.attention_tile_size();

        // Cast softmax SM<AP> → f16 for P×V matmul
        // Note: SM<AP> is expected to be f32 for all defined precisions
        let p_f16 = cmma::cast::<SM<AP>, half::f16>(&lhs.fragment);

        // Create zero f16 accumulator for the dot product
        let zero_f16 = unsafe {
            cmma::Matrix::<half::f16>::uninitialized(
                cmma::MatrixIdent::Accumulator,
                size.seq_q as usize,
                size.val_dim as usize,
                size.seq_kv as usize,
                cmma::MatrixLayout::RowMajor,
            )
        };
        cmma::fill(&zero_f16, half::f16::ZERO);

        // Execute f16×f16→f16 CMMA (reference: out_dtype=tl.float16)
        let pv_f16 = unsafe {
            cmma::Matrix::<half::f16>::uninitialized(
                cmma::MatrixIdent::Accumulator,
                size.seq_q as usize,
                size.val_dim as usize,
                size.seq_kv as usize,
                cmma::MatrixLayout::RowMajor,
            )
        };
        cmma::execute::<half::f16, half::f16, half::f16, half::f16>(&p_f16, rhs, &zero_f16, &pv_f16);

        // Store f16 result to SMEM, then add to ACC<AP> accumulator
        // First, store current ACC<AP> accumulator to SMEM
        cmma::store(
            &mut out.smem_slice,
            &out.fragment,
            out.stride,
            cmma::MatrixLayout::RowMajor,
        );

        sync_cube();

        // Store f16 result to separate SMEM, convert and add
        let smem_slot_size = size.seq_q * size.val_dim;
        let mut f16_smem = SharedMemory::<half::f16>::new(smem_slot_size as usize);
        let mut f16_slice = f16_smem.slice_mut(0, smem_slot_size as usize);
        cmma::store(&mut f16_slice, &pv_f16, size.val_dim, cmma::MatrixLayout::RowMajor);

        sync_cube();

        // Add f16 values to the ACC<AP> accumulator in SMEM
        // We go through f32 intermediate for the addition, then back to ACC<AP>
        let num_elements = smem_slot_size / config.shared.plane_dim;
        let start_idx = UNIT_POS_X * num_elements;

        for i in 0..num_elements {
            let idx = start_idx + i;
            let f16_val = f16_slice.to_slice()[idx as usize];
            let f16_as_f32 = f32::cast_from(f16_val);
            // Read current ACC<AP> value, convert to f32, add, convert back
            let current = f32::cast_from(out.smem_slice[idx as usize]);
            let new_val = current + f16_as_f32;
            out.smem_slice[idx as usize] = ACC::<AP>::cast_from(new_val);
        }

        sync_cube();

        // Reload ACC<AP> accumulator from SMEM
        cmma::load_with_layout(
            &out.fragment,
            &out.smem_slice.to_slice(),
            out.stride,
            cmma::MatrixLayout::RowMajor,
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
        // Quantize float input to INT8 with sm_scale baked in
        // Get dimensions from fragment layout (set during allocation)
        let num_rows = comptime!(fragment.layout.num_rows);
        let num_cols = comptime!(fragment.layout.num_cols);

        // Compute sm_scale = (1/√head_dim) * log2(e)
        // This bakes in attention temperature and exp→exp2 conversion
        let head_dim_f32 = comptime!(num_cols as f32);
        let sm_scale = f32::recip(f32::sqrt(head_dim_f32)) * LOG2_E;

        // First pass: find max(abs(val * sm_scale)) for quantization scale
        // Using flat array iteration for simplicity
        let (slice, stride) = tile.as_unlined();
        let mut max_abs = 0.0f32;
        for row in 0..num_rows {
            for col in 0..num_cols {
                let idx = row * stride + col;
                let val = f32::cast_from(slice[idx as usize]) * sm_scale;
                let abs_val = f32::abs(val);
                max_abs = f32::max(max_abs, abs_val);
            }
        }

        // Compute quantization scale
        let quant_scale = max_abs / 127.0 + 1e-10;
        fragment.scale = quant_scale;
        let inv_scale = 1.0 / quant_scale;

        // Second pass: quantize and load into SMEM for CMMA loading
        let total_elements = comptime!(num_rows * num_cols);
        let mut i8_smem = SharedMemory::<i8>::new(total_elements as usize);
        let mut i8_slice = i8_smem.slice_mut(0, total_elements as usize);

        for row in 0..num_rows {
            for col in 0..num_cols {
                let src_idx = row * stride + col;
                let dst_idx = row * num_cols + col;
                let val = f32::cast_from(slice[src_idx as usize]) * sm_scale;
                let quantized = f32::round(val * inv_scale);
                let clamped = clamp(quantized, -127.0, 127.0);
                i8_slice[dst_idx as usize] = i8::cast_from(clamped);
            }
        }

        sync_cube();

        // Load quantized data into CMMA fragment
        cmma::load(&fragment.fragment, &i8_slice.to_slice(), num_cols);
    }

    fn load_key_transposed<E: Numeric>(
        tile: &StridedTile<E>,
        fragment: &mut Self::Key,
        #[comptime] _config: Self::Config,
    ) {
        // Quantize float input to INT8 (K is not pre-scaled with sm_scale)
        // Key layout: (head_dim, seq_kv) for Q·K^T
        let num_rows = comptime!(fragment.layout.num_rows);
        let num_cols = comptime!(fragment.layout.num_cols);

        // First pass: find max(abs(val)) for quantization scale
        let (slice, stride) = tile.as_unlined();
        let mut max_abs = 0.0f32;
        for row in 0..num_rows {
            for col in 0..num_cols {
                let idx = row * stride + col;
                let val = f32::cast_from(slice[idx as usize]);
                let abs_val = f32::abs(val);
                max_abs = f32::max(max_abs, abs_val);
            }
        }

        // Compute quantization scale
        let quant_scale = max_abs / 127.0 + 1e-10;
        fragment.scale = quant_scale;
        let inv_scale = 1.0 / quant_scale;

        // Second pass: quantize and load into SMEM for CMMA loading
        // Note: K is loaded transposed (head_dim, seq_kv) for Q·K^T
        let total_elements = comptime!(num_rows * num_cols);
        let mut i8_smem = SharedMemory::<i8>::new(total_elements as usize);
        let mut i8_slice = i8_smem.slice_mut(0, total_elements as usize);

        for row in 0..num_rows {
            for col in 0..num_cols {
                let src_idx = row * stride + col;
                let dst_idx = row * num_cols + col;
                let val = f32::cast_from(slice[src_idx as usize]);
                let quantized = f32::round(val * inv_scale);
                let clamped = clamp(quantized, -127.0, 127.0);
                i8_slice[dst_idx as usize] = i8::cast_from(clamped);
            }
        }

        sync_cube();

        // Load quantized data into CMMA fragment
        cmma::load(&fragment.fragment, &i8_slice.to_slice(), num_cols);
    }

    fn load_value<E: Numeric>(
        tile: &StridedTile<E>,
        fragment: &mut Self::Value,
        #[comptime] _config: Self::Config,
    ) {
        // V stays f16, no quantization
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
