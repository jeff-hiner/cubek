use cubecl;
use cubecl::prelude::*;
use cubek_matmul::definition::TileSize;

use crate::components::tile::accelerated::local_tile::{LocalTile, LocalTileLayout};
use crate::components::tile::accelerated::setup::BlackboxAcceleratedAttentionMatmulConfig;
use crate::components::tile::{FragmentAccumulator, FragmentAccumulatorExpand};
use crate::components::tile::{FragmentSoftmax, FragmentSoftmaxExpand};
use crate::components::tile::{RowWise, TileAttentionConfig as _};

#[derive(CubeType)]
/// Navigates between cmma fragment (for matmuls) and shared memory (for row wise ops)
pub struct HybridFragment<E: Float> {
    // For matmul
    pub fragment: cmma::Matrix<E>,
    // A slice because knows only the slot for this plane
    smem_slice: SliceMut<E>,
    // Where to perform operations in register
    local_tile: LocalTile<E>,
    #[cube(comptime)]
    stride: u32,
}

#[cube]
impl<E: Float> HybridFragment<E> {
    pub fn new(
        #[comptime] tile_size: TileSize,
        #[comptime] config: BlackboxAcceleratedAttentionMatmulConfig,
    ) -> Self {
        let fragment = unsafe {
            cmma::Matrix::<E>::uninitialized(
                cmma::MatrixIdent::Accumulator,
                tile_size.m as usize,
                tile_size.n as usize,
                tile_size.k as usize,
                cmma::MatrixLayout::RowMajor,
            )
        };

        let array_tile_layout = LocalTileLayout::new(
            (tile_size.m, tile_size.n),
            config.shared.plane_dim,
            config.inner_layout,
        );

        let local_tile = LocalTile::new(array_tile_layout);

        let smem_slot_size = tile_size.m * tile_size.n;
        let smem_slice_start = UNIT_POS_Y * smem_slot_size;
        let mut shared_memory =
            SharedMemory::new(config.num_planes() as usize * smem_slot_size as usize);
        let smem_slice = shared_memory.slice_mut(
            smem_slice_start as usize,
            (smem_slice_start + smem_slot_size) as usize,
        );

        HybridFragment::<E> {
            fragment,
            smem_slice,
            local_tile,
            stride: tile_size.n,
        }
    }

    fn zero(&mut self) {
        cmma::fill(&self.fragment, E::from_int(0));
    }
}

#[cube]
impl<E: Float> FragmentSoftmax<E> for HybridFragment<E> {
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

    fn set_combined_scale(&mut self, _scale: f32) {
        // No-op for non-INT8 attention
    }
}

/// Like [`HybridFragment`], but outputs a V-typed CMMA fragment for P×V matmul.
///
/// Softmax computation stays in `E` (f32) for numerical stability.
/// `update_from_rowwise()` casts E→V during the SMEM store, producing a V-typed
/// fragment that can be used directly in P×V CMMA without an extra `cmma::cast`.
/// This halves SMEM bandwidth when V is narrower than E (e.g., f16 vs f32).
#[derive(CubeType)]
pub struct SoftmaxHybridFragment<E: Float, V: Numeric> {
    /// f32 CMMA accumulator for Q·K^T score accumulation.
    pub fragment: cmma::Matrix<E>,
    /// V-typed CMMA fragment for P×V matmul input (filled by `update_from_rowwise`).
    pub fragment_vt: cmma::Matrix<V>,
    /// f32 SMEM for `rowwise_mut` (CMMA fragment → local_tile).
    smem_slice: SliceMut<E>,
    /// V-typed SMEM for `update_from_rowwise` (local_tile → cast+store → fragment_vt).
    smem_vt_slice: SliceMut<V>,
    /// f32 local tile for softmax computation.
    local_tile: LocalTile<E>,
    /// Stride for SMEM layout (= seq_kv).
    #[cube(comptime)]
    stride: u32,
}

#[cube]
impl<E: Float, V: Numeric> SoftmaxHybridFragment<E, V> {
    pub fn new(
        #[comptime] tile_size: TileSize,
        #[comptime] config: BlackboxAcceleratedAttentionMatmulConfig,
    ) -> Self {
        let fragment = unsafe {
            cmma::Matrix::<E>::uninitialized(
                cmma::MatrixIdent::Accumulator,
                tile_size.m as usize,
                tile_size.n as usize,
                tile_size.k as usize,
                cmma::MatrixLayout::RowMajor,
            )
        };

        let fragment_vt = unsafe {
            cmma::Matrix::<V>::uninitialized(
                cmma::MatrixIdent::Accumulator,
                tile_size.m as usize,
                tile_size.n as usize,
                tile_size.k as usize,
                cmma::MatrixLayout::RowMajor,
            )
        };

        let array_tile_layout = LocalTileLayout::new(
            (tile_size.m, tile_size.n),
            config.shared.plane_dim,
            config.inner_layout,
        );

        let local_tile = LocalTile::new(array_tile_layout);

        let smem_slot_size = tile_size.m * tile_size.n;
        let smem_slice_start = UNIT_POS_Y * smem_slot_size;

        let mut shared_memory =
            SharedMemory::new(config.num_planes() as usize * smem_slot_size as usize);
        let smem_slice = shared_memory.slice_mut(
            smem_slice_start as usize,
            (smem_slice_start + smem_slot_size) as usize,
        );

        let mut shared_memory_vt =
            SharedMemory::<V>::new(config.num_planes() as usize * smem_slot_size as usize);
        let smem_vt_slice = shared_memory_vt.slice_mut(
            smem_slice_start as usize,
            (smem_slice_start + smem_slot_size) as usize,
        );

        SoftmaxHybridFragment::<E, V> {
            fragment,
            fragment_vt,
            smem_slice,
            smem_vt_slice,
            local_tile,
            stride: tile_size.n,
        }
    }

    fn zero(&mut self) {
        cmma::fill(&self.fragment, E::from_int(0));
    }
}

#[cube]
impl<E: Float, V: Numeric> FragmentSoftmax<E> for SoftmaxHybridFragment<E, V> {
    type Layout = LocalTileLayout;
    type SoftmaxScore = cmma::Matrix<E>;
    type SoftmaxRowFormat = LocalTile<E>;
    type SoftmaxVal = cmma::Matrix<V>;

    fn rowwise_mut(&mut self) -> &mut Self::SoftmaxRowFormat {
        // f32 fragment → f32 SMEM → f32 local_tile (unchanged)
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
        // f32 local_tile → cast+store → V SMEM → V fragment (halves bandwidth)
        self.local_tile.store_cast_to(&mut self.smem_vt_slice);

        sync_cube();

        cmma::load_with_layout(
            &self.fragment_vt,
            &self.smem_vt_slice.to_slice(),
            self.stride,
            cmma::MatrixLayout::RowMajor,
        );
    }

    fn zero(&mut self) {
        self.zero();
    }

    fn set_combined_scale(&mut self, _scale: f32) {
        // No-op for non-INT8 attention
    }
}

#[cube]
impl<E: Float> FragmentAccumulator<E> for HybridFragment<E> {
    fn rowwise_scale(&mut self, val: &RowWise<E>) {
        let local_tile = self.rowwise_mut();
        local_tile.rowwise_scale(val);
        self.update_from_rowwise();
    }

    fn zero(&mut self) {
        self.zero();
    }
}
