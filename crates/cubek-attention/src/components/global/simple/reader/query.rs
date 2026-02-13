use cubecl;
use cubecl::prelude::*;
use cubecl::std::Swizzle;
use cubecl::std::tensor::{View, layout::Coords2d};
use cubek_matmul::components::global::memory::GlobalMemoryConfig;
use cubek_matmul::components::tile::StridedTile;

use crate::components::stage::AttentionPartitioner;
use crate::definition::attention_types::QG;
use crate::definition::{AttentionPrecision, AttentionTileSize};

#[derive(CubeType)]
pub struct QueryReader<AP: AttentionPrecision> {
    query: View<Line<QG<AP>>, Coords2d>,
    #[cube(comptime)]
    gmem_config: GlobalMemoryConfig,
}

#[cube]
impl<AP: AttentionPrecision> QueryReader<AP> {
    pub fn new(
        stage_q_offset: u32,
        query: View<Line<QG<AP>>, Coords2d>,
        #[comptime] gmem_config: GlobalMemoryConfig,
    ) -> Self {
        let query = query.slice((stage_q_offset, 0), query.shape());

        QueryReader::<AP> { query, gmem_config }
    }

    pub fn get_tile<P: AttentionPartitioner>(
        &self,
        tile: Coords2d,
        #[comptime] attention_tile_size: AttentionTileSize,
        #[comptime] partition_seq_q: u32,
        #[comptime] _partition_head_dim: u32,
    ) -> StridedTile<QG<AP>> {
        let (row_in_partition, col) = tile;

        let row = row_in_partition + P::seq_q_index() * partition_seq_q;

        let line_size = self.gmem_config.line_size.comptime() as u32;

        let tile_head_dim = attention_tile_size.head_dim;

        // Get the View's actual column count for stride calculation.
        let full_head_dim = self.query.shape().1;
        let stride = full_head_dim / line_size;

        // CRITICAL FIX: Slice the full row width, not just tile_head_dim.
        // The to_linear_slice() returns a slice sized for CONTIGUOUS data.
        // With strided memory layout, we need a larger slice that spans:
        // from row 0 to row (seq_q-1), with full_head_dim stride between rows.
        // Slicing to (seq_q, tile_head_dim) creates a slice of size seq_q * tile_head_dim,
        // but strided access needs (seq_q-1) * full_head_dim + tile_head_dim elements.
        let tile_row_start = row * attention_tile_size.seq_q;
        let slice = self
            .query
            .slice(
                (tile_row_start, 0u32),
                (attention_tile_size.seq_q.runtime(), full_head_dim),
            )
            .to_linear_slice();

        // Start at the column offset for this tile
        let start = col * tile_head_dim / line_size;
        // End covers the last row's tile data
        let end = (attention_tile_size.seq_q - 1) * stride + (col + 1) * tile_head_dim / line_size;

        StridedTile::<QG<AP>>::new_strided(
            slice,
            start,
            end,
            stride,
            Swizzle::none(),
            self.gmem_config.matrix_layout,
            line_size,
        )
    }
}
