use cubecl;
use cubecl::prelude::*;

use crate::components::tile::TileAttention;
use crate::definition::AttentionPrecision;
use crate::definition::attention_types::{QG, QT};
use cubek_matmul::components::tile::StridedTile;

#[derive(CubeType)]
/// Query input to the Tile Attention
pub struct QueryTile<AP: AttentionPrecision, TA: TileAttention<AP>> {
    /// CMMA fragment for accelerated matmul
    pub fragment: TA::Query,
    /// Scalar storage for INT8 path - stores query data in row-major format
    /// Shape: (seq_q, head_dim), indexed as [row * head_dim + col]
    pub scalar: Array<QT<AP>>,
}

#[cube]
impl<AP: AttentionPrecision, TA: TileAttention<AP>> QueryTile<AP, TA> {
    pub fn new(#[comptime] tile_size: (u32, u32), #[comptime] config: TA::Config) -> QueryTile<AP, TA> {
        let (seq_q, head_dim) = tile_size;
        QueryTile::<AP, TA> {
            fragment: TA::allocate_query(config),
            scalar: Array::new((seq_q * head_dim) as usize),
        }
    }

    /// Loads the query data into both fragment (CMMA) and scalar storage
    pub fn update(&mut self, tile: &StridedTile<QG<AP>>, #[comptime] tile_size: (u32, u32)) {
        // Load into CMMA fragment
        TA::load_query(tile, &mut self.fragment);

        // Also store in scalar format for INT8 path
        // tile is (seq_q, head_dim) in row-major
        let (seq_q, head_dim) = tile_size;
        let line_size = tile.line_size;
        let col_iterations = comptime!(head_dim / line_size);

        for row in 0..seq_q {
            for col_line in 0..col_iterations {
                let line = tile.get_line(row, col_line);
                #[unroll]
                for i in 0..line_size {
                    let idx = row * head_dim + col_line * line_size + i;
                    self.scalar[idx as usize] = QT::<AP>::cast_from(line[i as usize]);
                }
            }
        }
    }

    /// Get a slice view of the scalar storage
    pub fn scalar_slice(&self) -> Slice<QT<AP>> {
        self.scalar.to_slice()
    }
}
