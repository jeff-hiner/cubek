use cubecl;
use cubecl::prelude::*;

use crate::components::stage::{KeyTile, ValueTile, PartitionAttentionConfig, StageAttentionConfig};
use crate::components::tile::TileAttention;
use crate::definition::AttentionPrecision;

/// Key and Value tile partitions for attention computation.
///
/// Key and Value are always stored separately since they may have different types
/// (e.g., i8 for Key and f16 for Value in INT8 CMMA mode).
///
/// For each `kv`:
/// - Key: iterate over one column of `head_dim`, multiplying each (hd, kv) tile with all `seq_q` tiles.
/// - Value: then iterate over one row of `val_dim`, multiplying each (kv, vd) tile with all `seq_q` tiles.
#[derive(CubeType)]
pub struct KeyValuePartition<AP: AttentionPrecision, TA: TileAttention<AP>> {
    keys: KeySequence<AP, TA>,
    values: ValueSequence<AP, TA>,
}

#[derive(CubeType)]
pub struct KeySequence<AP: AttentionPrecision, TA: TileAttention<AP>> {
    sequence: Sequence<KeyTile<AP, TA>>,
}

#[derive(CubeType)]
pub struct ValueSequence<AP: AttentionPrecision, TA: TileAttention<AP>> {
    sequence: Sequence<ValueTile<AP, TA>>,
}

#[cube]
impl<AP: AttentionPrecision, TA: TileAttention<AP>> KeyValuePartition<AP, TA> {
    pub fn new(
        #[comptime] config: PartitionAttentionConfig<TA::Config>,
    ) -> KeyValuePartition<AP, TA> {
        let mut keys = Sequence::new();
        let mut values = Sequence::new();

        keys.push(KeyTile::new(config.tile_config()));
        values.push(ValueTile::new(config.tile_config()));

        KeyValuePartition::<AP, TA> {
            keys: KeySequence::<AP, TA> { sequence: keys },
            values: ValueSequence::<AP, TA> { sequence: values },
        }
    }

    pub fn get_key(&self) -> &KeyTile<AP, TA> {
        &self.keys.sequence[0]
    }

    pub fn get_key_mut(&mut self) -> &mut KeyTile<AP, TA> {
        self.keys.sequence.index_mut(0usize)
    }

    pub fn get_value(&self) -> &ValueTile<AP, TA> {
        &self.values.sequence[0]
    }

    pub fn get_value_mut(&mut self) -> &mut ValueTile<AP, TA> {
        self.values.sequence.index_mut(0usize)
    }
}
