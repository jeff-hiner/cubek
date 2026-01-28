use cubecl;
use cubecl::prelude::*;

use crate::components::tile::TileAttention;
use crate::definition::AttentionPrecision;

/// Key tile fragment for Q·K^T matmul.
#[derive(CubeType)]
pub struct KeyTile<AP: AttentionPrecision, TA: TileAttention<AP>> {
    pub fragment: TA::Key,
}

#[cube]
impl<AP: AttentionPrecision, TA: TileAttention<AP>> KeyTile<AP, TA> {
    pub fn new(#[comptime] config: TA::Config) -> Self {
        KeyTile::<AP, TA> {
            fragment: TA::allocate_key(config),
        }
    }

    /// Get the underlying key fragment as readable
    pub fn key(&self) -> &TA::Key {
        &self.fragment
    }

    /// Get the underlying key fragment as writable
    pub fn key_mut(&mut self) -> &mut TA::Key {
        &mut self.fragment
    }
}

/// Value tile fragment for P×V matmul.
#[derive(CubeType)]
pub struct ValueTile<AP: AttentionPrecision, TA: TileAttention<AP>> {
    pub fragment: TA::Value,
}

#[cube]
impl<AP: AttentionPrecision, TA: TileAttention<AP>> ValueTile<AP, TA> {
    pub fn new(#[comptime] config: TA::Config) -> Self {
        ValueTile::<AP, TA> {
            fragment: TA::allocate_value(config),
        }
    }

    /// Get the underlying value fragment as readable
    pub fn value(&self) -> &TA::Value {
        &self.fragment
    }

    /// Get the underlying value fragment as writable
    pub fn value_mut(&mut self) -> &mut TA::Value {
        &mut self.fragment
    }
}
