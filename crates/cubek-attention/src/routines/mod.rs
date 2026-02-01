/// Accelerated but using shared memory for rowwise operations
pub mod blackbox_accelerated;
/// INT8 CMMA tensor core attention
pub mod int8_cmma;
/// SageAttention with INT8 quantization
pub mod sage;
/// Unit attention
pub mod unit;

mod base;

pub use base::*;
pub use int8_cmma::Int8CmmaRoutine;
pub use sage::SageRoutine;
