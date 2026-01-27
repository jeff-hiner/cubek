//! Standalone attention kernels
//!
//! These kernels bypass the complex tiling architecture for simpler
//! integration and experimentation.

pub mod sage_attention;

pub use sage_attention::*;
