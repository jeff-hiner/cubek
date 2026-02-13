#![allow(clippy::explicit_counter_loop)]
#![allow(clippy::manual_is_multiple_of)]

/// Components for matrix multiplication
pub mod components;
pub mod definition;
/// Standalone kernels for pre-processing (quantization, etc.)
pub mod kernels;
pub mod launch;
/// Contains attention kernels
pub mod routines;
