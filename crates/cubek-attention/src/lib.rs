#![allow(clippy::explicit_counter_loop)]
#![allow(clippy::manual_is_multiple_of)]

mod base;
/// Components for matrix multiplication
pub mod components;
/// Contains attention kernels
pub mod kernels;
pub use base::*;
