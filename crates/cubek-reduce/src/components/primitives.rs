use cubecl::prelude::*;

/// Warp-level sum reduction using shuffle operations.
///
/// All lanes get the sum of all 32 values in the warp using butterfly reduction:
/// ```ignored
/// Step 1 (offset=16): Lane 0 ← Lane 0 + Lane 16, Lane 1 ← Lane 1 + Lane 17, ...
/// Step 2 (offset=8):  Lane 0 ← Lane 0 + Lane 8,  Lane 1 ← Lane 1 + Lane 9,  ...
/// Step 3 (offset=4):  Lane 0 ← Lane 0 + Lane 4,  Lane 1 ← Lane 1 + Lane 5,  ...
/// Step 4 (offset=2):  Lane 0 ← Lane 0 + Lane 2,  Lane 1 ← Lane 1 + Lane 3,  ...
/// Step 5 (offset=1):  Lane 0 ← Lane 0 + Lane 1,  ...
/// ```
///
/// # Performance
/// - ~5 cycles per shuffle × 5 steps = ~25 cycles total
/// - Compare to shared memory: ~110 (write) + ~110 (read) × log2(32) = ~1100+ cycles
///
/// # Example
/// ```ignored
/// #[cube]
/// fn warp_sum_example(value: f32) -> f32 {
///     reduce_sum_shuffle(value)  // All lanes get the sum
/// }
/// ```
#[cube]
pub fn reduce_sum_shuffle<F: Float>(value: F) -> F {
    // Manually unrolled butterfly reduction
    let v1 = value + plane_shuffle_xor(value, 16);
    let v2 = v1 + plane_shuffle_xor(v1, 8);
    let v3 = v2 + plane_shuffle_xor(v2, 4);
    let v4 = v3 + plane_shuffle_xor(v3, 2);
    v4 + plane_shuffle_xor(v4, 1)
}

/// Warp-level max reduction using shuffle operations.
/// All lanes get the maximum of all 32 values in the warp.
#[cube]
pub fn reduce_max_shuffle<F: Float>(value: F) -> F {
    let v1 = F::max(value, plane_shuffle_xor(value, 16));
    let v2 = F::max(v1, plane_shuffle_xor(v1, 8));
    let v3 = F::max(v2, plane_shuffle_xor(v2, 4));
    let v4 = F::max(v3, plane_shuffle_xor(v3, 2));
    F::max(v4, plane_shuffle_xor(v4, 1))
}

/// Warp-level min reduction using shuffle operations.
/// All lanes get the minimum of all 32 values in the warp.
#[cube]
pub fn reduce_min_shuffle<F: Float>(value: F) -> F {
    let v1 = F::min(value, plane_shuffle_xor(value, 16));
    let v2 = F::min(v1, plane_shuffle_xor(v1, 8));
    let v3 = F::min(v2, plane_shuffle_xor(v2, 4));
    let v4 = F::min(v3, plane_shuffle_xor(v3, 2));
    F::min(v4, plane_shuffle_xor(v4, 1))
}

/// Warp-level product reduction using shuffle operations.
/// All lanes get the product of all 32 values in the warp.
#[cube]
pub fn reduce_prod_shuffle<F: Float>(value: F) -> F {
    let v1 = value * plane_shuffle_xor(value, 16);
    let v2 = v1 * plane_shuffle_xor(v1, 8);
    let v3 = v2 * plane_shuffle_xor(v2, 4);
    let v4 = v3 * plane_shuffle_xor(v3, 2);
    v4 * plane_shuffle_xor(v4, 1)
}
