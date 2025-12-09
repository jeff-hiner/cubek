#[cfg(feature = "matmul_tests_f16")]
mod f16_ty {
    use super::*;

    pub type TestEG = half::f16;
    pub type TestES = half::f16;
    pub type TestEA = half::f16;

    include!("tiling_scheme/tile.rs");
}

#[cfg(feature = "matmul_tests_f32")]
mod f32_ty {
    use super::*;

    pub type TestEG = f32;
    pub type TestES = f32;
    pub type TestEA = f32;

    include!("tiling_scheme/tile.rs");
}
