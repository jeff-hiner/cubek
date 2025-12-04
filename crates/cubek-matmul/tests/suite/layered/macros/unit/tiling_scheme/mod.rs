mod partition;
mod stage;
mod tile;

#[macro_export]
macro_rules! testgen_matmul_unit_tiling_scheme {
    ($algorithm: ty, $precision: ty) => {
        use cubek_matmul::components::TilingScheme;

        $crate::testgen_matmul_unit_tile!($algorithm, $precision, TilingScheme::builder());
    };
}
