mod algorithm;
mod precision;
mod tiling_scheme;

#[macro_export]
macro_rules! testgen_matmul_plane_accelerated {
    () => {
        mod matmul_plane_accelerated {
            use cubek_matmul::components::tile::io::Filled;
            type TMM = cubek_matmul::components::tile::cmma::CmmaMatmul<Filled>;

            #[cfg(all(feature = "matmul_tests_plane", not(feature = "matmul_tests_mma")))]
            $crate::testgen_matmul_plane_accelerated_algorithm!();

            #[cfg(all(feature = "matmul_tests_plane", feature = "matmul_tests_mma"))]
            mod cmma {
                use super::*;
                type TMM = cubek_matmul::components::tile::cmma::CmmaMatmul<Filled>;

                $crate::testgen_matmul_plane_accelerated_algorithm!();
            }

            #[cfg(all(feature = "matmul_tests_plane", feature = "matmul_tests_mma"))]
            mod mma {
                type TMM = cubek_matmul::components::tile::mma::MmaMatmul;

                $crate::testgen_matmul_plane_accelerated_algorithm!();
            }
        }
    };
}
