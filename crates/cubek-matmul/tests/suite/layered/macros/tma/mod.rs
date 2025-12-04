mod algorithm;
mod precision;
mod tiling_scheme;

#[macro_export]
macro_rules! testgen_matmul_tma {
    () => {
        mod matmul_tma {
            use cubek_matmul::components::tile::io::Filled;
            type TMM = cubek_matmul::components::tile::cmma::CmmaMatmul<Filled>;

            #[cfg(all(feature = "matmul_tests_tma", not(feature = "matmul_tests_mma")))]
            $crate::testgen_matmul_tma_algorithm!();

            #[cfg(all(feature = "matmul_tests_tma", feature = "matmul_tests_mma"))]
            mod cmma {
                use super::*;
                type TMM = cubek_matmul::components::tile::cmma::CmmaMatmul<Filled>;

                $crate::testgen_matmul_tma_algorithm!();
            }

            #[cfg(all(feature = "matmul_tests_tma", feature = "matmul_tests_mma"))]
            mod mma {
                use super::*;
                type TMM = cubek_matmul::components::tile::mma::MmaMatmul;

                $crate::testgen_matmul_tma_algorithm!();
            }
        }
    };
}
