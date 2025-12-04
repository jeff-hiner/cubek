#[macro_export]
macro_rules! testgen_matmul_tma_algorithm {
    () => {
        mod simple_tma {
            use super::*;
            use cubek_matmul::kernels::layered::simple::SimpleTmaAlgorithm;

            $crate::testgen_matmul_tma_precision!(SimpleTmaAlgorithm<TMM>);
        }

        #[cfg(all(feature = "matmul_tests_double"))]
        mod double_buffering_tma {
            use super::*;
            use cubek_matmul::kernels::layered::double_buffering::TmaDoubleBufferingAlgorithm;

            $crate::testgen_matmul_tma_precision!(TmaDoubleBufferingAlgorithm<TMM>);
        }

        #[cfg(all(feature = "matmul_tests_double"))]
        mod specialized_tma {
            use super::*;
            use cubek_matmul::kernels::layered::specialized::SpecializedAlgorithm;

            $crate::testgen_matmul_tma_precision!(SpecializedAlgorithm<TMM>);
        }
    };
}
