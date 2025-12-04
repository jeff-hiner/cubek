#[macro_export]
macro_rules! testgen_matmul_launch {
    (Normal, $algorithm: ty, $precision: ty, $selection: expr, $problem: expr) => {
        use cubecl::prelude::*;
        use $crate::suite::layered::matmul_test_launcher::test_matmul_algorithm;

        #[test]
        pub fn test() {
            let client = cubecl::TestRuntime::client(&Default::default());
            test_matmul_algorithm::<$algorithm, $precision, cubecl::TestRuntime>(
                client, $problem, $selection,
            );
        }
    };

    (Tma, $algorithm: ty, $precision: ty, $selection: expr, $problem: expr) => {
        use cubecl::prelude::*;
        use $crate::suite::layered::tma_test_launcher::test_tma_matmul_algorithm;

        #[test]
        pub fn test() {
            let client = cubecl::TestRuntime::client(&Default::default());
            test_tma_matmul_algorithm::<$algorithm, $precision, cubecl::TestRuntime>(
                client, $problem, $selection,
            );
        }
    };
}
