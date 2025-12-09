mod matmul_unit {

    #[cfg(feature = "matmul_tests_unit")]
    mod unit {
        use super::*;

        include!("algorithm.rs");
    }
}
