#![allow(missing_docs)]

#[macro_export]
macro_rules! testgen_matmul_simple {
    () => {
        mod simple {
            $crate::testgen_matmul_simple!(f32);
        }
    };
    ($float:ident) => {
            use crate::suite::naive;

            pub type FloatT = $float;
            type TestRuntime = cubecl::TestRuntime;

            #[test]
            pub fn test_small() {
                naive::tests::test_small::<TestRuntime, FloatT>(
                    &Default::default(),
                )
            }

            #[test]
            pub fn test_odd() {
                naive::tests::test_odd::<TestRuntime, FloatT>(
                    &Default::default(),
                )
            }

            #[test]
            pub fn test_simple_matmul_large() {
                naive::tests::test_large::<TestRuntime, FloatT>(
                    &Default::default(),
                )
            }

            #[test]
            pub fn test_with_check_bounds() {
                naive::tests::test_with_check_bounds::<
                    TestRuntime,
                    FloatT,
                >(&Default::default())
            }

            #[test]
            pub fn test_with_batches() {
                naive::tests::test_with_batches::<
                    TestRuntime,
                    FloatT,
                >(&Default::default())
            }
    };
    ([$($float:ident),*]) => {
        mod simple {
            use super::*;
            ::paste::paste! {
                $(mod [<$float _ty>] {
                    use super::*;

                    $crate::testgen_matmul_simple!($float);
                })*
            }
        }
    };
}
