#![allow(missing_docs)]

pub mod layered;
pub mod naive;
pub mod test_utils;

mod unit {
    crate::testgen_matmul_unit!();
}
mod tma {
    crate::testgen_matmul_tma!();
}
mod plane_vecmat {
    crate::testgen_matmul_plane_vecmat!();
}

mod plane_accelerated {
    crate::testgen_matmul_plane_accelerated!();
}

mod test_naive {
    crate::testgen_matmul_simple!();
}
