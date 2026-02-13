pub(crate) mod launcher;

mod int8_quant;
mod int8_vs_f16_comparison;
mod reference;
mod utils;

pub(crate) use reference::assert_result;
pub(crate) use utils::tiling_scheme_ops;

mod unit {
    use cubek_attention::{
        definition::{AttentionBlueprint, AttentionProblem, AttentionTileSize},
        launch::{BlueprintStrategy, Strategy},
    };
    fn strategy(blueprint: AttentionBlueprint) -> Strategy {
        Strategy::Unit(BlueprintStrategy::Forced(blueprint))
    }

    fn tile_size() -> AttentionTileSize {
        AttentionTileSize {
            seq_q: 4,
            seq_kv: 4,
            head_dim: 4,
            val_dim: 4,
        }
    }

    fn minimal_seq_q_stage() -> u32 {
        32
    }

    /// For non-INT8 routines, use the actual head_dim for softmax scaling.
    #[allow(dead_code)]
    fn original_head_dim_for_blueprint(problem: &AttentionProblem) -> u32 {
        problem.dims.head_dim as u32
    }

    mod f16_ty {
        use super::*;
        use cubecl::frontend::CubePrimitive;
        use cubek_attention::definition::AttentionGlobalTypes;

        fn global_dtypes() -> AttentionGlobalTypes {
            AttentionGlobalTypes::from_single_dtype(half::f16::as_type_native_unchecked())
        }

        include!("tests.rs");
    }

    mod f32_ty {
        use super::*;
        use cubecl::frontend::CubePrimitive;
        use cubek_attention::definition::AttentionGlobalTypes;

        fn global_dtypes() -> AttentionGlobalTypes {
            AttentionGlobalTypes::from_single_dtype(f32::as_type_native_unchecked())
        }

        include!("tests.rs");
    }
}

mod blackbox_accelerated {
    use cubek_attention::{
        definition::{AttentionBlueprint, AttentionProblem, AttentionTileSize},
        launch::{BlueprintStrategy, Strategy},
    };

    fn strategy(blueprint: AttentionBlueprint) -> Strategy {
        Strategy::BlackboxAccelerated(BlueprintStrategy::Forced(blueprint))
    }

    fn tile_size() -> AttentionTileSize {
        #[cfg(target_os = "macos")]
        {
            use cubek_attention::definition::AttentionTileSize;

            // Metal uses 8×8×8 CMMA
            AttentionTileSize {
                seq_q: 8,
                seq_kv: 8,
                head_dim: 8,
                val_dim: 8,
            }
        }

        #[cfg(not(target_os = "macos"))]
        // Intel Arc/Vulkan uses 8×16×16 for f16×f16→f32 CMMA
        AttentionTileSize {
            seq_q: 8,
            seq_kv: 16,
            head_dim: 16,
            val_dim: 16,
        }
    }

    fn minimal_seq_q_stage() -> u32 {
        1
    }

    /// For non-INT8 routines, use the actual head_dim for softmax scaling.
    #[allow(dead_code)]
    fn original_head_dim_for_blueprint(problem: &AttentionProblem) -> u32 {
        problem.dims.head_dim as u32
    }

    mod f16_ty {
        use super::*;
        use cubecl::frontend::CubePrimitive;
        use cubek_attention::definition::AttentionGlobalTypes;

        fn global_dtypes() -> AttentionGlobalTypes {
            AttentionGlobalTypes::from_single_dtype(half::f16::as_type_native_unchecked())
        }

        include!("tests.rs");
    }

    mod f32_ty {
        use super::*;
        use cubecl::frontend::CubePrimitive;
        use cubek_attention::definition::AttentionGlobalTypes;

        fn global_dtypes() -> AttentionGlobalTypes {
            AttentionGlobalTypes::from_single_dtype(f32::as_type_native_unchecked())
        }

        include!("tests.rs");
    }
}

/// INT8 CMMA tests - validates quantized attention implementation.
/// Uses i8×i8→i32 CMMA for Q·K^T, f16×f16→f32 for P×V.
mod int8_cmma {
    use cubek_attention::{
        definition::{AttentionBlueprint, AttentionProblem, AttentionTileSize},
        launch::{BlueprintStrategy, Strategy},
    };

    fn strategy(blueprint: AttentionBlueprint) -> Strategy {
        Strategy::Int8Cmma(BlueprintStrategy::Forced(blueprint))
    }

    /// INT8 CMMA tile sizes for i8×i8→i32 CMMA.
    /// Intel Arc/Vulkan supports (M=8, N=16, K=32) for i8×i8→i32 CMMA.
    /// P×V uses f16×f16→f32 CMMA which also supports (8, 16, 16).
    fn tile_size() -> AttentionTileSize {
        AttentionTileSize {
            seq_q: 8,     // M dimension (must be 8 for Intel Arc i8 CMMA)
            seq_kv: 16,   // N dimension
            head_dim: 32, // K dimension for Q·K^T
            val_dim: 16,  // Same as seq_kv for P×V
        }
    }

    fn minimal_seq_q_stage() -> u32 {
        1
    }

    /// For INT8 CMMA, sm_scale (1/sqrt(head_dim) * log2(e)) is baked into Q quantization.
    /// Set original_head_dim=1 so tile_softmax applies scale=1/sqrt(1)=1.0, avoiding double-scaling.
    #[allow(dead_code)]
    fn original_head_dim_for_blueprint(_problem: &AttentionProblem) -> u32 {
        1
    }

    mod f16_ty {
        use super::*;
        use cubecl::frontend::CubePrimitive;
        use cubek_attention::definition::AttentionGlobalTypes;

        fn global_dtypes() -> AttentionGlobalTypes {
            AttentionGlobalTypes::from_single_dtype(half::f16::as_type_native_unchecked())
        }

        include!("tests.rs");
    }
}
