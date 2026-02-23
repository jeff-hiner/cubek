pub(crate) mod launcher;

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

    /// Use the actual head_dim for softmax scaling.
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
        // Vulkan can generally use 8×16×16 for f16×f16→f32 CMMA
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

    /// Use the actual head_dim for softmax scaling.
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
