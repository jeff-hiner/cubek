use crate::{
    definition::{AccumulatorPrecision, AttentionGlobalTypes},
    launch::{AttentionArgs, TensorArgs},
};
use cubecl::prelude::*;
use half::{bf16, f16};

/// Attention spec defining each element types used in the computation as well as
/// how the arguments are passed to the kernel.
pub trait AttentionSpec: Send + Sync + Clone + 'static {
    type Precision: AttentionPrecision;
    /// How the input and output tensors are passed as arguments.
    type Args: AttentionArgs;
}

impl<AP: AttentionPrecision, Args: AttentionArgs> AttentionSpec for (AP, Args) {
    type Precision = AP;
    type Args = Args;
}

// A simple default for TensorArgs
impl<AP: AttentionPrecision> AttentionSpec for AP {
    type Precision = AP;
    type Args = TensorArgs;
}

pub trait QueryPrecision: Send + Sync + Copy + 'static {
    type Global: Numeric;
    type Tile: Numeric;
}

/// Precision for staged matrices (Key, Value) that may be quantized (e.g., INT8).
/// For INT8 SageAttention, Global is still f32 (loaded from f32 tensors), but Stage can be i8.
pub trait StagedMatrixPrecision: Send + Sync + Copy + 'static {
    type Global: Numeric;
    type Stage: Numeric;
}

/// Precision for output matrices. Always Float since attention output is floating-point.
pub trait OutputPrecision: Send + Sync + Copy + 'static {
    type Global: Float;
    type Stage: Float;
}

pub trait AttentionPrecision: Send + Sync + Copy + 'static {
    type Query: QueryPrecision;
    type Key: StagedMatrixPrecision;
    type Value: StagedMatrixPrecision;
    /// The element type used for Key tiles in CMMA Q·K^T.
    /// For float attention: same as K stage type.
    /// For INT8 attention: i8.
    type KeyTile: Numeric;
    /// The element type used for Value tiles in CMMA P×V.
    /// For float attention: same as V stage type.
    /// For INT8 attention: f16 (V stays float, not quantized).
    type ValueTile: Numeric;
    /// The CMMA accumulator type for Q·K^T score computation.
    /// For float attention: same as Softmax (e.g., f32).
    /// For INT8 CMMA: i32 (then converted to Softmax type after dequantization).
    type ScoreAccumulator: Numeric;
    type Softmax: Float;
    type Accumulator: Float;
    type Mask: Numeric;
    type Out: OutputPrecision;

    /// Whether score_matmul needs to convert from ScoreAccumulator to Softmax type.
    /// For float attention: false (both are f32). For INT8 CMMA: true (i32 → f32).
    const REQUIRES_SCORE_CONVERSION: bool;
}

impl QueryPrecision for f16 {
    type Global = f16;
    type Tile = f16;
}

impl QueryPrecision for bf16 {
    type Global = bf16;
    type Tile = bf16;
}

impl QueryPrecision for flex32 {
    type Global = f32;
    type Tile = f16;
}

impl QueryPrecision for f32 {
    type Global = f32;
    type Tile = f32;
}

impl QueryPrecision for f64 {
    type Global = f64;
    type Tile = f32;
}

impl<G: Numeric, T: Numeric> QueryPrecision for (G, T) {
    type Global = G;
    type Tile = T;
}

impl StagedMatrixPrecision for f16 {
    type Global = f16;
    type Stage = f16;
}

impl StagedMatrixPrecision for bf16 {
    type Global = bf16;
    type Stage = bf16;
}

impl StagedMatrixPrecision for flex32 {
    type Global = f32;
    type Stage = f16;
}

impl StagedMatrixPrecision for f32 {
    type Global = f32;
    type Stage = f32;
}

impl StagedMatrixPrecision for f64 {
    type Global = f64;
    type Stage = f32;
}

impl<G: Numeric, S: Numeric> StagedMatrixPrecision for (G, S) {
    type Global = G;
    type Stage = S;
}

impl OutputPrecision for f16 {
    type Global = f16;
    type Stage = f16;
}

impl OutputPrecision for bf16 {
    type Global = bf16;
    type Stage = bf16;
}

impl OutputPrecision for flex32 {
    type Global = f32;
    type Stage = f16;
}

impl OutputPrecision for f32 {
    type Global = f32;
    type Stage = f32;
}

impl OutputPrecision for f64 {
    type Global = f64;
    type Stage = f32;
}

impl<G: Float, S: Float> OutputPrecision for (G, S) {
    type Global = G;
    type Stage = S;
}

impl AttentionPrecision for f16 {
    type Query = f16;
    type Key = f16;
    type Value = f16;
    type KeyTile = f16;
    type ValueTile = f16;
    #[cfg(target_os = "macos")]
    type ScoreAccumulator = f16;
    #[cfg(target_os = "macos")]
    type Softmax = f16;
    #[cfg(target_os = "macos")]
    type Accumulator = f16;
    #[cfg(not(target_os = "macos"))]
    type ScoreAccumulator = f32;
    #[cfg(not(target_os = "macos"))]
    type Softmax = f32;
    #[cfg(not(target_os = "macos"))]
    type Accumulator = f32;
    type Mask = u8;
    type Out = f16;
    const REQUIRES_SCORE_CONVERSION: bool = false;
}

impl AttentionPrecision for flex32 {
    type Query = flex32;
    type Key = flex32;
    type Value = flex32;
    type KeyTile = f16;
    type ValueTile = f16;
    #[cfg(target_os = "macos")]
    type ScoreAccumulator = f16;
    #[cfg(target_os = "macos")]
    type Softmax = f16;
    #[cfg(target_os = "macos")]
    type Accumulator = f16;
    #[cfg(not(target_os = "macos"))]
    type ScoreAccumulator = f32;
    #[cfg(not(target_os = "macos"))]
    type Softmax = f32;
    #[cfg(not(target_os = "macos"))]
    type Accumulator = f32;
    type Mask = u8;
    type Out = f32;
    const REQUIRES_SCORE_CONVERSION: bool = false;
}

impl AttentionPrecision for bf16 {
    type Query = bf16;
    type Key = bf16;
    type Value = bf16;
    type KeyTile = bf16;
    type ValueTile = bf16;
    #[cfg(target_os = "macos")]
    type ScoreAccumulator = bf16;
    #[cfg(target_os = "macos")]
    type Softmax = bf16;
    #[cfg(target_os = "macos")]
    type Accumulator = bf16;
    #[cfg(not(target_os = "macos"))]
    type ScoreAccumulator = f32;
    #[cfg(not(target_os = "macos"))]
    type Softmax = f32;
    #[cfg(not(target_os = "macos"))]
    type Accumulator = f32;
    type Mask = u8;
    type Out = bf16;
    const REQUIRES_SCORE_CONVERSION: bool = false;
}

impl AttentionPrecision for f32 {
    type Query = f32;
    type Key = f32;
    type Value = f32;
    type KeyTile = f32;
    type ValueTile = f32;
    type ScoreAccumulator = f32;
    type Softmax = f32;
    type Accumulator = f32;
    type Mask = u8;
    type Out = f32;
    const REQUIRES_SCORE_CONVERSION: bool = false;
}

impl AttentionPrecision for f64 {
    type Query = f64;
    type Key = f64;
    type Value = f64;
    type KeyTile = f32;
    type ValueTile = f32;
    type ScoreAccumulator = f32;
    type Softmax = f32;
    type Accumulator = f32;
    type Mask = u8;
    type Out = f64;
    const REQUIRES_SCORE_CONVERSION: bool = false;
}

/// Marker type for INT8 CMMA attention precision.
///
/// This precision type enables hardware-accelerated INT8 CMMA (Cooperative Matrix Multiply-Accumulate)
/// for the Q·K^T score computation in SageAttention:
/// - Q and K are quantized to i8 with per-row scale factors
/// - CMMA computes i8 × i8 → i32 accumulator
/// - Scores are dequantized to f32 for softmax
/// - V remains f32, accumulated with f32 precision
/// - Output is f32
///
/// This matches the parallelism of reference implementations like Triton's `tl.dot()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Int8Cmma;

impl AttentionPrecision for Int8Cmma {
    // Q: loaded as f32, quantized on-the-fly to i8 tiles
    type Query = (f32, i8);
    // K: loaded as f32, quantized on-the-fly to i8 stage/tiles
    type Key = (f32, i8);
    // V: loaded as f16, stays f16 for P×V CMMA
    type Value = (f16, f16);
    /// i8 for K tiles in CMMA Q·K^T
    type KeyTile = i8;
    /// f16 for V tiles in CMMA P×V (reference uses f16 for V, not i8)
    type ValueTile = f16;
    /// i32 accumulator for INT8 CMMA (i8 × i8 → i32)
    type ScoreAccumulator = i32;
    /// f32 for softmax computation (after dequantization from i32)
    type Softmax = f32;
    /// f32 for output accumulation
    type Accumulator = f32;
    type Mask = u8;
    /// Output is f32
    type Out = f32;
    /// INT8 CMMA needs i32 → f32 conversion after score matmul
    const REQUIRES_SCORE_CONVERSION: bool = true;
}

impl<
    QG: Numeric,
    QT: Numeric,
    KG: Numeric,
    KS: Numeric,
    VG: Numeric,
    VS: Numeric,
    KT: Numeric,
    VT: Numeric,
    SACC: Numeric,
    SM: Float,
    ACC: Float,
    MSK: Numeric,
    OG: Float,
    OS: Float,
> AttentionPrecision for (QG, QT, KG, KS, VG, VS, KT, VT, SACC, SM, ACC, MSK, OG, OS)
{
    type Query = (QG, QT);
    type Key = (KG, KS);
    type Value = (VG, VS);
    type KeyTile = KT;
    type ValueTile = VT;
    type ScoreAccumulator = SACC;
    type Softmax = SM;
    type Accumulator = ACC;
    type Mask = MSK;
    type Out = (OG, OS);
    // TODO: For INT8 CMMA via tuple types, this should be true when SACC != SM.
    // For now, use Int8Cmma type directly for INT8 CMMA attention.
    const REQUIRES_SCORE_CONVERSION: bool = false;
}

// Element indices for #[define(QG, QT, KG, KS, VG, VS, KT, VT, SACC, SM, ACC, MSK, OG, OS)]
// QG=0, QT=1, KG=2, KS=3, VG=4, VS=5, KT=6, VT=7, SACC=8, SM=9, ACC=10, MSK=11, OG=12, OS=13

/// Input argument
pub type InputArg<AA> = <AA as AttentionArgs>::Input<
    NumericExpand<0>,  // QG
    NumericExpand<2>,  // KG
    NumericExpand<4>,  // VG
    NumericExpand<11>, // MSK
>;

/// Output argument
pub type OutputArg<AA> = <AA as AttentionArgs>::Output<NumericExpand<12>>; // OG

/// Input runtime argument
pub type InputRuntimeArg<'a, AA, R> = <InputArg<AA> as LaunchArg>::RuntimeArg<'a, R>;

/// Output runtime argument
pub type OutputRuntimeArg<'a, AA, R> = <OutputArg<AA> as LaunchArg>::RuntimeArg<'a, R>;

pub mod attention_types {
    use crate::definition::{
        AttentionPrecision, AttentionSpec, OutputPrecision, QueryPrecision, StagedMatrixPrecision,
    };

    pub type QG<AS> =
        <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Query as QueryPrecision>::Global;
    pub type QT<AS> =
        <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Query as QueryPrecision>::Tile;
    pub type KG<AS> =
    <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Key as StagedMatrixPrecision>::Global;
    pub type KS<AS> =
        <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Key as StagedMatrixPrecision>::Stage;
    pub type VG<AS> =
    <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Value as StagedMatrixPrecision>::Global;
    pub type VS<AS> =
    <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Value as StagedMatrixPrecision>::Stage;

    /// Key tile type for Q·K^T CMMA. For float: f16. For INT8: i8.
    pub type KT<AS> = <<AS as AttentionSpec>::Precision as AttentionPrecision>::KeyTile;
    /// Value tile type for P×V CMMA. For float: f16. For INT8: f16 (V stays float).
    pub type VT<AS> = <<AS as AttentionSpec>::Precision as AttentionPrecision>::ValueTile;
    /// CMMA accumulator type for Q·K^T score computation.
    /// For float attention: same as SM. For INT8 CMMA: i32.
    pub type SACC<AS> = <<AS as AttentionSpec>::Precision as AttentionPrecision>::ScoreAccumulator;
    pub type SM<AS> = <<AS as AttentionSpec>::Precision as AttentionPrecision>::Softmax;
    pub type ACC<AS> = <<AS as AttentionSpec>::Precision as AttentionPrecision>::Accumulator;
    pub type MSK<AS> = <<AS as AttentionSpec>::Precision as AttentionPrecision>::Mask;

    pub type OG<AS> =
        <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Out as OutputPrecision>::Global;
    pub type OS<AS> =
        <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Out as OutputPrecision>::Stage;
}

pub type Args<MS> = <MS as AttentionSpec>::Args;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct AttentionElems {
    pub query_global: StorageType,
    pub query_tile: StorageType,
    pub key_global: StorageType,
    pub key_stage: StorageType,
    pub value_global: StorageType,
    pub value_stage: StorageType,
    /// Key tile type for Q·K^T CMMA. For float: f16. For INT8: i8.
    pub key_tile: StorageType,
    /// Value tile type for P×V CMMA. For float: f16. For INT8: f16 (V stays float).
    pub value_tile: StorageType,
    /// CMMA accumulator type for Q·K^T. For float: same as softmax. For INT8: i32.
    pub score_accumulator: StorageType,
    pub softmax: StorageType,
    pub accumulator: StorageType,
    pub mask: StorageType,
    pub out_global: StorageType,
    pub out_stage: StorageType,
}

impl AttentionElems {
    pub fn from_global_types(
        global_dtypes: &AttentionGlobalTypes,
        accumulator_precision: &AccumulatorPrecision,
    ) -> AttentionElems {
        let accumulator = match accumulator_precision {
            AccumulatorPrecision::Strict(storage_type) => *storage_type,
            AccumulatorPrecision::Loose => AccumulatorPrecision::default_accumulator_type(),
        };

        Self {
            query_global: global_dtypes.query,
            query_tile: global_dtypes.query,
            key_global: global_dtypes.key,
            key_stage: global_dtypes.key,
            value_global: global_dtypes.value,
            value_stage: global_dtypes.value,
            // For standard float attention, K and V tiles use the same type
            key_tile: global_dtypes.value,
            value_tile: global_dtypes.value,
            // For standard float attention, score_accumulator matches softmax type
            score_accumulator: accumulator,
            softmax: accumulator,
            accumulator,
            mask: global_dtypes.mask,
            out_global: global_dtypes.out,
            out_stage: global_dtypes.out,
        }
    }

    /// Create element types for INT8 CMMA attention (SageAttention-style).
    ///
    /// - Q and K are loaded as f16, quantized to i8 tiles for Q·K^T
    /// - CMMA computes i8 × i8 → i32 for Q·K^T
    /// - Scores are converted to f32 for softmax
    /// - V stays f16, P×V uses f16 × f16 → f32 CMMA
    /// - Output is f32
    pub fn for_int8_cmma(global_dtypes: &AttentionGlobalTypes) -> AttentionElems {
        use cubecl::ir::{ElemType, FloatKind, IntKind};
        let i8_type = StorageType::Scalar(ElemType::Int(IntKind::I8));
        let i32_type = StorageType::Scalar(ElemType::Int(IntKind::I32));
        let f16_type = StorageType::Scalar(ElemType::Float(FloatKind::F16));
        let f32_type = AccumulatorPrecision::default_accumulator_type();

        Self {
            // Q: loaded as f16, quantized to i8 for CMMA
            query_global: global_dtypes.query,
            query_tile: i8_type,
            // K: loaded as f16, quantized to i8 for CMMA
            key_global: global_dtypes.key,
            key_stage: i8_type,
            // V: loaded as f16, stays f16 for P×V CMMA
            value_global: global_dtypes.value,
            value_stage: f16_type,
            // K tiles are i8 for Q·K^T CMMA
            key_tile: i8_type,
            // V tiles are f16 for P×V CMMA (reference uses f16 for V, not i8)
            value_tile: f16_type,
            // INT8 CMMA uses i32 accumulator for Q·K^T
            score_accumulator: i32_type,
            // Softmax computed in f32 (after dequantization from i32)
            softmax: f32_type,
            // Output accumulation in f32
            accumulator: f32_type,
            mask: global_dtypes.mask,
            // Output is f32
            out_global: f32_type,
            out_stage: f32_type,
        }
    }

    pub fn from_define_array(elem_types: [StorageType; 14]) -> AttentionElems {
        AttentionElems {
            query_global: elem_types[0],
            query_tile: elem_types[1],
            key_global: elem_types[2],
            key_stage: elem_types[3],
            value_global: elem_types[4],
            value_stage: elem_types[5],
            key_tile: elem_types[6],
            value_tile: elem_types[7],
            score_accumulator: elem_types[8],
            softmax: elem_types[9],
            accumulator: elem_types[10],
            mask: elem_types[11],
            out_global: elem_types[12],
            out_stage: elem_types[13],
        }
    }
}

impl From<&AttentionElems> for [StorageType; 14] {
    fn from(elems: &AttentionElems) -> Self {
        [
            elems.query_global,
            elems.query_tile,
            elems.key_global,
            elems.key_stage,
            elems.value_global,
            elems.value_stage,
            elems.key_tile,
            elems.value_tile,
            elems.score_accumulator,
            elems.softmax,
            elems.accumulator,
            elems.mask,
            elems.out_global,
            elems.out_stage,
        ]
    }
}
