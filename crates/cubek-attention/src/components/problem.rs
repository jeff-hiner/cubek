use crate::components::{AttentionIdent, AttentionLineSizes, AvailableLineSizes};
use cubecl::{
    Runtime,
    client::ComputeClient,
    ir::{ElemType, FloatKind, StorageType},
};

#[derive(Clone, Debug)]
/// Description of an attention problem to solve, regardless of actual data
pub struct AttentionProblem {
    /// Batch size
    pub batch: usize,
    /// Number of attention heads
    pub num_heads: usize,

    /// Query sequence length
    pub seq_q: usize,
    /// Key/Value sequence length
    pub seq_kv: usize,
    /// Dimension of each head (d)
    pub head_dim: usize,
    /// Dimension of each value vector.  
    /// Usually equal to `head_dim`, but may differ in some variants
    pub val_dim: usize,

    /// Whether a mask is supplied (shape is always [batch, seq_q, heads, seq_kv])
    pub masked: bool,
    /// Whether there is a causal mask
    pub causal: bool,

    pub line_sizes: AttentionLineSizes,
    pub global_dtypes: AttentionStorageTypes,
    pub accumulator_precision: AccumulatorPrecision,
}

#[derive(Clone, Debug)]
/// Description of an attention problem to solve, regardless of actual data
pub struct AttentionProblemDims {
    /// Batch size
    pub batch: usize,
    /// Number of attention heads
    pub num_heads: usize,

    /// Query sequence length
    pub seq_q: usize,
    /// Key/Value sequence length
    pub seq_kv: usize,
    /// Dimension of each head (d)
    pub head_dim: usize,
    /// Dimension of each value vector.  
    /// Usually equal to `head_dim`, but may differ in some variants
    pub val_dim: usize,
}

impl AttentionProblem {
    pub fn new<R: Runtime>(
        client: &ComputeClient<R>,
        dims: AttentionProblemDims,
        masked: bool,
        causal: bool,
        global_dtypes: AttentionStorageTypes,
        accumulator_precision: AccumulatorPrecision,
    ) -> Self {
        let line_sizes = AvailableLineSizes::from_global_types::<R>(client, global_dtypes.clone())
            .filter(
                |ls| dims.head_dim % *ls as usize == 0,
                AttentionIdent::Query,
            )
            .filter(|ls| dims.head_dim % *ls as usize == 0, AttentionIdent::Key)
            .filter(|ls| dims.val_dim % *ls as usize == 0, AttentionIdent::Value)
            // Lined mask not always supported
            .filter(|ls| *ls == 1, AttentionIdent::Mask)
            .filter(|ls| dims.val_dim % *ls as usize == 0, AttentionIdent::Out)
            .pick_max()
            .unwrap();

        Self {
            batch: dims.batch,
            num_heads: dims.num_heads,
            seq_q: dims.seq_q,
            seq_kv: dims.seq_kv,
            head_dim: dims.head_dim,
            val_dim: dims.val_dim,
            masked,
            causal,
            line_sizes,
            global_dtypes,
            accumulator_precision,
        }
    }

    pub fn shape(&self, ident: AttentionIdent) -> [usize; 4] {
        match ident {
            AttentionIdent::Query => [self.batch, self.num_heads, self.seq_q, self.head_dim],
            AttentionIdent::Key => [self.batch, self.num_heads, self.seq_kv, self.head_dim],
            AttentionIdent::Value => [self.batch, self.num_heads, self.seq_kv, self.val_dim],
            AttentionIdent::Mask => [self.batch, self.num_heads, self.seq_q, self.seq_kv],
            AttentionIdent::Out => [self.batch, self.num_heads, self.seq_q, self.val_dim],
            AttentionIdent::Softmax => unreachable!("Not a materialized tensor"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct AttentionStorageTypes {
    pub query: StorageType,
    pub key: StorageType,
    pub value: StorageType,
    pub mask: StorageType,
    pub out: StorageType,
}

impl AttentionStorageTypes {
    pub fn from_single_dtype(dtype: StorageType) -> AttentionStorageTypes {
        Self {
            query: dtype,
            key: dtype,
            value: dtype,
            mask: dtype,
            out: dtype,
        }
    }
}

#[derive(Clone, Debug)]
pub enum AccumulatorPrecision {
    Strict(StorageType),
    // Let algorithm decide
    Loose,
}

impl AccumulatorPrecision {
    pub fn default_accumulator_type() -> StorageType {
        StorageType::Scalar(ElemType::Float(FloatKind::F32))
    }
}

impl Default for AccumulatorPrecision {
    fn default() -> Self {
        Self::Strict(Self::default_accumulator_type())
    }
}
