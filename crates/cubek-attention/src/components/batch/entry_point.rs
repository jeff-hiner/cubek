use crate::{
    components::batch::{BatchAttentionConfig as _, BatchAttentionFamily, base::BatchAttention},
    components::global::GlobalAttentionConfig as _,
    components::stage::StageAttentionConfig as _,
    definition::{AttentionBlueprint, AttentionElems, CubeCountInput},
    launch::{AttentionArgs, TensorKey, TensorMask, TensorOutput, TensorQuery, TensorValue},
};
use cubecl::{
    self,
    prelude::*,
    std::{CubeOption, CubeOptionExpand, tensor::r#virtual::VirtualTensor},
};

type Input<Args, QG, KG, VG, MSK> = <Args as AttentionArgs>::Input<QG, KG, VG, MSK>;
type Output<Args, OG> = <Args as AttentionArgs>::Output<OG>;

#[cube(launch, launch_unchecked)]
/// Launches the attention kernel
pub(crate) fn attention<
    Args: AttentionArgs,
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
    BMMF: BatchAttentionFamily,
>(
    inputs: &Input<Args, QG, KG, VG, MSK>,
    output: &mut Output<Args, OG>,
    cube_count_args: CubeCountInput,
    #[comptime] blueprint: AttentionBlueprint,
    #[define(QG, QT, KG, KS, VG, VS, KT, VT, SACC, SM, ACC, MSK, OG, OS)] elem_types: [StorageType;
        14],
) {
    let config = comptime!(BMMF::expand_config(
        blueprint,
        &AttentionElems::from_define_array(elem_types)
    ));
    if config.is_err() {
        push_validation_error(config.err().unwrap().to_string());
        comptime!(return);
    }
    let config = config.unwrap();

    let mut state = Args::init_state(inputs, output);

    let query = TensorQuery::<QG, KG, VG, MSK, OG, Args>::new(&state);
    let query = VirtualTensor::<QG>::new::<TensorQuery<QG, KG, VG, MSK, OG, Args>>(&query);

    let key = TensorKey::<QG, KG, VG, MSK, OG, Args>::new(&state);
    let key = VirtualTensor::<KG>::new::<TensorKey<QG, KG, VG, MSK, OG, Args>>(&key);

    let value = TensorValue::<QG, KG, VG, MSK, OG, Args>::new(&state);
    let value = VirtualTensor::<VG>::new::<TensorValue<QG, KG, VG, MSK, OG, Args>>(&value);

    let has_mask = Args::has_mask(&state);
    let mask: CubeOption<VirtualTensor<MSK>> = match has_mask {
        CubeOption::Some(_) => {
            let mask = TensorMask::<QG, KG, VG, MSK, OG, Args>::new(&state);
            let mask = VirtualTensor::<MSK>::new::<TensorMask<QG, KG, VG, MSK, OG, Args>>(&mask);
            CubeOption::new_Some(mask)
        }
        CubeOption::None => CubeOption::new_None(),
    };

    let mut out = TensorOutput::<QG, KG, VG, MSK, OG, Args>::new(&mut state);
    let out =
        VirtualTensor::<OG, ReadWrite>::new::<TensorOutput<QG, KG, VG, MSK, OG, Args>>(&mut out);

    // Create scale reader for quantization scales.
    // CUBE_POS_Y encodes the flattened batch*num_heads index.
    // CUBE_POS_X encodes the Q block index.
    // For non-INT8 attention, this creates a uniform 1.0 scale reader.
    let batch_head_idx = CUBE_POS_Y;
    let q_block_idx = CUBE_POS_X;

    // Compute number of blocks per head for scale tensor indexing.
    // Block size matches the stage partition sizes used in the attention loop.
    let global_config = config.global_config();
    let stage_config = global_config.stage_config();
    let q_block_size = stage_config.elements_in_stage_seq_q();
    let k_block_size = stage_config.elements_in_partition_seq_kv();
    let seq_q = Args::shape_query::<QG, KG, VG, MSK, OG>(&state, 2) as u32;
    let seq_kv = Args::shape_key::<QG, KG, VG, MSK, OG>(&state, 2) as u32;
    let num_q_blocks_per_head = seq_q.div_ceil(q_block_size);
    let num_k_blocks_per_head = seq_kv.div_ceil(k_block_size);

    let scale_reader = Args::create_scale_reader::<QG, KG, VG, MSK, OG>(
        &state,
        batch_head_idx,
        q_block_idx,
        num_q_blocks_per_head,
        num_k_blocks_per_head,
    );

    BMMF::Attention::<(QG, QT, KG, KS, VG, VS, KT, VT, SACC, SM, ACC, MSK, OG, OS)>::execute(
        query,
        key,
        value,
        mask,
        out,
        cube_count_args,
        scale_reader,
        config,
    );
}
