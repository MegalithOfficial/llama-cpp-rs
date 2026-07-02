use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams, LlamaContextType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use std::num::NonZeroU32;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let target_path = &args[1];
    let draft_path = &args[2];
    let n_ctx: u32 = args
        .get(3)
        .map(|v| v.parse().expect("n_ctx"))
        .unwrap_or(16128);
    let draft_batch: u32 = args
        .get(4)
        .map(|v| v.parse().expect("draft_batch"))
        .unwrap_or(5);
    let target_gpu_layers: u32 = args
        .get(5)
        .map(|v| v.parse().expect("target_gpu_layers"))
        .unwrap_or(0);
    let draft_gpu_layers: u32 = args
        .get(6)
        .map(|v| v.parse().expect("draft_gpu_layers"))
        .unwrap_or(0);
    let draft_op_offload: bool = args
        .get(7)
        .map(|v| v.parse().expect("draft_op_offload"))
        .unwrap_or(true);

    let backend = LlamaBackend::init().expect("backend");
    let target_params = LlamaModelParams::default().with_n_gpu_layers(target_gpu_layers);
    let target =
        LlamaModel::load_from_file(&backend, target_path, &target_params).expect("target model");
    let draft_params_m = LlamaModelParams::default().with_n_gpu_layers(draft_gpu_layers);
    let draft_model =
        LlamaModel::load_from_file(&backend, draft_path, &draft_params_m).expect("draft model");

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(512)
        .with_offload_kqv(false)
        .with_type_k(KvCacheType::Q8_0)
        .with_type_v(KvCacheType::Q8_0);
    let target_ctx = target.new_context(&backend, ctx_params).expect("target ctx");
    eprintln!("=== PROBE: target context created (n_ctx={n_ctx}) ===");

    let mut draft_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(draft_batch)
        .with_n_ubatch(draft_batch)
        .with_n_outputs_max(draft_batch)
        .with_offload_kqv(false)
        .with_type_k(KvCacheType::Q8_0)
        .with_type_v(KvCacheType::Q8_0)
        .with_ctx_type(LlamaContextType::Mtp)
        .with_ctx_other(&target_ctx)
        .with_op_offload(draft_op_offload)
        .with_n_rs_seq(0);
    if std::env::var("PROBE_NO_OP_OFFLOAD").is_ok() {
        draft_params = draft_params.with_op_offload(false);
    }
    let _draft_ctx = draft_model
        .new_context(&backend, draft_params)
        .expect("draft ctx");
    eprintln!("=== PROBE: draft context created OK (n_batch={draft_batch}) ===");
}
