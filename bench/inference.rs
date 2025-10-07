#![recursion_limit = "512"]

use burn::tensor::backend::Backend as BackendTrait;
use burn_dragon_hatchling::{
    prefill_state, sample_next_token, BDH, BDHConfig, wgpu::init_runtime,
};
use burn_wgpu::Wgpu;
use criterion::{black_box, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

#[cfg(feature = "cuda")]
use burn_cuda::Cuda;

#[derive(Clone, Copy)]
struct InferenceConfig {
    name: &'static str,
    batch: usize,
    block: usize,
}

const INFERENCE_CONFIGS: &[InferenceConfig] = &[
    InferenceConfig {
        name: "b8_t128",
        batch: 8,
        block: 128,
    },
    InferenceConfig {
        name: "b16_t512",
        batch: 16,
        block: 512,
    },
    InferenceConfig {
        name: "b16_t1024",
        batch: 16,
        block: 1024,
    },
];

const STORAGE_BUFFER_LIMIT_BYTES: u64 = 1 << 30; // ~1 GiB limit on most wgpu drivers
const FLOAT_BYTES: u64 = std::mem::size_of::<f32>() as u64;

fn inference_bench(c: &mut Criterion) {
    run_inference_backend::<Wgpu<f32>, _, _>(
        c,
        "wgpu",
        |device| init_runtime(device),
        |config, cfg| skip_reason_wgpu(config, cfg),
    );

    #[cfg(feature = "cuda")]
    run_inference_backend::<Cuda<f32>, _, _>(c, "cuda", |_| {}, |_, _| None);
}

fn run_inference_backend<B, Init, Skip>(
    c: &mut Criterion,
    backend_name: &'static str,
    init_backend: Init,
    skip_reason: Skip,
) where
    B: BackendTrait + Clone + 'static,
    Init: Fn(&<B as BackendTrait>::Device),
    Skip: Fn(&BDHConfig, &InferenceConfig) -> Option<String>,
{
    <B as BackendTrait>::seed(42);
    let device = <B as BackendTrait>::Device::default();
    init_backend(&device);

    let model_config = BDHConfig::default();
    let mut group = c.benchmark_group(format!("bdh_inference_forward/{backend_name}"));

    for cfg in INFERENCE_CONFIGS {
        if let Some(reason) = skip_reason(&model_config, cfg) {
            log_theoretical_profile(&model_config, cfg);
            println!(
                "[inference:{name}@{backend}] skipping: {reason}",
                name = cfg.name,
                backend = backend_name
            );
            continue;
        }

        let model = BDH::<B>::new(model_config.clone(), &device);
        let prompt_tokens: Vec<i64> = (0..cfg.block).map(|idx| (idx % 255) as i64).collect();
        let (base_state, base_logits) =
            prefill_state(&model, &prompt_tokens, &device).expect("prefill state");

        // Warm-up ensures shader compilation / graph construction is amortized.
        let mut warm_state = base_state.clone();
        let mut warm_logits = base_logits.clone();
        for _ in 0..cfg.batch {
            let (_, next_logits) = sample_next_token(
                &model,
                &mut warm_state,
                warm_logits,
                1.0,
                None,
                &device,
            )
            .expect("warm-up token");
            warm_logits = next_logits;
        }

        log_theoretical_profile(&model_config, cfg);

        group.throughput(Throughput::Elements(cfg.batch as u64));
        group.bench_with_input(BenchmarkId::from_parameter(cfg.name), cfg, |b, _| {
            b.iter(|| {
                let mut state = base_state.clone();
                let mut logits = base_logits.clone();
                for _ in 0..cfg.batch {
                    let (token, next_logits) = sample_next_token(
                        &model,
                        &mut state,
                        logits,
                        1.0,
                        None,
                        &device,
                    )
                    .expect("sample token");
                    logits = next_logits;
                    black_box(token);
                }
            });
        });
    }

    group.finish();
}

fn skip_reason_wgpu(config: &BDHConfig, cfg: &InferenceConfig) -> Option<String> {
    let estimated_bytes = estimated_query_tensor_bytes(config, cfg);
    if estimated_bytes > STORAGE_BUFFER_LIMIT_BYTES as u128 {
        let requested_gib = estimated_bytes as f64 / (1024.0_f64.powi(3));
        let limit_gib = STORAGE_BUFFER_LIMIT_BYTES as f64 / (1024.0_f64.powi(3));
        return Some(format!(
            "query tensor needs {requested_gib:.2} GiB but wgpu storage buffers cap at \
             {limit_gib:.2} GiB; reduce batch/sequence length or the MLP multiplier."
        ));
    }
    None
}

fn log_theoretical_profile(config: &BDHConfig, cfg: &InferenceConfig) {
    let batch = cfg.batch as u64;
    let time = cfg.block as u64;
    let embed = config.n_embd as u64;
    let latent_per_head = compute_latent_per_head(config) as u64;
    let latent_total = compute_latent_total(config) as u64;
    let heads = config.n_head as u64;
    let bt = batch * time;

    let encoder_matmul = 2 * bt * embed * latent_total;
    let attn_scores = 2 * batch * heads * time * time * latent_per_head;
    let attn_value = 2 * batch * heads * time * time * embed;
    let decoder_matmul = 2 * bt * latent_total * embed;
    let total = encoder_matmul + attn_scores + attn_value + decoder_matmul;

    println!(
        "[inference:{name}] approx GFLOPs per forward: total={total_gflops:.2}, encoder={enc:.2}, \
         attn_scores={scores:.2}, attn_value={value:.2}, decoder={dec:.2}",
        name = cfg.name,
        total_gflops = total as f64 / 1e9,
        enc = encoder_matmul as f64 / 1e9,
        scores = attn_scores as f64 / 1e9,
        value = attn_value as f64 / 1e9,
        dec = decoder_matmul as f64 / 1e9,
    );
}

fn compute_latent_per_head(config: &BDHConfig) -> usize {
    (config.mlp_internal_dim_multiplier * config.n_embd) / config.n_head
}

fn compute_latent_total(config: &BDHConfig) -> usize {
    compute_latent_per_head(config) * config.n_head
}

fn estimated_query_tensor_bytes(config: &BDHConfig, cfg: &InferenceConfig) -> u128 {
    let batch = cfg.batch as u128;
    let time = cfg.block as u128;
    let heads = config.n_head as u128;
    let latent_per_head = compute_latent_per_head(config) as u128;

    batch * heads * time * latent_per_head * (FLOAT_BYTES as u128)
}

criterion_group!(benches, inference_bench);
criterion_main!(benches);
