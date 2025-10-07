#![recursion_limit = "512"]

use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Int, Tensor, TensorData};
use burn_dragon_hatchling::{BDH, BDHConfig, wgpu::init_runtime};
use burn_wgpu::Wgpu;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

#[derive(Clone, Copy)]
struct InferenceConfig {
    name: &'static str,
    batch: usize,
    block: usize,
}

const STORAGE_BUFFER_LIMIT_BYTES: u64 = 1 << 30; // 1 GiB, wgpu storage buffer cap on most drivers
const FLOAT_BYTES: u64 = std::mem::size_of::<f32>() as u64;

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

fn inference_bench(c: &mut Criterion) {
    type Backend = Wgpu<f32>;
    <Backend as BackendTrait>::seed(42);
    let device = <Backend as BackendTrait>::Device::default();
    init_runtime(&device);

    let model_config = BDHConfig::default();

    for cfg in INFERENCE_CONFIGS {
        let estimated_bytes = estimated_query_tensor_bytes(&model_config, cfg);
        if estimated_bytes > STORAGE_BUFFER_LIMIT_BYTES as u128 {
            log_theoretical_profile(&model_config, cfg);
            let requested_gib = estimated_bytes as f64 / (1024.0_f64 * 1024.0_f64 * 1024.0_f64);
            let limit_gib =
                STORAGE_BUFFER_LIMIT_BYTES as f64 / (1024.0_f64 * 1024.0_f64 * 1024.0_f64);
            println!(
                "[inference:{name}] skipping: query tensor needs {requested_gib:.2} GiB, \
                 wgpu storage buffers support up to {limit_gib:.2} GiB. \
                 Reduce batch/sequence length or the MLP multiplier to run this case.",
                name = cfg.name
            );
            continue;
        }

        let model = BDH::<Backend>::new(model_config.clone(), &device);
        let token_count = cfg.batch * cfg.block;
        let tokens: Vec<i64> = (0..token_count).map(|idx| (idx % 255) as i64).collect();
        let input = Tensor::<Backend, 2, Int>::from_data(
            TensorData::new(tokens, [cfg.batch, cfg.block]),
            &device,
        );

        // Warm-up: ensures shader compilation and graph building are not part of the timed runs.
        let _ = model.forward(input.clone()).into_data();

        log_theoretical_profile(&model_config, cfg);

        c.bench_with_input(
            BenchmarkId::new("bdh_inference_forward", cfg.name),
            cfg,
            |b, _| {
                b.iter(|| {
                    let _ = model.forward(input.clone());
                });
            },
        );
    }
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
        "[inference:{name}] approx GFLOPs per forward: total={total_gflops:.2}, encoder={enc:.2}, attn_scores={scores:.2}, attn_value={value:.2}, decoder={dec:.2}",
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

criterion_group!(benches, inference_bench);
criterion_main!(benches);

fn estimated_query_tensor_bytes(config: &BDHConfig, cfg: &InferenceConfig) -> u128 {
    let batch = cfg.batch as u128;
    let time = cfg.block as u128;
    let heads = config.n_head as u128;
    let latent_per_head = compute_latent_per_head(config) as u128;

    batch * heads * time * latent_per_head * (FLOAT_BYTES as u128)
}
