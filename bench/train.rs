#![recursion_limit = "512"]

use std::time::{Duration, Instant};

use burn::LearningRate;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
use burn::tensor::{Int, Tensor, TensorData};
use burn_autodiff::Autodiff;
use burn_dragon_hatchling::{BDH, BDHConfig, language_model_loss, wgpu::init_runtime};
use burn_wgpu::Wgpu;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

#[cfg(feature = "cuda")]
use burn_cuda::Cuda;

#[derive(Clone, Copy)]
struct TrainConfig {
    name: &'static str,
    batch: usize,
    block: usize,
}

const TRAIN_CONFIGS: &[TrainConfig] = &[
    TrainConfig {
        name: "b4_t64",
        batch: 4,
        block: 64,
    },
    TrainConfig {
        name: "b8_t128",
        batch: 8,
        block: 128,
    },
    TrainConfig {
        name: "b16_t256",
        batch: 16,
        block: 256,
    },
    TrainConfig {
        name: "b16_t512",
        batch: 8,
        block: 512,
    },
];

fn training_step_bench(c: &mut Criterion) {
    run_training_backend::<Autodiff<Wgpu<f32>>, _>(c, "wgpu", |device| init_runtime(device));

    #[cfg(feature = "cuda")]
    run_training_backend::<Autodiff<Cuda<f32>>, _>(c, "cuda", |_| {});
}

fn run_training_backend<B, Init>(c: &mut Criterion, backend_name: &'static str, init_backend: Init)
where
    B: AutodiffBackend + Clone + 'static,
    Init: Fn(&<B as BackendTrait>::Device),
{
    <B as BackendTrait>::seed(24);
    let device = <B as BackendTrait>::Device::default();
    init_backend(&device);

    let model_config = BDHConfig::default();
    let base_model = BDH::<B>::new(model_config.clone(), &device);
    let optimizer_config = AdamWConfig::new().with_weight_decay(0.1);
    let lr: LearningRate = 1e-3;

    let mut group = c.benchmark_group(format!("bdh_single_train_step/{backend_name}"));

    for cfg in TRAIN_CONFIGS {
        let token_count = cfg.batch * cfg.block;
        let input_tokens: Vec<i64> = (0..token_count).map(|idx| (idx % 255) as i64).collect();
        let target_tokens: Vec<i64> = input_tokens.iter().map(|tok| (*tok + 1) % 255).collect();

        let inputs = Tensor::<B, 2, Int>::from_data(
            TensorData::new(input_tokens, [cfg.batch, cfg.block]),
            &device,
        );
        let targets = Tensor::<B, 2, Int>::from_data(
            TensorData::new(target_tokens, [cfg.batch, cfg.block]),
            &device,
        );

        // Warm-up pass to avoid counting shader compilation and graph building.
        {
            let model = base_model.clone();
            let mut optimizer = optimizer_config.clone().init::<B, BDH<B>>();
            let logits = model.forward(inputs.clone());
            let loss = language_model_loss::<B>(logits, targets.clone());
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            let _ = optimizer.step(lr, model, grads);
        }

        log_theoretical_profile(&model_config, cfg);

        group.throughput(Throughput::Elements(cfg.batch as u64));
        group.bench_with_input(BenchmarkId::from_parameter(cfg.name), cfg, |b, _| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;

                for _ in 0..iters {
                    let model = base_model.clone();
                    let mut optimizer = optimizer_config.clone().init::<B, BDH<B>>();

                    let start = Instant::now();
                    let logits = model.forward(inputs.clone());
                    let loss = language_model_loss::<B>(logits, targets.clone());
                    let grads = loss.backward();
                    let grads = GradientsParams::from_grads(grads, &model);
                    let _ = optimizer.step(lr, model, grads);
                    total += start.elapsed();
                }

                total
            });
        });
    }

    group.finish();
}

fn log_theoretical_profile(config: &BDHConfig, cfg: &TrainConfig) {
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
        "[train:{name}] approx forward GFLOPs: total={total_gflops:.2}, encoder={enc:.2}, \
         attn_scores={scores:.2}, attn_value={value:.2}, decoder={dec:.2} (backward ~2x forward)",
        name = cfg.name,
        total_gflops = total as f64 / 1e9,
        enc = encoder_matmul as f64 / 1e9,
        scores = attn_scores as f64 / 1e9,
        value = attn_value as f64 / 1e9,
        dec = decoder_matmul as f64 / 1e9,
    );
}

fn compute_latent_per_head(config: &BDHConfig) -> usize {
    config.latent_per_head()
}

fn compute_latent_total(config: &BDHConfig) -> usize {
    config.latent_total()
}

criterion_group!(benches, training_step_bench);
criterion_main!(benches);
