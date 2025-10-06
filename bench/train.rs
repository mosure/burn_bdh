#![recursion_limit = "512"]

use burn::LearningRate;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Int, Tensor, TensorData};
use burn_autodiff::Autodiff;
use burn_dragon_hatchling::{BDH, BDHConfig, language_model_loss, wgpu::init_runtime};
use burn_wgpu::Wgpu;
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};

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
];

fn training_step_bench(c: &mut Criterion) {
    type Backend = Autodiff<Wgpu<f32>>;
    <Backend as BackendTrait>::seed(24);
    let device = <Backend as BackendTrait>::Device::default();
    init_runtime(&device);

    let model_config = BDHConfig::default();
    let base_model = BDH::<Backend>::new(model_config.clone(), &device);
    let optimizer_config = AdamWConfig::new().with_weight_decay(0.1);
    let lr: LearningRate = 1e-3;

    for cfg in TRAIN_CONFIGS {
        let token_count = cfg.batch * cfg.block;
        let input_tokens: Vec<i64> = (0..token_count).map(|idx| (idx % 255) as i64).collect();
        let target_tokens: Vec<i64> = input_tokens.iter().map(|tok| (*tok + 1) % 255).collect();

        let inputs = Tensor::<Backend, 2, Int>::from_data(
            TensorData::new(input_tokens, [cfg.batch, cfg.block]),
            &device,
        );
        let targets = Tensor::<Backend, 2, Int>::from_data(
            TensorData::new(target_tokens, [cfg.batch, cfg.block]),
            &device,
        );

        // Warm-up pass so shader compilation and graph building do not skew measurements.
        {
            let model = base_model.clone();
            let mut optimizer = optimizer_config.clone().init::<Backend, BDH<Backend>>();
            let logits = model.forward(inputs.clone());
            let loss = language_model_loss::<Backend>(logits, targets.clone());
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            let _ = optimizer.step(lr, model, grads);
        }

        log_theoretical_profile(&model_config, cfg);

        c.bench_with_input(
            BenchmarkId::new("bdh_single_train_step", cfg.name),
            cfg,
            |b, _| {
                b.iter_batched(
                    || {
                        let model = base_model.clone();
                        let optimizer = optimizer_config.clone().init::<Backend, BDH<Backend>>();
                        (model, optimizer)
                    },
                    |(model, mut optimizer)| {
                        let logits = model.forward(inputs.clone());
                        let loss = language_model_loss::<Backend>(logits, targets.clone());
                        let grads = loss.backward();
                        let grads = GradientsParams::from_grads(grads, &model);
                        let _ = optimizer.step(lr, model, grads);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
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
        "[train:{name}] approx forward GFLOPs: total={total_gflops:.2}, encoder={enc:.2}, attn_scores={scores:.2}, attn_value={value:.2}, decoder={dec:.2} (backward ~2x forward)",
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

criterion_group!(benches, training_step_bench);
criterion_main!(benches);
