#![recursion_limit = "512"]

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use clap::{Parser, ValueEnum};

use burn::LearningRate;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
use burn::tensor::{Int, Tensor, TensorData};
use burn_autodiff::Autodiff;
use burn_dragon_hatchling::dataset::{ShakespeareDataset, ShakespeareSplit};
use burn_dragon_hatchling::{
    BDH, BDHConfig, GenerationConfig, ModelOverrides, TrainingConfig, language_model_loss,
    load_training_config, wgpu::init_runtime,
};
use burn_wgpu::Wgpu;

#[cfg(feature = "cuda")]
use burn_cuda::Cuda;

#[derive(Parser, Debug)]
#[command(author, version, about = "Train the Baby Dragon Hatchling model")]
struct Args {
    /// Additional configuration files applied in order (later files override earlier ones).
    #[arg(short = 'c', long = "config", value_name = "PATH")]
    config: Vec<PathBuf>,
    /// Backend to use for training.
    #[arg(long, value_enum, default_value_t = BackendArg::Wgpu)]
    backend: BackendArg,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum BackendArg {
    Wgpu,
    Cuda,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = Args::parse();

    let mut config_paths = vec![PathBuf::from("config/base.toml")];
    config_paths.extend(args.config);
    let config = load_training_config(&config_paths)?;

    let training = &config.training;
    let dataset = ShakespeareDataset::new(
        &config.dataset.cache_dir,
        training.block_size,
        training.batch_size,
        config.dataset.train_split_ratio,
    )
    .with_context(|| "failed to prepare Shakespeare dataset")?;

    match args.backend {
        BackendArg::Wgpu => {
            train_backend::<Autodiff<Wgpu<f32>>, _>(&config, &dataset, "wgpu", |device| {
                init_runtime(device)
            })
        }
        BackendArg::Cuda => {
            #[cfg(feature = "cuda")]
            {
                train_backend::<Autodiff<Cuda<f32>>, _>(&config, &dataset, "cuda", |_| {})
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(anyhow!(
                    "cuda backend selected but this build lacks `cuda` feature; rebuild with `--features cuda`"
                ))
            }
        }
    }
}

fn train_backend<B, Init>(
    config: &TrainingConfig,
    dataset: &ShakespeareDataset,
    backend_name: &str,
    init_backend: Init,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
    Init: Fn(&<B as BackendTrait>::Device),
{
    <B as BackendTrait>::seed(1337);
    let device = <B as BackendTrait>::Device::default();
    init_backend(&device);

    let training = &config.training;
    let optimizer_cfg = &config.optimizer;

    let model_config = build_model_config(&config.model);
    let mut model = BDH::<B>::new(model_config, &device);
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(optimizer_cfg.weight_decay)
        .init::<B, BDH<B>>();
    let lr: LearningRate = optimizer_cfg.learning_rate;

    let mut loss_acc = 0.0f32;
    let mut loss_steps = 0usize;
    let mut samples_since_log = 0usize;
    let mut throughput_timer = Instant::now();

    for step in 0..training.max_iters {
        let (inputs, targets) = dataset.sample_batch::<B>(ShakespeareSplit::Train, &device);
        samples_since_log += training.batch_size;

        let logits = model.forward(inputs);
        let loss = language_model_loss::<B>(logits, targets);
        let loss_value = loss
            .clone()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .map_err(|err| anyhow!("{err:?}"))?[0];

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(lr, model, grads);

        loss_acc += loss_value;
        loss_steps += 1;

        if training.log_frequency > 0 && step % training.log_frequency == 0 {
            let avg_loss = if loss_steps > 0 {
                loss_acc / loss_steps as f32
            } else {
                0.0
            };
            let val_loss = compute_validation_loss::<B>(&model, dataset, &device)?;
            let elapsed = throughput_timer.elapsed();
            let samples_per_s = if elapsed.as_secs_f64() > 0.0 {
                samples_since_log as f64 / elapsed.as_secs_f64()
            } else {
                f64::NAN
            };
            println!(
                "step {step}/{max_iters} [{backend}] train_loss {avg_loss:.3} val_loss {val_loss:.3} samples_per_s {samples_per_s:.2}",
                max_iters = training.max_iters,
                backend = backend_name,
                samples_per_s = samples_per_s
            );
            loss_acc = 0.0;
            loss_steps = 0;
            samples_since_log = 0;
            throughput_timer = Instant::now();
        }
    }

    println!("Training complete on {backend_name}. Generating sample...");
    generate_sample::<B>(
        &model,
        dataset,
        &device,
        training.block_size,
        &config.generation,
    )?;

    Ok(())
}

fn build_model_config(overrides: &ModelOverrides) -> BDHConfig {
    let mut model_config = BDHConfig::default();

    if let Some(n_layer) = overrides.n_layer {
        model_config.n_layer = n_layer;
    }
    if let Some(n_embd) = overrides.n_embd {
        model_config.n_embd = n_embd;
    }
    if let Some(n_head) = overrides.n_head {
        model_config.n_head = n_head;
    }
    if let Some(multiplier) = overrides.mlp_internal_dim_multiplier {
        model_config.mlp_internal_dim_multiplier = multiplier;
    }
    if let Some(dropout) = overrides.dropout {
        model_config.dropout = dropout;
    }
    if let Some(enabled) = overrides.fused_kernels {
        model_config.fused_kernels.enabled = enabled;
    }
    if let Some(block) = overrides.block_size {
        model_config.fused_kernels.set_block_sizes(block, block);
    }
    if let Some(use_alibi) = overrides.use_alibi {
        model_config.fused_kernels.set_use_alibi(use_alibi);
        if !use_alibi {
            model_config
                .fused_kernels
                .set_alibi_slopes(vec![0.0; model_config.n_head]);
        }
    }

    model_config
}

fn compute_validation_loss<B>(
    model: &BDH<B>,
    dataset: &ShakespeareDataset,
    device: &<B as BackendTrait>::Device,
) -> Result<f32>
where
    B: BackendTrait + Clone + 'static,
{
    if (dataset.train_split_ratio() - 1.0).abs() < f32::EPSILON {
        // No validation split available.
        return Ok(f32::NAN);
    }

    let (val_inputs, val_targets) = dataset.sample_batch::<B>(ShakespeareSplit::Val, device);
    let val_loss = language_model_loss::<B>(model.forward(val_inputs), val_targets)
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(|err| anyhow!("{err:?}"))?[0];
    Ok(val_loss)
}

fn generate_sample<B>(
    model: &BDH<B>,
    dataset: &ShakespeareDataset,
    device: &<B as BackendTrait>::Device,
    block_size: usize,
    generation: &GenerationConfig,
) -> Result<()>
where
    B: BackendTrait + Clone + 'static,
{
    let mut prompt_bytes: Vec<i64> = generation
        .prompt
        .as_bytes()
        .iter()
        .map(|byte| *byte as i64)
        .collect();
    if prompt_bytes.len() > block_size {
        prompt_bytes = prompt_bytes[prompt_bytes.len() - block_size..].to_vec();
    }
    let prompt_len = prompt_bytes.len();
    let prompt =
        Tensor::<B, 2, Int>::from_data(TensorData::new(prompt_bytes, [1, prompt_len]), device);

    let generated = model.generate(
        prompt,
        generation.max_tokens,
        generation.temperature,
        generation.top_k,
    );
    let generated_tokens = generated
        .into_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .map_err(|err| anyhow!("{err:?}"))?;
    let text = dataset.decode(&generated_tokens);
    println!("{text}");
    Ok(())
}
