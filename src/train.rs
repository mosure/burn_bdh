use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use clap::Parser;

use burn::LearningRate;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Int, Tensor, TensorData};
use burn_autodiff::Autodiff;
use burn_dragon_hatchling::dataset::{ShakespeareDataset, ShakespeareSplit};
use burn_dragon_hatchling::{
    BDH, BDHConfig, GenerationConfig, language_model_loss, load_training_config,
};
use burn_ndarray::NdArray;

#[derive(Parser, Debug)]
#[command(author, version, about = "Train the Baby Dragon Hatchling model")]
struct Args {
    /// Additional configuration files applied in order (later files override earlier ones).
    #[arg(short = 'c', long = "config", value_name = "PATH")]
    config: Vec<PathBuf>,
}

type ADBackend = Autodiff<NdArray<f32>>;

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

    <ADBackend as BackendTrait>::seed(1337);
    let device = <ADBackend as BackendTrait>::Device::default();

    let training = &config.training;
    let optimizer_cfg = &config.optimizer;

    let dataset = ShakespeareDataset::new(
        &config.dataset.cache_dir,
        training.block_size,
        training.batch_size,
        config.dataset.train_split_ratio,
    )
    .with_context(|| "failed to prepare Shakespeare dataset")?;

    let mut model_config = BDHConfig::default();
    if let Some(n_layer) = config.model.n_layer {
        model_config.n_layer = n_layer;
    }
    if let Some(n_embd) = config.model.n_embd {
        model_config.n_embd = n_embd;
    }
    if let Some(n_head) = config.model.n_head {
        model_config.n_head = n_head;
    }
    if let Some(multiplier) = config.model.mlp_internal_dim_multiplier {
        model_config.mlp_internal_dim_multiplier = multiplier;
    }
    if let Some(dropout) = config.model.dropout {
        model_config.dropout = dropout;
    }

    let mut model = BDH::<ADBackend>::new(model_config, &device);
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(optimizer_cfg.weight_decay)
        .init::<ADBackend, BDH<ADBackend>>();
    let lr: LearningRate = optimizer_cfg.learning_rate;

    let mut loss_acc = 0.0f32;
    let mut loss_steps = 0usize;

    for step in 0..training.max_iters {
        let (inputs, targets) = dataset.sample_batch::<ADBackend>(ShakespeareSplit::Train, &device);

        let logits = model.forward(inputs);
        let loss = language_model_loss::<ADBackend>(logits, targets);
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
            let avg_loss = loss_acc / loss_steps as f32;
            let val_loss = compute_validation_loss(&model, &dataset, &device)?;
            println!(
                "step {step}/{max_iters} train_loss {avg_loss:.3} val_loss {val_loss:.3}",
                max_iters = training.max_iters
            );
            loss_acc = 0.0;
            loss_steps = 0;
        }
    }

    println!("Training complete. Generating sample...");
    generate_sample(
        &model,
        &dataset,
        &device,
        training.block_size,
        &config.generation,
    )?;

    Ok(())
}

fn compute_validation_loss(
    model: &BDH<ADBackend>,
    dataset: &ShakespeareDataset,
    device: &<ADBackend as BackendTrait>::Device,
) -> Result<f32> {
    if (dataset.train_split_ratio() - 1.0).abs() < f32::EPSILON {
        // No validation split available.
        return Ok(f32::NAN);
    }

    let (val_inputs, val_targets) =
        dataset.sample_batch::<ADBackend>(ShakespeareSplit::Val, device);
    let val_loss = language_model_loss::<ADBackend>(model.forward(val_inputs), val_targets)
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(|err| anyhow!("{err:?}"))?[0];
    Ok(val_loss)
}

fn generate_sample(
    model: &BDH<ADBackend>,
    dataset: &ShakespeareDataset,
    device: &<ADBackend as BackendTrait>::Device,
    block_size: usize,
    generation: &GenerationConfig,
) -> Result<()> {
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
    let prompt = Tensor::<ADBackend, 2, Int>::from_data(
        TensorData::new(prompt_bytes, [1, prompt_len]),
        device,
    );

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
