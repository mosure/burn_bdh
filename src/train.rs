#![recursion_limit = "512"]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use clap::{Parser, ValueEnum};

use burn::LearningRate;
use burn::data::dataloader::DataLoader;
use burn::lr_scheduler::{
    LrScheduler,
    cosine::{CosineAnnealingLrScheduler, CosineAnnealingLrSchedulerConfig},
    exponential::{ExponentialLrScheduler, ExponentialLrSchedulerConfig},
    linear::{LinearLrScheduler, LinearLrSchedulerConfig},
    noam::{NoamLrScheduler, NoamLrSchedulerConfig},
    step::{StepLrScheduler, StepLrSchedulerConfig},
};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamW, AdamWConfig};
use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
use burn::tensor::{Int, Tensor, TensorData};
use burn_autodiff::Autodiff;
use burn_train::metric::{Adaptor, ItemLazy, LearningRateMetric, LossInput, LossMetric};
use burn_train::{LearnerBuilder, TrainOutput, TrainStep, ValidStep};
use burn_wgpu::Wgpu;
use tracing::info;

#[cfg(feature = "cuda")]
use burn_cuda::Cuda;

use burn::record::{BinFileRecorder, FullPrecisionSettings};

use burn_dragon_hatchling::wgpu::init_runtime;
use burn_dragon_hatchling::{
    BDH, BDHConfig, DatasetConfig, GenerationConfig, LearningRateScheduleConfig, ModelOverrides,
    OptimizerConfig, ShakespeareBatch, ShakespeareDataset, ShakespeareRandomDataLoader,
    ShakespeareSplit, TrainingConfig, TrainingHyperparameters, language_model_loss,
    load_training_config,
};

#[derive(Parser, Debug)]
#[command(author, version, about = "Train the Baby Dragon Hatchling model")]
struct Args {
    /// Additional configuration files applied in order (later files override earlier ones).
    #[arg(short = 'c', long = "config", value_name = "PATH")]
    config: Vec<PathBuf>,
    /// Backend to use for training.
    #[arg(long, value_enum, default_value_t = BackendArg::Cuda)]
    backend: BackendArg,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum BackendArg {
    Cuda,
    Wgpu,
}

#[derive(Clone)]
struct LanguageModelOutput<B: BackendTrait> {
    loss: Tensor<B, 1>,
}

impl<B: BackendTrait> LanguageModelOutput<B> {
    fn new(loss: Tensor<B, 1>) -> Self {
        Self { loss }
    }
}

impl<B: BackendTrait> ItemLazy for LanguageModelOutput<B> {
    type ItemSync = Self;

    fn sync(self) -> Self::ItemSync {
        self
    }
}

impl<B: BackendTrait> Adaptor<LossInput<B>> for LanguageModelOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

struct LanguageModelTrainItem<B: AutodiffBackend> {
    loss: Tensor<B, 1>,
}

impl<B: AutodiffBackend> LanguageModelTrainItem<B> {
    fn new(loss: Tensor<B, 1>) -> Self {
        Self { loss }
    }
}

impl<B: AutodiffBackend> ItemLazy for LanguageModelTrainItem<B> {
    type ItemSync = LanguageModelOutput<B::InnerBackend>;

    fn sync(self) -> Self::ItemSync {
        LanguageModelOutput::new(self.loss.inner())
    }
}

type ValidBackend<B> = <B as AutodiffBackend>::InnerBackend;

impl<B: AutodiffBackend> TrainStep<ShakespeareBatch<B>, LanguageModelTrainItem<B>> for BDH<B> {
    fn step(&self, batch: ShakespeareBatch<B>) -> TrainOutput<LanguageModelTrainItem<B>> {
        let logits = self.forward(batch.inputs);
        let loss = language_model_loss::<B>(logits, batch.targets);
        let grads = loss.backward();

        TrainOutput::new(self, grads, LanguageModelTrainItem::new(loss))
    }
}

impl<B: BackendTrait> ValidStep<ShakespeareBatch<B>, LanguageModelOutput<B>> for BDH<B> {
    fn step(&self, batch: ShakespeareBatch<B>) -> LanguageModelOutput<B> {
        let logits = self.forward(batch.inputs);
        let loss = language_model_loss::<B>(logits, batch.targets);
        LanguageModelOutput::new(loss)
    }
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

    let dataset = Arc::new(prepare_dataset(&config.dataset, &config.training)?);

    match args.backend {
        BackendArg::Wgpu => train_backend::<Autodiff<Wgpu<f32>>, _>(
            &config,
            Arc::clone(&dataset),
            "wgpu",
            |device| init_runtime(device),
        ),
        BackendArg::Cuda => {
            #[cfg(feature = "cuda")]
            {
                train_backend::<Autodiff<Cuda<f32>>, _>(&config, dataset, "cuda", |_| {})
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
    dataset: Arc<ShakespeareDataset>,
    backend_name: &str,
    init_backend: Init,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    Init: Fn(&B::Device),
{
    B::seed(1337);
    let device = B::Device::default();
    init_backend(&device);

    let training = &config.training;
    let optimizer_cfg = &config.optimizer;

    let model_config = build_model_config(&config.model);

    let train_steps = training.max_iters.max(1);
    let valid_steps = usize::max(1, training.max_iters / training.log_frequency.max(1));

    let train_loader: Arc<dyn DataLoader<B, ShakespeareBatch<B>>> =
        Arc::new(ShakespeareRandomDataLoader::<B>::new(
            Arc::clone(&dataset),
            ShakespeareSplit::Train,
            &device,
            train_steps,
        ));

    let valid_device = device.clone();
    let valid_loader: Arc<dyn DataLoader<ValidBackend<B>, ShakespeareBatch<ValidBackend<B>>>> =
        Arc::new(ShakespeareRandomDataLoader::<ValidBackend<B>>::new(
            Arc::clone(&dataset),
            ShakespeareSplit::Val,
            &valid_device,
            valid_steps,
        ));

    let mut model = Some(BDH::<B>::new(model_config.clone(), &device));
    let mut optim = Some(
        AdamWConfig::new()
            .with_weight_decay(optimizer_cfg.weight_decay)
            .init::<B, BDH<B>>(),
    );
    let scheduler = resolve_lr_scheduler(optimizer_cfg, training, &model_config)?;

    let run_dir = PathBuf::from("runs").join(backend_name);
    let model = match scheduler {
        ResolvedLrScheduler::Constant(lr) => train_with_scheduler(
            &run_dir,
            backend_name,
            training,
            &model_config,
            &device,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            lr,
            Arc::clone(&train_loader),
            Arc::clone(&valid_loader),
        )?,
        ResolvedLrScheduler::Cosine(scheduler) => train_with_scheduler(
            &run_dir,
            backend_name,
            training,
            &model_config,
            &device,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            scheduler,
            Arc::clone(&train_loader),
            Arc::clone(&valid_loader),
        )?,
        ResolvedLrScheduler::Linear(scheduler) => train_with_scheduler(
            &run_dir,
            backend_name,
            training,
            &model_config,
            &device,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            scheduler,
            Arc::clone(&train_loader),
            Arc::clone(&valid_loader),
        )?,
        ResolvedLrScheduler::Exponential(scheduler) => train_with_scheduler(
            &run_dir,
            backend_name,
            training,
            &model_config,
            &device,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            scheduler,
            Arc::clone(&train_loader),
            Arc::clone(&valid_loader),
        )?,
        ResolvedLrScheduler::Step(scheduler) => train_with_scheduler(
            &run_dir,
            backend_name,
            training,
            &model_config,
            &device,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            scheduler,
            Arc::clone(&train_loader),
            Arc::clone(&valid_loader),
        )?,
        ResolvedLrScheduler::Noam(scheduler) => train_with_scheduler(
            &run_dir,
            backend_name,
            training,
            &model_config,
            &device,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            scheduler,
            Arc::clone(&train_loader),
            Arc::clone(&valid_loader),
        )?,
    };

    info!("Training complete on {backend_name}. Generating sample...");
    generate_sample::<B>(
        &model,
        dataset.as_ref(),
        &device,
        training.block_size,
        &config.generation,
    )?;

    Ok(())
}

enum ResolvedLrScheduler {
    Constant(LearningRate),
    Cosine(CosineAnnealingLrScheduler),
    Linear(LinearLrScheduler),
    Exponential(ExponentialLrScheduler),
    Step(StepLrScheduler),
    Noam(NoamLrScheduler),
}

fn train_with_scheduler<B, S>(
    run_dir: &Path,
    backend_name: &str,
    training: &TrainingHyperparameters,
    model_config: &BDHConfig,
    device: &B::Device,
    model: BDH<B>,
    optimizer: OptimizerAdaptor<AdamW, BDH<B>, B>,
    scheduler: S,
    train_loader: Arc<dyn DataLoader<B, ShakespeareBatch<B>>>,
    valid_loader: Arc<dyn DataLoader<ValidBackend<B>, ShakespeareBatch<ValidBackend<B>>>>,
) -> Result<BDH<B>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    S: LrScheduler + 'static,
{
    fs::create_dir_all(run_dir)?;

    let builder = LearnerBuilder::new(run_dir)
        .num_epochs(1)
        .devices(vec![device.clone()])
        .with_file_checkpointer(BinFileRecorder::<FullPrecisionSettings>::new())
        .metric_train_numeric(LossMetric::<ValidBackend<B>>::new())
        .metric_valid_numeric(LossMetric::<ValidBackend<B>>::new())
        .metric_train_numeric(LearningRateMetric::new())
        .summary();

    let learner = builder.build(model, optimizer, scheduler);

    log_theoretical_profile(
        model_config,
        training.batch_size,
        training.block_size,
        backend_name,
    );

    Ok(learner.fit(train_loader, valid_loader))
}

fn resolve_lr_scheduler(
    optimizer_cfg: &OptimizerConfig,
    training: &TrainingHyperparameters,
    model_config: &BDHConfig,
) -> Result<ResolvedLrScheduler> {
    let base_lr = optimizer_cfg.learning_rate;
    let fallback_iters = training.max_iters.max(1);

    let schedule = match &optimizer_cfg.lr_schedule {
        None => ResolvedLrScheduler::Constant(base_lr),
        Some(LearningRateScheduleConfig::Constant { initial_lr }) => {
            ResolvedLrScheduler::Constant(initial_lr.unwrap_or(base_lr))
        }
        Some(LearningRateScheduleConfig::Cosine {
            initial_lr,
            min_lr,
            num_iters,
        }) => {
            let init_lr = initial_lr.unwrap_or(base_lr);
            let scheduler = CosineAnnealingLrSchedulerConfig::new(
                init_lr,
                num_iters.unwrap_or(fallback_iters).max(1),
            )
            .with_min_lr(min_lr.unwrap_or(0.0))
            .init()
            .map_err(|err| anyhow!("failed to initialize cosine lr scheduler: {err}"))?;
            ResolvedLrScheduler::Cosine(scheduler)
        }
        Some(LearningRateScheduleConfig::Linear {
            initial_lr,
            final_lr,
            num_iters,
        }) => {
            let init_lr = initial_lr.unwrap_or(base_lr);
            let scheduler = LinearLrSchedulerConfig::new(
                init_lr,
                *final_lr,
                num_iters.unwrap_or(fallback_iters).max(1),
            )
            .init()
            .map_err(|err| anyhow!("failed to initialize linear lr scheduler: {err}"))?;
            ResolvedLrScheduler::Linear(scheduler)
        }
        Some(LearningRateScheduleConfig::Exponential { initial_lr, gamma }) => {
            let init_lr = initial_lr.unwrap_or(base_lr);
            let scheduler = ExponentialLrSchedulerConfig::new(init_lr, *gamma)
                .init()
                .map_err(|err| anyhow!("failed to initialize exponential lr scheduler: {err}"))?;
            ResolvedLrScheduler::Exponential(scheduler)
        }
        Some(LearningRateScheduleConfig::Step {
            initial_lr,
            gamma,
            step_size,
        }) => {
            let init_lr = initial_lr.unwrap_or(base_lr);
            let scheduler =
                StepLrSchedulerConfig::new(init_lr, step_size.unwrap_or(fallback_iters).max(1))
                    .with_gamma(*gamma)
                    .init()
                    .map_err(|err| anyhow!("failed to initialize step lr scheduler: {err}"))?;
            ResolvedLrScheduler::Step(scheduler)
        }
        Some(LearningRateScheduleConfig::Noam {
            initial_lr,
            warmup_steps,
            model_size,
        }) => {
            let init_lr = initial_lr.unwrap_or(base_lr);
            let mut config = NoamLrSchedulerConfig::new(init_lr);
            config = config.with_warmup_steps(warmup_steps.unwrap_or(fallback_iters).max(1));
            config = config.with_model_size(model_size.unwrap_or(model_config.n_embd).max(1));
            let scheduler = config
                .init()
                .map_err(|err| anyhow!("failed to initialize noam lr scheduler: {err}"))?;
            ResolvedLrScheduler::Noam(scheduler)
        }
    };

    Ok(schedule)
}

fn prepare_dataset(
    dataset_cfg: &DatasetConfig,
    training: &TrainingHyperparameters,
) -> Result<ShakespeareDataset> {
    ShakespeareDataset::new(
        &dataset_cfg.cache_dir,
        training.block_size,
        training.batch_size,
        dataset_cfg.train_split_ratio,
    )
    .with_context(|| "failed to prepare Shakespeare dataset")
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

fn log_theoretical_profile(config: &BDHConfig, batch: usize, block: usize, backend: &str) {
    let batch = batch as u64;
    let time = block as u64;
    let embed = config.n_embd as u64;
    let latent_per_head =
        (config.mlp_internal_dim_multiplier * config.n_embd / config.n_head) as u64;
    let latent_total = latent_per_head * config.n_head as u64;
    let heads = config.n_head as u64;
    let bt = batch * time;

    let encoder_matmul = 2 * bt * embed * latent_total;
    let attn_scores = 2 * batch * heads * time * time * latent_per_head;
    let attn_value = 2 * batch * heads * time * time * embed;
    let decoder_matmul = 2 * bt * latent_total * embed;
    let total = encoder_matmul + attn_scores + attn_value + decoder_matmul;

    info!(
        "[train:{backend}] approx forward GFLOPs: total={total_gflops:.2}, encoder={enc:.2}, \
         attn_scores={scores:.2}, attn_value={value:.2}, decoder={dec:.2} (backward ~2x forward)",
        total_gflops = total as f64 / 1e9,
        enc = encoder_matmul as f64 / 1e9,
        scores = attn_scores as f64 / 1e9,
        value = attn_value as f64 / 1e9,
        dec = decoder_matmul as f64 / 1e9,
    );
}

fn generate_sample<B: BackendTrait>(
    model: &BDH<B>,
    dataset: &ShakespeareDataset,
    device: &B::Device,
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
