use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use clap::{Parser, ValueEnum};

use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::Backend;
use burn_dragon_hatchling::wgpu::init_runtime;
use burn_dragon_hatchling::{
    generate_text, ContextStrategy, ContextStrategyConfig, BDH, BDHConfig, GenerationConfig,
    ModelOverrides, TrainingConfig, TrainingHyperparameters, load_training_config, prefill_state,
    sample_next_token,
};
use burn_dragon_hatchling::tokenizer::{SharedTokenizer, Tokenizer};
use burn_wgpu::Wgpu;

#[cfg(feature = "cuda")]
use burn_cuda::Cuda;

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = Args::parse();
    let mut config_paths = vec![PathBuf::from("config/base.toml")];
    config_paths.extend(args.config.clone());
    let config = load_training_config(&config_paths)?;

    match args.backend {
        BackendArg::Wgpu => infer_backend::<Wgpu<f32>, _>(&config, &args, "wgpu", init_runtime),
        BackendArg::Cuda => {
            #[cfg(feature = "cuda")]
            {
                infer_backend::<Cuda<f32>, _>(&config, &args, "cuda", |_| {})
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

fn infer_backend<B, Init>(
    config: &TrainingConfig,
    args: &Args,
    backend_name: &str,
    init_backend: Init,
) -> Result<()>
where
    B: Backend + 'static,
    B::Device: Clone,
    Init: Fn(&B::Device),
{
    B::seed(1337);
    let device = B::Device::default();
    init_backend(&device);

    let tokenizer = load_tokenizer(config)?;
    let checkpoint_dir = args
        .checkpoint
        .clone()
        .unwrap_or_else(|| PathBuf::from("runs").join(backend_name).join("checkpoint"));
    let (checkpoint_base, epoch) =
        resolve_checkpoint_base(&checkpoint_dir, args.epoch, backend_name)?;
    let model = load_model::<B>(&device, &config.model, tokenizer.len(), &checkpoint_base)?;

    let mut generation = config.generation.clone();
    apply_generation_overrides(&mut generation, args, config.training.block_size);

    eprintln!(
        "Loaded epoch {epoch} from {} using {backend_name} backend.",
        format_checkpoint(&checkpoint_base)
    );

    if args.streaming {
        stream_text(
            &model,
            tokenizer.as_ref(),
            &device,
            &config.training,
            &generation,
        )?;
        io::stdout().flush().ok();
        println!();
    } else {
        let output = generate_text::<B>(
            &model,
            tokenizer.as_ref(),
            &device,
            &config.training,
            &generation,
        )?;
        println!("{output}");
    }

    Ok(())
}

fn load_tokenizer(config: &TrainingConfig) -> Result<SharedTokenizer> {
    let tokenizer_cfg = &config.dataset.tokenizer;
    if let Some(path) = tokenizer_cfg.storage_path(&config.dataset.cache_dir) {
        tokenizer_cfg
            .load(&path)
            .with_context(|| format!("failed to load tokenizer {}", path.display()))
    } else {
        tokenizer_cfg
            .fit(std::iter::empty::<&str>())
            .context("failed to initialize tokenizer")
    }
}

fn load_model<B: Backend>(
    device: &B::Device,
    overrides: &ModelOverrides,
    vocab_size: usize,
    checkpoint_base: &Path,
) -> Result<BDH<B>> {
    let mut model_config = build_model_config(overrides);
    model_config.vocab_size = vocab_size;
    let model = BDH::<B>::new(model_config, device);
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let record = recorder
        .load::<<BDH<B> as Module<B>>::Record>(checkpoint_base.to_path_buf(), device)
        .with_context(|| {
            format!(
                "failed to load checkpoint {}",
                format_checkpoint(checkpoint_base)
            )
        })?;
    Ok(model.load_record(record))
}

fn stream_text<B: Backend>(
    model: &BDH<B>,
    tokenizer: &dyn Tokenizer,
    device: &B::Device,
    training: &TrainingHyperparameters,
    generation: &GenerationConfig,
) -> Result<()> {
    let strategy =
        resolve_context_strategy_infer(&generation.context_strategy, training.block_size);
    let mut prompt_ids = tokenizer.encode(&generation.prompt, false, false);
    if let ContextStrategy::Sliding { window } = strategy {
        if window > 0 && prompt_ids.len() > window {
            prompt_ids = prompt_ids[prompt_ids.len() - window..].to_vec();
        }
    }

    let prompt_tokens: Vec<i64> = prompt_ids.iter().map(|&id| id as i64).collect();
    let prompt_text = tokenizer.decode(&prompt_ids);
    let (mut state, mut last_logits) = prefill_state(model, &prompt_tokens, device)?;

    let mut stdout = io::stdout();
    write!(stdout, "{prompt_text}")?;
    stdout.flush()?;

    for _ in 0..generation.max_tokens {
        let (next_token, logits) = sample_next_token(
            model,
            &mut state,
            last_logits,
            generation.temperature,
            generation.top_k,
            device,
        )?;
        last_logits = logits;

        if let ContextStrategy::Sliding { window } = strategy {
            if window > 0 {
                state.trim_stream(0, window);
            }
        }

        if next_token < 0 {
            continue;
        }

        let token_id = next_token as u32;
        let piece = tokenizer.decode(&[token_id]);
        write!(stdout, "{piece}")?;
        stdout.flush()?;
    }

    Ok(())
}

fn resolve_context_strategy_infer(
    config: &ContextStrategyConfig,
    default_window: usize,
) -> ContextStrategy {
    match config {
        ContextStrategyConfig::Infinite => ContextStrategy::Infinite,
        ContextStrategyConfig::Sliding { window } => {
            let win = if *window == 0 {
                default_window.max(1)
            } else {
                *window
            };
            ContextStrategy::Sliding { window: win }
        }
    }
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

fn apply_generation_overrides(
    generation: &mut GenerationConfig,
    args: &Args,
    block_size: usize,
) {
    if let Some(prompt) = &args.prompt {
        generation.prompt = prompt.clone();
    }
    if let Some(max_tokens) = args.max_tokens {
        generation.max_tokens = max_tokens;
    }
    if let Some(temperature) = args.temperature {
        generation.temperature = temperature;
    }
    if let Some(top_k) = args.top_k {
        generation.top_k = Some(top_k);
    }
    if let Some(mode) = args.context_mode {
        generation.context_strategy = match mode {
            ContextModeArg::Infinite => ContextStrategyConfig::Infinite,
            ContextModeArg::Sliding => ContextStrategyConfig::Sliding {
                window: args.context_window.unwrap_or(block_size).max(1),
            },
        };
    }
}

fn resolve_checkpoint_base(
    path: &Path,
    epoch: Option<usize>,
    backend_name: &str,
) -> Result<(PathBuf, usize)> {
    if path.is_dir() {
        let target_epoch = epoch.unwrap_or(find_latest_epoch(path)?);
        let base = path.join(format!("model-{target_epoch}"));
        ensure_checkpoint_exists(&base)?;
        return Ok((base, target_epoch));
    }

    let mut base = if path.extension().is_some() {
        let mut without_ext = path.to_path_buf();
        without_ext.set_extension("");
        without_ext
    } else {
        path.to_path_buf()
    };

    let detected_epoch = parse_epoch_from_stem(&base);
    let target_epoch = match (epoch, detected_epoch) {
        (Some(explicit), Some(detected)) if explicit != detected => {
            let parent = base.parent().map(Path::to_path_buf).unwrap_or_default();
            base = parent.join(format!("model-{explicit}"));
            explicit
        }
        (Some(explicit), _) => {
            if detected_epoch.is_none() {
                let parent = base
                    .parent()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| PathBuf::from("runs").join(backend_name).join("checkpoint"));
                base = parent.join(format!("model-{explicit}"));
            }
            explicit
        }
        (None, Some(detected)) => detected,
        (None, None) => {
            return Err(anyhow!(
                "unable to infer checkpoint epoch from {}; provide --epoch",
                path.display()
            ));
        }
    };

    ensure_checkpoint_exists(&base)?;
    Ok((base, target_epoch))
}

fn ensure_checkpoint_exists(base: &Path) -> Result<()> {
    let mut candidate = base.to_path_buf();
    candidate.set_extension("bin");
    if candidate.is_file() {
        return Ok(());
    }

    Err(anyhow!("checkpoint file {}.bin not found", base.display()))
}

fn find_latest_epoch(dir: &Path) -> Result<usize> {
    let mut max_epoch = None;
    for entry in fs::read_dir(dir)
        .with_context(|| format!("failed to read checkpoint directory {}", dir.display()))?
    {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let mut base = entry.path();
        base.set_extension("");
        if let Some(epoch) = parse_epoch_from_stem(&base) {
            let updated = max_epoch
                .map(|current: usize| current.max(epoch))
                .unwrap_or(epoch);
            max_epoch = Some(updated);
        }
    }

    max_epoch.ok_or_else(|| anyhow!("no model checkpoints found in {}", dir.display()))
}

fn parse_epoch_from_stem(path: &Path) -> Option<usize> {
    let stem = path.file_name()?.to_string_lossy();
    let stem = stem.strip_suffix(".bin").unwrap_or(&stem);
    let epoch_part = stem.strip_prefix("model-")?;
    epoch_part.parse().ok()
}

fn format_checkpoint(base: &Path) -> String {
    let mut path = base.to_path_buf();
    path.set_extension("bin");
    path.display().to_string()
}

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Run inference with a trained Baby Dragon Hatchling model"
)]
struct Args {
    /// Additional configuration files applied in order (later files override earlier ones).
    #[arg(short = 'c', long = "config", value_name = "PATH")]
    config: Vec<PathBuf>,
    /// Backend to use for inference.
    #[arg(long, value_enum, default_value_t = BackendArg::Cuda)]
    backend: BackendArg,
    /// Path to the checkpoint directory or file.
    #[arg(long, value_name = "PATH")]
    checkpoint: Option<PathBuf>,
    /// Specific checkpoint epoch to load.
    #[arg(long, value_name = "N")]
    epoch: Option<usize>,
    /// Override the prompt used for generation.
    #[arg(long)]
    prompt: Option<String>,
    /// Override the number of tokens to generate.
    #[arg(long, value_name = "N")]
    max_tokens: Option<usize>,
    /// Override the sampling temperature.
    #[arg(long, value_name = "T")]
    temperature: Option<f32>,
    /// Override the top-k sampling parameter.
    #[arg(long, value_name = "K")]
    top_k: Option<usize>,
    /// Override the context strategy.
    #[arg(long, value_enum)]
    context_mode: Option<ContextModeArg>,
    /// Sliding window size when using `--context-mode=sliding`.
    #[arg(long, value_name = "N")]
    context_window: Option<usize>,
    /// Stream tokens to stdout as they are generated.
    #[arg(long)]
    streaming: bool,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum BackendArg {
    Wgpu,
    Cuda,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum ContextModeArg {
    Infinite,
    Sliding,
}
