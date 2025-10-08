use anyhow::{Result, anyhow};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use std::cmp::Ordering;

use crate::config::ContextStrategyConfig;
use crate::tokenizer::Tokenizer;
use crate::{BDH, GenerationConfig, TrainingHyperparameters, ModelState};

#[derive(Clone, Copy, Debug)]
pub enum ContextStrategy {
    Infinite,
    Sliding { window: usize },
}

pub fn prefill_state<B: Backend>(
    model: &BDH<B>,
    prompt_tokens: &[i64],
    device: &B::Device,
) -> Result<(ModelState<B>, Tensor<B, 1>)> {
    let prompt_len = prompt_tokens.len();
    if prompt_len == 0 {
        return Err(anyhow!("prompt must contain at least one token"));
    }

    let prompt_tensor = Tensor::<B, 2, Int>::from_data(
        TensorData::new(prompt_tokens.to_vec(), [1, prompt_len]),
        device,
    );

    let mut state = model.init_state();
    let logits = model.forward_with_state(prompt_tensor, &mut state);
    let [_, time, vocab] = logits.shape().dims::<3>();
    if time != prompt_len {
        return Err(anyhow!(
            "prefill produced mismatched length: expected {prompt_len}, got {time}"
        ));
    }

    let last_logits = logits
        .slice_dim(1, (time - 1)..time)
        .reshape([vocab]);

    Ok((state, last_logits))
}

pub fn sample_next_token<B: Backend>(
    model: &BDH<B>,
    state: &mut ModelState<B>,
    last_logits: Tensor<B, 1>,
    temperature: f32,
    top_k: Option<usize>,
    device: &B::Device,
) -> Result<(i64, Tensor<B, 1>)> {
    let logits_temp = last_logits.clone().div_scalar(temperature);
    let vocab = logits_temp.dims()[0];

    let mut logits_values = logits_temp
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(|err| anyhow!("{err:?}"))?;

    if let Some(k) = top_k
        && k > 0
        && k < vocab
    {
        let mut sorted = logits_values.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        let threshold = sorted[k - 1];
        for value in logits_values.iter_mut() {
            if *value < threshold {
                *value = f32::NEG_INFINITY;
            }
        }
    }

    let max_logit = logits_values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits_values
        .iter()
        .map(|value| (value - max_logit).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    if sum == 0.0 || sum.is_nan() {
        let uniform = 1.0 / vocab as f32;
        for p in probs.iter_mut() {
            *p = uniform;
        }
    } else {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }

    let dist = WeightedIndex::new(&probs).map_err(|err| anyhow!(err.to_string()))?;
    let mut rng = thread_rng();
    let next = dist.sample(&mut rng) as i64;

    let next_tensor =
        Tensor::<B, 2, Int>::from_data(TensorData::new(vec![next], [1, 1]), device);

    let logits = model.forward_with_state(next_tensor, state);
    let [_, time, vocab] = logits.shape().dims::<3>();
    let new_last_logits = logits
        .slice_dim(1, (time - 1)..time)
        .reshape([vocab]);

    Ok((next, new_last_logits))
}

pub fn generate_tokens<B: Backend>(
    model: &BDH<B>,
    prompt_tokens: Vec<i64>,
    device: &B::Device,
    max_new_tokens: usize,
    temperature: f32,
    top_k: Option<usize>,
    strategy: ContextStrategy,
) -> Result<Vec<i64>> {
    let mut full_tokens = prompt_tokens;
    let (mut state, last_logits) = prefill_state(model, &full_tokens, device)?;
    let (generated, _) =
        advance_generation(model, &mut state, last_logits, device, max_new_tokens, temperature, top_k, strategy)?;
    full_tokens.extend(generated);
    Ok(full_tokens)
}

pub fn generate_text<B: Backend>(
    model: &BDH<B>,
    tokenizer: &dyn Tokenizer,
    device: &B::Device,
    training: &TrainingHyperparameters,
    generation: &GenerationConfig,
) -> Result<String> {
    let strategy = resolve_context_strategy(&generation.context_strategy, training.block_size);
    let mut prompt_ids = tokenizer.encode(&generation.prompt, false, false);
    if let ContextStrategy::Sliding { window } = strategy {
        if prompt_ids.len() > window {
            prompt_ids = prompt_ids[prompt_ids.len() - window..].to_vec();
        }
    }

    let prompt_tokens: Vec<i64> = prompt_ids.iter().map(|&id| id as i64).collect();
    let tokens_all = generate_tokens(
        model,
        prompt_tokens,
        device,
        generation.max_tokens,
        generation.temperature,
        generation.top_k,
        strategy,
    )?;

    let decoded_ids: Vec<u32> = tokens_all
        .iter()
        .filter_map(|&tok| (tok >= 0).then(|| tok as u32))
        .collect();

    Ok(tokenizer.decode(&decoded_ids))
}

pub fn advance_generation<B: Backend>(
    model: &BDH<B>,
    state: &mut ModelState<B>,
    mut last_logits: Tensor<B, 1>,
    device: &B::Device,
    steps: usize,
    temperature: f32,
    top_k: Option<usize>,
    strategy: ContextStrategy,
) -> Result<(Vec<i64>, Tensor<B, 1>)> {
    let mut generated = Vec::with_capacity(steps);
    for _ in 0..steps {
        let (next, logits) =
            sample_next_token(model, state, last_logits, temperature, top_k, device)?;
        generated.push(next);
        last_logits = logits;

        if let ContextStrategy::Sliding { window } = strategy {
            if window > 0 {
                state.trim_stream(0, window);
            }
        }
    }

    Ok((generated, last_logits))
}

fn resolve_context_strategy(config: &ContextStrategyConfig, default_window: usize) -> ContextStrategy {
    match config {
        ContextStrategyConfig::Infinite => ContextStrategy::Infinite,
        ContextStrategyConfig::Sliding { window } => {
            let win = if *window == 0 { default_window.max(1) } else { *window };
            ContextStrategy::Sliding { window: win }
        }
    }
}
