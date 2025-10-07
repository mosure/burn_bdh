use burn::module::{Module, Param};
use burn::nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution as TensorDistribution, Int, Tensor, TensorData, activation};
use rand::distributions::{Distribution, WeightedIndex};
use rand::prelude::*;
use std::cmp::Ordering;

use crate::kernel::{BlockPattern1d, relu_lowrank};

use super::attention::Attention;
use super::config::{BDHConfig, FusedKernelConfig};
use super::router::Router;

const LAYER_NORM_EPS: f32 = 1e-5;

#[derive(Module, Debug)]
pub struct BDH<B: Backend> {
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    mlp_internal_dim_multiplier: usize,
    n_expert: usize,
    vocab_size: usize,
    kernel: FusedKernelConfig,
    embed: Embedding<B>,
    dropout: Dropout,
    attention: Attention<B>,
    router: Router<B>,
    encoder: Param<Tensor<B, 3>>,
    encoder_v: Param<Tensor<B, 3>>,
    decoder: Param<Tensor<B, 2>>,
    lm_head: Param<Tensor<B, 2>>,
}

impl<B: Backend> BDH<B> {
    pub fn new(config: BDHConfig, device: &B::Device) -> Self {
        let embed = EmbeddingConfig::new(config.vocab_size, config.n_embd).init(device);
        let dropout = DropoutConfig::new(config.dropout).init();

        let latent_per_head = config.latent_per_head();
        let latent_total = config.latent_total();
        let attention = Attention::new(
            latent_per_head,
            config.n_head,
            device,
            &config.fused_kernels,
        );
        let router = Router::new(
            config.n_expert,
            config.n_head,
            config.n_embd,
            latent_per_head,
            device,
        );

        let weight_init = |shape: [usize; 2]| {
            Tensor::<B, 2>::random(shape, TensorDistribution::Normal(0.0, 0.02), device)
        };

        let encoder = Param::from_tensor(Tensor::<B, 3>::random(
            [config.n_head, config.n_embd, latent_per_head],
            TensorDistribution::Normal(0.0, 0.02),
            device,
        ));

        let encoder_v = Param::from_tensor(Tensor::<B, 3>::random(
            [config.n_head, config.n_embd, latent_per_head],
            TensorDistribution::Normal(0.0, 0.02),
            device,
        ));

        let decoder = Param::from_tensor(weight_init([latent_total, config.n_embd]));
        let lm_head = Param::from_tensor(weight_init([config.n_embd, config.vocab_size]));

        Self {
            n_layer: config.n_layer,
            n_embd: config.n_embd,
            n_head: config.n_head,
            mlp_internal_dim_multiplier: config.mlp_internal_dim_multiplier,
            n_expert: config.n_expert,
            vocab_size: config.vocab_size,
            kernel: config.fused_kernels,
            embed,
            dropout,
            attention,
            router,
            encoder,
            encoder_v,
            decoder,
            lm_head,
        }
    }

    fn layer_norm<const D: usize>(&self, tensor: Tensor<B, D>) -> Tensor<B, D> {
        let (var, mean) = tensor.clone().var_mean_bias(D - 1);
        tensor.sub(mean).div(var.add_scalar(LAYER_NORM_EPS).sqrt())
    }

    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut state = self.embed.forward(tokens).unsqueeze_dim::<4>(1);
        state = self.layer_norm(state);

        let encoder = self.encoder.val().unsqueeze_dim::<4>(0);
        let encoder_v = self.encoder_v.val().unsqueeze_dim::<4>(0);
        let decoder = self.decoder.val();
        let fused = self.kernel.enabled;
        let latent_pattern: &BlockPattern1d = &self.kernel.block_sparse.latent;

        for _ in 0..self.n_layer {
            let x_sparse = if fused {
                relu_lowrank::fused_forward(
                    state.clone(),
                    encoder.clone(),
                    None,
                    self.kernel.relu_threshold,
                    latent_pattern,
                )
            } else {
                let mut x_latent = state.clone().matmul(encoder.clone());
                if self.kernel.relu_threshold != 0.0 {
                    x_latent = x_latent.sub_scalar(self.kernel.relu_threshold);
                }
                activation::relu(x_latent)
            };

            let attn = self.attention.forward(x_sparse.clone(), state.clone());
            let attn = self.layer_norm(attn);

            let y_sparse = if fused {
                relu_lowrank::fused_forward(
                    attn.clone(),
                    encoder_v.clone(),
                    None,
                    self.kernel.relu_threshold,
                    latent_pattern,
                )
            } else {
                let mut y_latent = attn.matmul(encoder_v.clone());
                if self.kernel.relu_threshold != 0.0 {
                    y_latent = y_latent.sub_scalar(self.kernel.relu_threshold);
                }
                activation::relu(y_latent)
            };
            let xy_sparse = x_sparse * y_sparse;
            let xy_sparse = self.router.route(state.clone(), xy_sparse);
            let xy_sparse = self.dropout.forward(xy_sparse);

            let mixed = xy_sparse.swap_dims(1, 2);
            let [batch, time, heads, latent] = mixed.shape().dims();
            let mixed_flat = mixed.reshape([batch * time, heads * latent]);

            let mlp_flat = mixed_flat.matmul(decoder.clone());
            let mlp_out = mlp_flat
                .reshape([batch, time, self.n_embd])
                .unsqueeze_dim::<4>(1);
            let mlp_out = self.layer_norm(mlp_out);
            state = self.layer_norm(state + mlp_out);
        }

        let [batch, _, time, dim] = state.shape().dims();
        state
            .reshape([batch * time, dim])
            .matmul(self.lm_head.val())
            .reshape([batch, time, self.vocab_size])
    }

    pub fn generate(
        &self,
        mut indices: Tensor<B, 2, Int>,
        max_new_tokens: usize,
        temperature: f32,
        top_k: Option<usize>,
    ) -> Tensor<B, 2, Int> {
        for _ in 0..max_new_tokens {
            let logits = self.forward(indices.clone());
            let [batch, time, vocab] = logits.shape().dims();
            assert_eq!(batch, 1, "generation currently supports batch size 1");

            let last_logits = logits
                .slice_dim(1, (time - 1)..time)
                .reshape([vocab])
                .div_scalar(temperature);

            let mut logits_values = last_logits
                .clone()
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("logits to vec");

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

            let dist = WeightedIndex::new(&probs).expect("valid probability distribution");
            let mut rng = thread_rng();
            let next = dist.sample(&mut rng) as i64;

            let next_token = Tensor::<B, 2, Int>::from_data(
                TensorData::new(vec![next], [1, 1]),
                &indices.device(),
            );
            indices = Tensor::cat(vec![indices, next_token], 1);
        }

        indices
    }
}
