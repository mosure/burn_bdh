use burn::module::{
    AutodiffModule, Content, Devices, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor, Param,
};
use burn::nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Distribution as TensorDistribution, Int, Tensor, TensorData, activation};
use rand::distributions::{Distribution, WeightedIndex};
use rand::prelude::*;
use std::cell::{RefCell, RefMut};
use std::cmp::Ordering;

use crate::ContextStrategy;
use crate::kernel::{BlockPattern1d, relu_lowrank};

use super::attention::{Attention, AttentionCache};
use super::config::{BDHConfig, FusedKernelConfig};
use super::state::ModelState;

const LAYER_NORM_EPS: f32 = 1e-5;

#[derive(Module, Debug)]
pub struct BDH<B: Backend> {
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    mlp_internal_dim_multiplier: usize,
    vocab_size: usize,
    kernel: FusedKernelConfig,
    embed: Embedding<B>,
    dropout: Dropout,
    attention: Attention<B>,
    encoder: Param<Tensor<B, 3>>,
    encoder_v: Param<Tensor<B, 3>>,
    decoder: Param<Tensor<B, 2>>,
    lm_head: Param<Tensor<B, 2>>,
    #[module(skip)]
    tbptt: TbpttHandle<B>,
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
            vocab_size: config.vocab_size,
            kernel: config.fused_kernels,
            embed,
            dropout,
            attention,
            encoder,
            encoder_v,
            decoder,
            lm_head,
            tbptt: TbpttHandle::default(),
        }
    }

    pub fn configure_tbptt(&self, strategy: ContextStrategy, batch_size: usize, block_size: usize) {
        self.tbptt.borrow_mut().replace(TbpttRuntime::new(
            strategy,
            batch_size,
            self.n_layer,
            block_size,
        ));
    }

    pub fn disable_tbptt(&self) {
        self.tbptt.borrow_mut().take();
    }

    pub fn forward_tbptt(
        &self,
        tokens: Tensor<B, 2, Int>,
        resets: &[bool],
        stream_ids: &[usize],
    ) -> Option<Tensor<B, 3>>
    where
        B: burn::tensor::backend::AutodiffBackend,
    {
        let mut guard = self.tbptt.borrow_mut();
        let runtime = guard.as_mut()?;

        let [batch_size, _] = tokens.shape().dims::<2>();
        if resets.len() != batch_size || stream_ids.len() != batch_size {
            return None;
        }

        if let Some(max_stream) = stream_ids.iter().copied().max() {
            runtime.ensure_capacity(max_stream + 1);
        }

        for (idx, &reset) in resets.iter().enumerate() {
            if reset {
                runtime.reset_stream(stream_ids[idx]);
            }
        }

        let logits = self.forward_with_state_streams(tokens, runtime.state_mut(), stream_ids);

        runtime.apply_strategy(stream_ids);
        runtime.detach();

        Some(logits)
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
        let [batch, _] = indices.shape().dims();
        assert_eq!(batch, 1, "generation currently supports batch size 1");

        let mut state = self.init_state();
        let mut logits = self.forward_with_state(indices.clone(), &mut state);
        let [_, mut time, vocab] = logits.shape().dims();
        assert_eq!(time, indices.shape().dims::<2>()[1]);

        let mut last_logits = logits
            .slice_dim(1, (time - 1)..time)
            .reshape([vocab])
            .div_scalar(temperature);

        for _ in 0..max_new_tokens {
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
            indices = Tensor::cat(vec![indices, next_token.clone()], 1);

            logits = self.forward_with_state(next_token, &mut state);
            let [_, new_time, _] = logits.shape().dims();
            time = new_time;
            last_logits = logits
                .slice_dim(1, (time - 1)..time)
                .reshape([vocab])
                .div_scalar(temperature);
        }

        indices
    }

    pub fn init_state(&self) -> ModelState<B> {
        ModelState::new(self.n_layer)
    }

    pub fn forward_with_state(
        &self,
        tokens: Tensor<B, 2, Int>,
        state: &mut ModelState<B>,
    ) -> Tensor<B, 3> {
        let [batch_size, _] = tokens.shape().dims::<2>();
        let stream_ids: Vec<usize> = (0..batch_size).collect();
        self.forward_with_state_streams(tokens, state, &stream_ids)
    }

    fn forward_with_state_streams(
        &self,
        tokens: Tensor<B, 2, Int>,
        state: &mut ModelState<B>,
        stream_ids: &[usize],
    ) -> Tensor<B, 3> {
        assert_eq!(
            state.layers.len(),
            self.n_layer,
            "model state layers mismatch"
        );
        assert_eq!(
            tokens.shape().dims::<2>()[0],
            stream_ids.len(),
            "stream id count must match batch size"
        );

        if let Some(max_stream) = stream_ids.iter().copied().max() {
            state.ensure_capacity(max_stream + 1);
        }

        let mut current = self.embed.forward(tokens).unsqueeze_dim::<4>(1);
        current = self.layer_norm(current);

        let encoder = self.encoder.val().unsqueeze_dim::<4>(0);
        let encoder_v = self.encoder_v.val().unsqueeze_dim::<4>(0);
        let decoder = self.decoder.val();
        let fused = self.kernel.enabled;
        let latent_pattern: &BlockPattern1d = &self.kernel.block_sparse.latent;

        for layer_state in &mut state.layers {
            let x_sparse = if fused {
                relu_lowrank::fused_forward(
                    current.clone(),
                    encoder.clone(),
                    None,
                    self.kernel.relu_threshold,
                    latent_pattern,
                )
            } else {
                let mut x_latent = current.clone().matmul(encoder.clone());
                if self.kernel.relu_threshold != 0.0 {
                    x_latent = x_latent.sub_scalar(self.kernel.relu_threshold);
                }
                activation::relu(x_latent)
            };

            let attn = self.forward_attention_streams(
                x_sparse.clone(),
                current.clone(),
                &mut layer_state.attention,
                stream_ids,
            );
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
            let xy_sparse = self.dropout.forward(xy_sparse);

            let mixed = xy_sparse.swap_dims(1, 2);
            let [batch, time, heads, latent] = mixed.shape().dims();
            let mixed_flat = mixed.reshape([batch * time, heads * latent]);

            let mlp_flat = mixed_flat.matmul(decoder.clone());
            let mlp_out = mlp_flat
                .reshape([batch, time, self.n_embd])
                .unsqueeze_dim::<4>(1);
            let mlp_out = self.layer_norm(mlp_out);
            current = self.layer_norm(current + mlp_out);
        }

        let [batch, _, time, dim] = current.shape().dims();
        current
            .reshape([batch * time, dim])
            .matmul(self.lm_head.val())
            .reshape([batch, time, self.vocab_size])
    }

    fn forward_attention_streams(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        cache: &mut AttentionCache<B>,
        stream_ids: &[usize],
    ) -> Tensor<B, 4> {
        cache.forward_batch(&self.attention, query, value, stream_ids)
    }
}

#[derive(Debug, Clone, Default)]
struct TbpttHandle<B: Backend> {
    inner: RefCell<Option<TbpttRuntime<B>>>,
}

impl<B: Backend> TbpttHandle<B> {
    fn borrow_mut(&self) -> RefMut<'_, Option<TbpttRuntime<B>>> {
        self.inner.borrow_mut()
    }
}

impl<B: Backend> Module<B> for TbpttHandle<B> {
    type Record = ();

    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {
        ()
    }
}

impl<B: AutodiffBackend> AutodiffModule<B> for TbpttHandle<B> {
    type InnerModule = TbpttHandle<B::InnerBackend>;

    fn valid(&self) -> Self::InnerModule {
        TbpttHandle::default()
    }
}

impl<B: Backend> ModuleDisplayDefault for TbpttHandle<B> {
    fn content(&self, content: Content) -> Option<Content> {
        Some(content)
    }
}

impl<B: Backend> ModuleDisplay for TbpttHandle<B> {}

#[derive(Debug, Clone)]
struct TbpttRuntime<B: Backend> {
    state: ModelState<B>,
    strategy: ContextStrategy,
    block_size: usize,
}

impl<B: Backend> TbpttRuntime<B> {
    fn new(
        strategy: ContextStrategy,
        initial_streams: usize,
        num_layers: usize,
        block_size: usize,
    ) -> Self {
        let mut state = ModelState::new(num_layers);
        if initial_streams > 0 {
            state.ensure_capacity(initial_streams);
        }
        Self {
            state,
            strategy,
            block_size,
        }
    }

    fn ensure_capacity(&mut self, capacity: usize) {
        self.state.ensure_capacity(capacity);
    }

    fn reset_stream(&mut self, stream: usize) {
        self.state.reset_stream(stream);
    }

    fn state_mut(&mut self) -> &mut ModelState<B> {
        &mut self.state
    }

    fn detach(&mut self)
    where
        B: burn::tensor::backend::AutodiffBackend,
    {
        self.state.detach();
    }

    fn apply_strategy(&mut self, stream_ids: &[usize]) {
        let limit = match self.strategy {
            ContextStrategy::Infinite => self.block_size,
            ContextStrategy::Sliding { window } => window,
        }
        .max(self.block_size)
        .max(1);
        self.state.trim_streams(stream_ids, limit);
    }
}
