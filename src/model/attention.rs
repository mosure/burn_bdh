use std::f32::consts::PI;

use burn::module::Module;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor, TensorData};

use super::config::FusedKernelConfig;
use crate::kernel::{BlockPattern2d, linear_attention};

#[derive(Default, Debug, Clone)]
pub struct AttentionCache<B: Backend> {
    streams: Vec<StreamCache<B>>,
}

impl<B: Backend> AttentionCache<B> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn ensure_streams(&mut self, count: usize) {
        if self.streams.len() < count {
            self.streams
                .resize_with(count, StreamCache::<B>::default);
        }
    }

    pub fn forward_batch(
        &mut self,
        attention: &Attention<B>,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        stream_ids: &[usize],
    ) -> Tensor<B, 4> {
        if stream_ids.is_empty() {
            return value;
        }

        if stream_ids.len() == 1 {
            let stream_id = stream_ids[0];
            self.ensure_streams(stream_id + 1);
            let stream_cache = &mut self.streams[stream_id];
            let position = stream_cache.len();
            let (context, _) =
                attention.forward_stream_cached(query, value, stream_cache, position);
            return context;
        }

        let batch = stream_ids.len();
        let time_new = query.shape().dims::<4>()[2];
        let latent_q = query.shape().dims::<4>()[3];
        let latent_v = value.shape().dims::<4>()[3];
        let device = query.device();

        if let Some(max_stream) = stream_ids.iter().copied().max() {
            self.ensure_streams(max_stream + 1);
        }

        let mut prev_lengths = Vec::with_capacity(batch);
        let mut prev_keys: Vec<Option<Tensor<B, 4>>> = Vec::with_capacity(batch);
        let mut prev_values: Vec<Option<Tensor<B, 4>>> = Vec::with_capacity(batch);

        for &stream_id in stream_ids {
            let cache = &self.streams[stream_id];
            let len = cache.len();
            prev_lengths.push(len);
            prev_keys.push(cache.q_rot.clone());
            prev_values.push(cache.value.clone());
        }

        let max_prev_len = prev_lengths.iter().copied().max().unwrap_or(0);
        let effective_prev_len = max_prev_len.max(1);

        let prev_keys_tensor = if max_prev_len > 0 {
            let mut tensors = Vec::with_capacity(batch);
            for (idx, key_opt) in prev_keys.into_iter().enumerate() {
                let len = prev_lengths[idx];
                if let Some(key) = key_opt {
                    if len == effective_prev_len {
                        tensors.push(key);
                    } else {
                        let pad = Tensor::<B, 4>::zeros(
                            [1, attention.n_head, effective_prev_len - len, latent_q],
                            &device,
                        );
                        tensors.push(Tensor::cat(vec![key, pad], 2));
                    }
                } else {
                    tensors.push(Tensor::<B, 4>::zeros(
                        [1, attention.n_head, effective_prev_len, latent_q],
                        &device,
                    ));
                }
            }
            Some(Tensor::cat(tensors, 0))
        } else {
            None
        };

        let prev_values_tensor = if max_prev_len > 0 {
            let mut tensors = Vec::with_capacity(batch);
            for (idx, value_opt) in prev_values.into_iter().enumerate() {
                let len = prev_lengths[idx];
                if let Some(value_cached) = value_opt {
                    if len == effective_prev_len {
                        tensors.push(value_cached);
                    } else {
                        let pad = Tensor::<B, 4>::zeros(
                            [1, attention.n_head, effective_prev_len - len, latent_v],
                            &device,
                        );
                        tensors.push(Tensor::cat(vec![value_cached, pad], 2));
                    }
                } else {
                    tensors.push(Tensor::<B, 4>::zeros(
                        [1, attention.n_head, effective_prev_len, latent_v],
                        &device,
                    ));
                }
            }
            Some(Tensor::cat(tensors, 0))
        } else {
            None
        };

        let (context, new_keys, new_values) = attention.forward_batch_cached(
            query.clone(),
            value.clone(),
            prev_keys_tensor,
            prev_values_tensor,
            &prev_lengths,
            effective_prev_len,
        );

        for (batch_idx, &stream_id) in stream_ids.iter().enumerate() {
            let cache = &mut self.streams[stream_id];
            let key_slice = new_keys.clone().slice([
                batch_idx..batch_idx + 1,
                0..attention.n_head,
                0..time_new,
                0..latent_q,
            ]);
            let value_slice = new_values.clone().slice([
                batch_idx..batch_idx + 1,
                0..attention.n_head,
                0..time_new,
                0..latent_v,
            ]);
            cache.append(key_slice, value_slice);
        }

        context
    }

    pub fn reset_stream(&mut self, stream: usize) {
        if stream < self.streams.len() {
            self.streams[stream].reset();
        }
    }

    pub fn reset_all(&mut self) {
        for stream in &mut self.streams {
            stream.reset();
        }
    }

    pub fn trim_stream(&mut self, stream: usize, max_len: usize) {
        if stream < self.streams.len() {
            self.streams[stream].retain_last(max_len);
        }
    }
}

impl<B: burn::tensor::backend::AutodiffBackend> AttentionCache<B> {
    pub fn detach(&mut self) {
        for stream in &mut self.streams {
            stream.detach();
        }
    }
}

#[derive(Default, Debug, Clone)]
struct StreamCache<B: Backend> {
    q_rot: Option<Tensor<B, 4>>,
    value: Option<Tensor<B, 4>>,
    len: usize,
}

impl<B: Backend> StreamCache<B> {
    fn reset(&mut self) {
        self.q_rot = None;
        self.value = None;
        self.len = 0;
    }

    fn append(&mut self, q: Tensor<B, 4>, v: Tensor<B, 4>) {
        let q_new = match self.q_rot.take() {
            Some(prev) => Tensor::cat(vec![prev, q], 2),
            None => q,
        };
        let v_new = match self.value.take() {
            Some(prev) => Tensor::cat(vec![prev, v], 2),
            None => v,
        };
        self.len = q_new.shape().dims::<4>()[2];
        self.q_rot = Some(q_new);
        self.value = Some(v_new);
    }

    fn len(&self) -> usize {
        self.len
    }

    fn retain_last(&mut self, max_len: usize) {
        if max_len == 0 {
            self.reset();
            return;
        }

        if let Some(q) = &self.q_rot {
            let dims = q.shape().dims::<4>();
            if dims[2] > max_len {
                let start = dims[2] - max_len;
                let q_new = q.clone().slice([
                    0..dims[0],
                    0..dims[1],
                    start..dims[2],
                    0..dims[3],
                ]);
                self.q_rot = Some(q_new);
            }
        }

        if let Some(v) = &self.value {
            let dims = v.shape().dims::<4>();
            if dims[2] > max_len {
                let start = dims[2] - max_len;
                let v_new = v.clone().slice([
                    0..dims[0],
                    0..dims[1],
                    start..dims[2],
                    0..dims[3],
                ]);
                self.value = Some(v_new);
            }
        }

        self.len = self
            .q_rot
            .as_ref()
            .map(|tensor| tensor.shape().dims::<4>()[2])
            .unwrap_or(0);
    }
}

impl<B: AutodiffBackend> StreamCache<B> {
    fn detach(&mut self) {
        if let Some(q) = self.q_rot.take() {
            self.q_rot = Some(q.detach());
        }
        if let Some(v) = self.value.take() {
            self.value = Some(v.detach());
        }
    }
}

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    freqs: Tensor<B, 4>,
    n_head: usize,
    fused: bool,
    block_pattern: BlockPattern2d,
    use_alibi: bool,
    alibi_slopes: Tensor<B, 1>,
}

impl<B: Backend> Attention<B> {
    pub fn new(
        latent: usize,
        n_head: usize,
        device: &B::Device,
        kernel: &FusedKernelConfig,
    ) -> Self {
        let freqs = Self::build_freqs(latent, kernel.rope_theta, device);
        let (use_alibi, alibi_slopes) = if kernel.enabled && kernel.use_alibi {
            let slopes = kernel
                .alibi_slopes
                .clone()
                .unwrap_or_else(|| linear_attention::default_alibi_slopes(n_head));
            (true, Tensor::<B, 1>::from_floats(slopes.as_slice(), device))
        } else {
            (false, Tensor::<B, 1>::zeros([n_head], device))
        };

        Self {
            freqs,
            n_head,
            fused: kernel.enabled,
            block_pattern: kernel.block_sparse.time.clone(),
            use_alibi,
            alibi_slopes,
        }
    }

    pub fn forward(&self, query: Tensor<B, 4>, value: Tensor<B, 4>) -> Tensor<B, 4> {
        if self.fused {
            return linear_attention::fused_state_aligned(
                query,
                value,
                self.freqs.clone(),
                self.use_alibi.then(|| self.alibi_slopes.clone()),
                &self.block_pattern,
            );
        }


        let q_rot = self.rotate(query, 0);
        let k_rot = q_rot.clone();

        let scores = q_rot.matmul(k_rot.swap_dims(2, 3)).tril(-1);
        let value = value.repeat_dim(1, self.n_head);

        scores.matmul(value)
    }

    pub fn forward_batch_cached(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        prev_keys: Option<Tensor<B, 4>>,
        prev_values: Option<Tensor<B, 4>>,
        prev_lengths: &[usize],
        max_prev_len: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let batch = query.shape().dims::<4>()[0];
        let time_new = query.shape().dims::<4>()[2];
        let device = query.device();

        let lengths_data: Vec<i64> = prev_lengths.iter().map(|&len| len as i64).collect();
        let starts = Tensor::<B, 1, Int>::from_data(
            TensorData::new(lengths_data.clone(), [batch]),
            &device,
        )
        .reshape([batch, 1, 1, 1]);

        let q_rot = self.rotate_with_starts(query.clone(), starts.clone());
        let k_rot = q_rot.clone();
        let value_new = value.repeat_dim(1, self.n_head);

        let offsets = starts.clone().float();
        let arange_time = Tensor::<B, 1, Int>::arange(0..time_new as i64, &device).float();
        let pos_row = offsets.clone() + arange_time.clone().reshape([1, 1, time_new, 1]);
        let pos_new = offsets.clone() + arange_time.clone().reshape([1, 1, 1, time_new]);

        if max_prev_len == 0 || prev_keys.is_none() || prev_values.is_none() {
            let mut scores_self = q_rot.matmul(k_rot.clone().swap_dims(2, 3)).tril(-1);
            if self.use_alibi {
                let slopes = self.alibi_slopes.clone().reshape([1, self.n_head, 1, 1]);
                let alibi_self = slopes * (pos_new - pos_row).tril(-1);
                scores_self = scores_self + alibi_self;
            }
            let context = scores_self.matmul(value_new.clone());
            return (context, k_rot, value_new);
        }

        let prev_keys_tensor = prev_keys.expect("previous keys missing");
        let prev_values_tensor = prev_values.expect("previous values missing");

        let mut value_all = value_new.clone();
        let mut scores_prev_opt = None;

        if max_prev_len > 0 {
            let lengths_tensor = Tensor::<B, 1, Int>::from_data(
                TensorData::new(lengths_data, [batch]),
                &device,
            )
            .reshape([batch, 1, 1, 1])
            .float();
            let positions_prev = Tensor::<B, 1, Int>::arange(0..max_prev_len as i64, &device)
                .float()
                .reshape([1, 1, 1, max_prev_len]);
            let mask_scores = positions_prev
                .clone()
                .lower(lengths_tensor.clone())
                .float();
            let mask_values = mask_scores.clone().swap_dims(2, 3);

            let prev_values_masked = prev_values_tensor.clone() * mask_values.clone();
            let mut scores_prev =
                q_rot.clone().matmul(prev_keys_tensor.clone().swap_dims(2, 3));
            scores_prev = scores_prev * mask_scores.clone();

            if self.use_alibi {
                let slopes = self.alibi_slopes.clone().reshape([1, self.n_head, 1, 1]);
                let alibi_prev = slopes.clone()
                    * (positions_prev - pos_row.clone());
                scores_prev = scores_prev + alibi_prev * mask_scores.clone();
            }

            scores_prev_opt = Some(scores_prev);
            value_all = Tensor::cat(vec![prev_values_masked, value_new.clone()], 2);
        }

        let mut scores_self = q_rot.matmul(k_rot.clone().swap_dims(2, 3)).tril(-1);
        if self.use_alibi {
            let slopes = self.alibi_slopes.clone().reshape([1, self.n_head, 1, 1]);
            let alibi_self = slopes * (pos_new - pos_row).tril(-1);
            scores_self = scores_self + alibi_self;
        }

        let scores = if let Some(scores_prev) = scores_prev_opt {
            Tensor::cat(vec![scores_prev, scores_self], 3)
        } else {
            scores_self
        };

        let context = scores.matmul(value_all);

        (context, k_rot, value_new)
    }

    #[allow(dead_code)]
    fn forward_stream_cached(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        cache: &mut StreamCache<B>,
        position: usize,
    ) -> (Tensor<B, 4>, usize) {
        let time_new = query.shape().dims::<4>()[2];

        let q_rot = self.rotate(query, position);
        let k_rot = q_rot.clone();
        let value_rep = value.repeat_dim(1, self.n_head);

        let context = if let (Some(prev_q), Some(prev_v)) = (&cache.q_rot, &cache.value) {
            let scores_prev = q_rot.clone().matmul(prev_q.clone().swap_dims(2, 3));
            let mut scores_self = q_rot
                .clone()
                .matmul(k_rot.clone().swap_dims(2, 3))
                .tril(-1);

            let scores_prev = if self.use_alibi {
                let device = q_rot.device();
                let slopes = self.alibi_slopes.clone().reshape([1, self.n_head, 1, 1]);
                let prev_len = cache.len();

                let pos_row = Tensor::<B, 1, Int>::arange(
                    position as i64..(position + time_new) as i64,
                    &device,
                )
                .float()
                .reshape([1, 1, time_new, 1]);

                let pos_prev = Tensor::<B, 1, Int>::arange(0..prev_len as i64, &device)
                    .float()
                    .reshape([1, 1, 1, prev_len]);
                let alibi_prev = slopes.clone() * (pos_prev - pos_row.clone());

                let pos_new = Tensor::<B, 1, Int>::arange(
                    position as i64..(position + time_new) as i64,
                    &device,
                )
                .float()
                .reshape([1, 1, 1, time_new]);
                let alibi_self = slopes * (pos_new - pos_row).tril(-1);

                scores_self = scores_self + alibi_self;
                scores_prev + alibi_prev
            } else {
                scores_prev
            };

            let scores = Tensor::cat(vec![scores_prev, scores_self], 3);
            let value_all = Tensor::cat(vec![prev_v.clone(), value_rep.clone()], 2);
            scores.matmul(value_all)
        } else {
            let mut scores = q_rot
                .clone()
                .matmul(k_rot.clone().swap_dims(2, 3))
                .tril(-1);
            if self.use_alibi {
                let device = q_rot.device();
                let slopes = self.alibi_slopes.clone().reshape([1, self.n_head, 1, 1]);
                let pos_row = Tensor::<B, 1, Int>::arange(
                    position as i64..(position + time_new) as i64,
                    &device,
                )
                .float()
                .reshape([1, 1, time_new, 1]);
                let pos_new = Tensor::<B, 1, Int>::arange(
                    position as i64..(position + time_new) as i64,
                    &device,
                )
                .float()
                .reshape([1, 1, 1, time_new]);
                let alibi = slopes * (pos_new - pos_row).tril(-1);
                scores = scores + alibi;
            }
            scores.matmul(value_rep.clone())
        };

        cache.append(k_rot.clone(), value_rep.clone());
        let new_position = position + time_new;

        (context, new_position)
    }

    fn rope(&self, phases: Tensor<B, 4>, values: Tensor<B, 4>) -> Tensor<B, 4> {
        let cos = phases.clone().cos();
        let sin = phases.sin();

        let [b, h, t, n] = values.shape().dims();
        let pairs = values.clone().reshape([b, h, t, n / 2, 2]);

        let even = pairs.clone().slice_dim(4, 0..1).squeeze::<4>(4);
        let odd = pairs.slice_dim(4, 1..2).squeeze::<4>(4);

        let rotated = Tensor::stack::<5>(vec![odd.clone().neg(), even], 4).reshape([b, h, t, n]);

        values * cos + rotated * sin
    }

    fn rotate(&self, values: Tensor<B, 4>, start: usize) -> Tensor<B, 4> {
        let batch = values.shape().dims::<4>()[0];
        let device = values.device();
        let starts = Tensor::<B, 1, Int>::from_data(
            TensorData::new(vec![start as i64; batch], [batch]),
            &device,
        )
        .reshape([batch, 1, 1, 1]);
        self.rotate_with_starts(values, starts)
    }

    fn rotate_with_starts(&self, values: Tensor<B, 4>, starts: Tensor<B, 4, Int>) -> Tensor<B, 4> {
        let time = values.shape().dims::<4>()[2];
        let device = values.device();
        let steps = Tensor::<B, 1, Int>::arange(0..time as i64, &device)
            .float()
            .reshape([1, 1, time, 1]);
        let positions = starts.clone().float() + steps;
        let raw = positions.clone() * self.freqs.clone();
        let phases = (raw.clone() - raw.floor()) * (2.0 * PI);
        self.rope(phases, values)
    }

    fn build_freqs(latent: usize, theta: f32, device: &B::Device) -> Tensor<B, 4> {
        let mut data = Vec::with_capacity(latent);
        for idx in 0..latent {
            let quantized = (idx as f32 / 2.0).floor() * 2.0;
            let exponent = quantized / latent as f32;
            let value = 1.0 / theta.powf(exponent) / (2.0 * PI);
            data.push(value);
        }
        Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([1, 1, 1, latent])
    }
}
