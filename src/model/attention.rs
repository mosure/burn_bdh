use std::f32::consts::PI;

use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

const DEFAULT_THETA: f32 = 65_536.0;

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    freqs: Tensor<B, 4>,
    n_head: usize,
}

impl<B: Backend> Attention<B> {
    pub fn new(latent: usize, n_head: usize, device: &B::Device) -> Self {
        Self {
            freqs: Self::build_freqs(latent, device),
            n_head,
        }
    }

    pub fn forward(&self, query: Tensor<B, 4>, value: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_batch, _heads, time, _dim] = query.shape().dims();
        let device = self.freqs.device();
        let positions = Tensor::<B, 1, Int>::arange(0..time as i64, &device)
            .float()
            .reshape([1, 1, time, 1]);

        let raw = positions * self.freqs.clone();
        let phases = (raw.clone() - raw.floor()) * (2.0 * PI);
        let q_rot = self.rope(phases.clone(), query);
        let k_rot = q_rot.clone();

        let scores = q_rot.matmul(k_rot.swap_dims(2, 3)).tril(-1);
        let value = value.repeat_dim(1, self.n_head);

        scores.matmul(value)
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

    fn build_freqs(latent: usize, device: &B::Device) -> Tensor<B, 4> {
        let mut data = Vec::with_capacity(latent);
        for idx in 0..latent {
            let quantized = (idx as f32 / 2.0).floor() * 2.0;
            let exponent = quantized / latent as f32;
            let value = 1.0 / DEFAULT_THETA.powf(exponent) / (2.0 * PI);
            data.push(value);
        }
        Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([1, 1, 1, latent])
    }
}
