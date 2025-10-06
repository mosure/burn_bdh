use burn::tensor::Tensor;
use burn::tensor::activation;
use burn::tensor::backend::Backend;

use super::block_sparse::BlockPattern1d;

pub fn fused_forward<B: Backend>(
    input: Tensor<B, 4>,
    weight: Tensor<B, 4>,
    bias: Option<Tensor<B, 3>>,
    threshold: f32,
    layout: &BlockPattern1d,
) -> Tensor<B, 4> {
    let device = input.device();
    let latent = weight.shape().dims::<4>()[3];

    let mut projected = input.matmul(weight);

    if let Some(bias) = bias {
        let dims = bias.shape().dims::<3>();
        let bias = bias.reshape([1, dims[0], 1, dims[2]]);
        projected = projected + bias;
    }

    if threshold != 0.0 {
        projected = projected.sub_scalar(threshold);
    }

    let mut activated = activation::relu(projected);

    if layout.is_sparse() {
        let mask = layout.mask::<B>(latent, &device);
        activated = activated * mask;
    }

    activated
}
