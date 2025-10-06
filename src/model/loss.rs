use burn::nn::loss::CrossEntropyLossConfig;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

pub fn language_model_loss<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
) -> Tensor<B, 1> {
    let [batch, time, vocab] = logits.shape().dims();

    let logits_flat = logits.reshape([batch * time, vocab]);
    let targets_flat = targets.reshape([batch * time]);

    let device = logits_flat.device();
    CrossEntropyLossConfig::new()
        .init::<B>(&device)
        .forward(logits_flat, targets_flat)
}
