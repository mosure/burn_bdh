use std::fs;

use burn::LearningRate;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend as BackendTrait;
use burn_autodiff::Autodiff;
use burn_dragon_hatchling::dataset::{ShakespeareDataset, ShakespeareSplit};
use burn_dragon_hatchling::tokenizer::TokenizerConfig;
use burn_dragon_hatchling::{BDH, BDHConfig, language_model_loss};
use burn_ndarray::NdArray;
use tempfile::tempdir;

#[test]
fn single_training_step_executes() {
    let dir = tempdir().expect("tempdir");
    let cache_dir = dir.path();
    let file_path = cache_dir.join("tinyshakespeare.txt");
    let content =
        b"All the world's a stage, and all the men and women merely players.\n".repeat(256);
    fs::write(&file_path, content).expect("write dataset");

    let block_size = 32;
    let batch_size = 4;
    let tokenizer = TokenizerConfig::default();
    let dataset = ShakespeareDataset::new(cache_dir, block_size, batch_size, 0.9, &tokenizer)
        .expect("dataset");

    type Backend = Autodiff<NdArray<f32>>;
    <Backend as BackendTrait>::seed(123);
    let device = <Backend as BackendTrait>::Device::default();

    let mut model_config = BDHConfig::default();
    let vocab = dataset.tokenizer();
    model_config.vocab_size = vocab.len();
    let model = BDH::<Backend>::new(model_config, &device);
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(0.1)
        .init::<Backend, BDH<Backend>>();
    let lr: LearningRate = 1e-3;

    let (inputs, targets) = dataset.sample_batch::<Backend>(ShakespeareSplit::Train, &device);

    let logits = model.forward(inputs);
    let loss = language_model_loss::<Backend>(logits, targets);
    let loss_scalar = loss
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("loss to vec")[0];
    assert!(loss_scalar.is_finite());

    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &model);
    let _ = optimizer.step(lr, model, grads);
}
