use std::fs;

use burn::tensor::backend::Backend as BackendTrait;
use burn_dragon_hatchling::dataset::{ShakespeareDataset, ShakespeareSplit};
use burn_ndarray::NdArray;
use tempfile::tempdir;

#[test]
fn dataset_batches_match_expected_shape() {
    let dir = tempdir().expect("tempdir");
    let cache_dir = dir.path();
    let file_path = cache_dir.join("tinyshakespeare.txt");
    let content = b"The quick brown fox jumps over the lazy dog. ".repeat(512);
    fs::write(&file_path, content).expect("write dataset");

    let block_size = 32;
    let batch_size = 4;
    let dataset =
        ShakespeareDataset::new(cache_dir, block_size, batch_size, 0.8).expect("create dataset");

    type Backend = NdArray<f32>;
    <Backend as BackendTrait>::seed(0);
    let device = <Backend as BackendTrait>::Device::default();

    let (inputs, targets) = dataset.sample_batch::<Backend>(ShakespeareSplit::Train, &device);
    assert_eq!(inputs.shape().dims(), [batch_size, block_size]);
    assert_eq!(targets.shape().dims(), [batch_size, block_size]);
}
