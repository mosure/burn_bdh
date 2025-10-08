use std::fs;

use burn::tensor::backend::Backend as BackendTrait;
use burn_dragon_hatchling::dataset::{ShakespeareDataset, ShakespeareSplit};
use burn_dragon_hatchling::tokenizer::{ByteTokenizerConfig, TokenizerConfig, TokenizerKind};
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
    let tokenizer = TokenizerConfig::default();
    let dataset = ShakespeareDataset::new(cache_dir, block_size, batch_size, 0.8, &tokenizer)
        .expect("create dataset");

    type Backend = NdArray<f32>;
    <Backend as BackendTrait>::seed(0);
    let device = <Backend as BackendTrait>::Device::default();

    let batch = dataset.sample_batch::<Backend>(ShakespeareSplit::Train, &device);
    assert_eq!(batch.inputs.shape().dims(), [batch_size, block_size]);
    assert_eq!(batch.targets.shape().dims(), [batch_size, block_size]);
}

#[test]
fn byte_tokenizer_dataset_initializes() {
    let dir = tempdir().expect("tempdir");
    let cache_dir = dir.path();
    let file_path = cache_dir.join("tinyshakespeare.txt");
    let content = b"The quick brown fox jumps over the lazy dog. ".repeat(128);
    fs::write(&file_path, content).expect("write dataset");

    let tokenizer = TokenizerConfig {
        vocab_path: None,
        kind: TokenizerKind::Byte(ByteTokenizerConfig::default()),
    };

    let dataset = ShakespeareDataset::new(cache_dir, 16, 2, 0.75, &tokenizer)
        .expect("create dataset with byte tokenizer");
    let tokenizer = dataset.tokenizer();
    assert!(tokenizer.len() >= 256);
}
