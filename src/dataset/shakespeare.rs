use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::Path;

use burn::tensor::backend::Backend;

use super::DatasetSplit;
use super::scheduler::{SequenceBatch, TokenSequenceDataset};
use crate::tokenizer::{SharedTokenizer, TokenizerConfig};

const SHAKESPEARE_URL: &str =
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";

#[derive(Clone)]
pub struct ShakespeareDataset {
    tokens: Vec<u32>,
    train_len: usize,
    block_size: usize,
    batch_size: usize,
    train_split_ratio: f32,
    tokenizer: SharedTokenizer,
}

impl ShakespeareDataset {
    pub fn new(
        cache_dir: impl AsRef<Path>,
        block_size: usize,
        batch_size: usize,
        train_split_ratio: f32,
        tokenizer_cfg: &TokenizerConfig,
    ) -> io::Result<Self> {
        Self::new_with_source(
            cache_dir,
            block_size,
            batch_size,
            train_split_ratio,
            tokenizer_cfg,
            None,
        )
    }

    pub fn new_with_source(
        cache_dir: impl AsRef<Path>,
        block_size: usize,
        batch_size: usize,
        train_split_ratio: f32,
        tokenizer_cfg: &TokenizerConfig,
        source_url: Option<&str>,
    ) -> io::Result<Self> {
        let cache_dir = cache_dir.as_ref();
        fs::create_dir_all(cache_dir)?;
        let input_path = cache_dir.join("tinyshakespeare.txt");

        if !input_path.exists() {
            download_shakespeare(&input_path, source_url)?;
        }

        let text = fs::read_to_string(&input_path)?;

        let split_ratio = train_split_ratio.clamp(0.0, 1.0);

        let tokenizer_path = tokenizer_cfg.storage_path(cache_dir);
        let tokenizer = if let Some(path) = tokenizer_path {
            if path.is_file() {
                tokenizer_cfg.load(&path).map_err(io::Error::other)?
            } else {
                let tokenizer = tokenizer_cfg
                    .fit(std::iter::once(text.as_str()))
                    .map_err(io::Error::other)?;
                tokenizer_cfg
                    .save(&*tokenizer, &path)
                    .map_err(io::Error::other)?;
                tokenizer
            }
        } else {
            tokenizer_cfg
                .fit(std::iter::once(text.as_str()))
                .map_err(io::Error::other)?
        };

        tokenizer_cfg
            .validate_corpus(&*tokenizer, text.as_str())
            .map_err(io::Error::other)?;

        let tokens = tokenizer.encode(text.as_str(), false, false);
        if tokens.len() <= block_size + 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "encoded dataset smaller than block size",
            ));
        }

        let mut train_len = ((tokens.len() as f32) * split_ratio) as usize;
        let min_len = block_size + 1;
        let max_len = tokens.len() - 1;
        if train_len < min_len {
            train_len = min_len;
        } else if train_len > max_len {
            train_len = max_len;
        }

        Ok(Self {
            tokens,
            train_len,
            block_size,
            batch_size,
            train_split_ratio: split_ratio,
            tokenizer: tokenizer.clone(),
        })
    }

    pub fn tokenizer(&self) -> SharedTokenizer {
        self.tokenizer.clone()
    }

    pub fn train_split_ratio(&self) -> f32 {
        self.train_split_ratio
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub fn train_len(&self) -> usize {
        self.train_len
    }

    pub fn steps_per_epoch(&self, split: DatasetSplit) -> usize {
        TokenSequenceDataset::steps_per_epoch(self, split)
    }

    pub fn sample_batch<B: Backend>(
        &self,
        split: DatasetSplit,
        device: &B::Device,
    ) -> SequenceBatch<B> {
        super::scheduler::sample_batch(self, split, device)
    }

    pub fn decode(&self, tokens: &[i64]) -> String {
        TokenSequenceDataset::decode(self, tokens)
    }
}

impl TokenSequenceDataset for ShakespeareDataset {
    fn tokenizer(&self) -> SharedTokenizer {
        self.tokenizer.clone()
    }

    fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    fn train_len(&self) -> usize {
        self.train_len
    }

    fn block_size(&self) -> usize {
        self.block_size
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn train_split_ratio(&self) -> f32 {
        self.train_split_ratio
    }
}

fn download_shakespeare(path: &Path, source_url: Option<&str>) -> io::Result<()> {
    let url = source_url.unwrap_or(SHAKESPEARE_URL);
    let response = ureq::get(url)
        .call()
        .map_err(|err| io::Error::other(err.to_string()))?;

    let mut reader = response.into_reader();
    let mut contents = Vec::new();
    reader.read_to_end(&mut contents)?;

    let mut file = File::create(path)?;
    file.write_all(&contents)?;
    Ok(())
}
