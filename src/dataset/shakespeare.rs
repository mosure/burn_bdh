use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::Path;
use std::sync::Arc;

use burn::data::dataloader::{DataLoader, DataLoaderIterator, Progress};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use rand::prelude::*;

const SHAKESPEARE_URL: &str =
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";

#[derive(Clone, Copy, Debug)]
pub enum ShakespeareSplit {
    Train,
    Val,
}

#[derive(Clone)]
pub struct ShakespeareDataset {
    data: Vec<u8>,
    train_len: usize,
    block_size: usize,
    batch_size: usize,
    train_split_ratio: f32,
}

impl ShakespeareDataset {
    pub fn new(
        cache_dir: impl AsRef<Path>,
        block_size: usize,
        batch_size: usize,
        train_split_ratio: f32,
    ) -> io::Result<Self> {
        let cache_dir = cache_dir.as_ref();
        fs::create_dir_all(cache_dir)?;
        let input_path = cache_dir.join("tinyshakespeare.txt");

        if !input_path.exists() {
            download_shakespeare(&input_path)?;
        }

        let data = fs::read(&input_path)?;
        if data.len() <= block_size + 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "dataset content smaller than block size",
            ));
        }

        let mut train_len = ((data.len() as f32) * train_split_ratio.clamp(0.0, 1.0)) as usize;
        let min_len = block_size + 1;
        let max_len = data.len() - 1;
        if train_len < min_len {
            train_len = min_len;
        } else if train_len > max_len {
            train_len = max_len;
        }

        Ok(Self {
            data,
            train_len,
            block_size,
            batch_size,
            train_split_ratio: train_split_ratio.clamp(0.0, 1.0),
        })
    }

    pub fn sample_batch<B: Backend>(
        &self,
        split: ShakespeareSplit,
        device: &B::Device,
    ) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
        let (offset, span) = match split {
            ShakespeareSplit::Train => (0, self.train_len),
            ShakespeareSplit::Val => {
                let remaining = self.data.len().saturating_sub(self.train_len);
                if remaining <= self.block_size + 1 {
                    (0, self.train_len)
                } else {
                    (self.train_len, remaining)
                }
            }
        };

        let mut rng = thread_rng();
        let mut inputs = vec![0i64; self.batch_size * self.block_size];
        let mut targets = vec![0i64; self.batch_size * self.block_size];

        for batch_idx in 0..self.batch_size {
            let max_start = span.saturating_sub(self.block_size + 1);
            let start_offset = if max_start == 0 {
                0
            } else {
                rng.gen_range(0..=max_start)
            };
            let start = offset + start_offset;
            for t in 0..self.block_size {
                let data_idx = start + t;
                inputs[batch_idx * self.block_size + t] = self.data[data_idx] as i64;
                targets[batch_idx * self.block_size + t] = self.data[data_idx + 1] as i64;
            }
        }

        let inputs_tensor = Tensor::<B, 2, Int>::from_data(
            TensorData::new(inputs, [self.batch_size, self.block_size]),
            device,
        );
        let targets_tensor = Tensor::<B, 2, Int>::from_data(
            TensorData::new(targets, [self.batch_size, self.block_size]),
            device,
        );

        (inputs_tensor, targets_tensor)
    }

    pub fn decode(&self, tokens: &[i64]) -> String {
        let bytes: Vec<u8> = tokens.iter().map(|&tok| tok as u8).collect();
        String::from_utf8_lossy(&bytes).to_string()
    }

    pub fn train_split_ratio(&self) -> f32 {
        self.train_split_ratio
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

#[derive(Clone)]
pub struct ShakespeareBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> ShakespeareBatch<B> {
    fn new(inputs: Tensor<B, 2, Int>, targets: Tensor<B, 2, Int>) -> Self {
        Self { inputs, targets }
    }
}

#[derive(Clone)]
pub struct ShakespeareRandomDataLoader<B: Backend> {
    dataset: Arc<ShakespeareDataset>,
    split: ShakespeareSplit,
    device: B::Device,
    steps_per_epoch: usize,
}

impl<B: Backend> ShakespeareRandomDataLoader<B> {
    pub fn new(
        dataset: Arc<ShakespeareDataset>,
        split: ShakespeareSplit,
        device: &B::Device,
        steps_per_epoch: usize,
    ) -> Self {
        Self {
            dataset,
            split,
            device: device.clone(),
            steps_per_epoch: steps_per_epoch.max(1),
        }
    }
}

impl<B> DataLoader<B, ShakespeareBatch<B>> for ShakespeareRandomDataLoader<B>
where
    B: Backend + 'static,
    B::Device: Clone,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<ShakespeareBatch<B>> + 'a> {
        Box::new(ShakespeareRandomIterator {
            dataset: Arc::clone(&self.dataset),
            split: self.split,
            device: self.device.clone(),
            steps_total: self.steps_per_epoch,
            step: 0,
        })
    }

    fn num_items(&self) -> usize {
        self.steps_per_epoch * self.dataset.batch_size()
    }

    fn to_device(&self, device: &B::Device) -> Arc<dyn DataLoader<B, ShakespeareBatch<B>>> {
        Arc::new(Self {
            dataset: Arc::clone(&self.dataset),
            split: self.split,
            device: device.clone(),
            steps_per_epoch: self.steps_per_epoch,
        })
    }

    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<B, ShakespeareBatch<B>>> {
        let end = end.min(self.steps_per_epoch);
        let start = start.min(end);
        let steps = (end - start).max(1);

        Arc::new(Self {
            dataset: Arc::clone(&self.dataset),
            split: self.split,
            device: self.device.clone(),
            steps_per_epoch: steps,
        })
    }
}

struct ShakespeareRandomIterator<B: Backend> {
    dataset: Arc<ShakespeareDataset>,
    split: ShakespeareSplit,
    device: B::Device,
    steps_total: usize,
    step: usize,
}

impl<B: Backend> Iterator for ShakespeareRandomIterator<B> {
    type Item = ShakespeareBatch<B>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.step >= self.steps_total {
            return None;
        }
        self.step += 1;

        let (inputs, targets) = self.dataset.sample_batch::<B>(self.split, &self.device);

        Some(ShakespeareBatch::new(inputs, targets))
    }
}

impl<B: Backend> DataLoaderIterator<ShakespeareBatch<B>> for ShakespeareRandomIterator<B> {
    fn progress(&self) -> Progress {
        Progress::new(
            self.step * self.dataset.batch_size(),
            self.steps_total * self.dataset.batch_size(),
        )
    }
}

fn download_shakespeare(path: &Path) -> io::Result<()> {
    let response = ureq::get(SHAKESPEARE_URL)
        .call()
        .map_err(|err| io::Error::other(err.to_string()))?;

    let mut reader = response.into_reader();
    let mut contents = Vec::new();
    reader.read_to_end(&mut contents)?;

    let mut file = File::create(path)?;
    file.write_all(&contents)?;
    Ok(())
}
