use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use burn::data::dataloader::{DataLoader, DataLoaderIterator, Progress};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use rand::SeedableRng;
use rand::prelude::*;
use rand::rngs::StdRng;

use crate::ContextStrategy;
use crate::tokenizer::{SharedTokenizer, TokenizerConfig};

const SHAKESPEARE_URL: &str =
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";

#[derive(Clone, Copy, Debug)]
pub enum ShakespeareSplit {
    Train,
    Val,
}

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
        let cache_dir = cache_dir.as_ref();
        fs::create_dir_all(cache_dir)?;
        let input_path = cache_dir.join("tinyshakespeare.txt");

        if !input_path.exists() {
            download_shakespeare(&input_path)?;
        }

        let text = fs::read_to_string(&input_path)?;

        let split_ratio = train_split_ratio.clamp(0.0, 1.0);

        let tokenizer_path = tokenizer_cfg.storage_path(cache_dir);
        let tokenizer = if let Some(path) = tokenizer_path {
            if path.is_file() {
                tokenizer_cfg
                    .load(&path)
                    .map_err(|err| io::Error::new(io::ErrorKind::Other, err))?
            } else {
                let tokenizer = tokenizer_cfg
                    .fit(std::iter::once(text.as_str()))
                    .map_err(|err| io::Error::new(io::ErrorKind::Other, err))?;
                tokenizer_cfg
                    .save(&*tokenizer, &path)
                    .map_err(|err| io::Error::new(io::ErrorKind::Other, err))?;
                tokenizer
            }
        } else {
            tokenizer_cfg
                .fit(std::iter::once(text.as_str()))
                .map_err(|err| io::Error::new(io::ErrorKind::Other, err))?
        };

        tokenizer_cfg
            .validate_corpus(&*tokenizer, text.as_str())
            .map_err(|err| io::Error::new(io::ErrorKind::Other, err))?;

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

        let dataset = Self {
            tokens,
            train_len,
            block_size,
            batch_size,
            train_split_ratio: split_ratio,
            tokenizer: tokenizer.clone(),
        };

        Ok(dataset)
    }

    fn split_offset_and_span(&self, split: ShakespeareSplit) -> (usize, usize) {
        match split {
            ShakespeareSplit::Train => (0, self.train_len),
            ShakespeareSplit::Val => {
                let remaining = self.tokens.len().saturating_sub(self.train_len);
                if remaining <= self.block_size + 1 {
                    (0, self.train_len)
                } else {
                    (self.train_len, remaining)
                }
            }
        }
    }

    pub fn steps_per_epoch(&self, split: ShakespeareSplit) -> usize {
        let (_offset, span) = self.split_offset_and_span(split);
        let tokens_per_step = self.block_size * self.batch_size;
        if tokens_per_step == 0 {
            return 1;
        }
        let steps = span.div_ceil(tokens_per_step);
        steps.max(1)
    }

    pub fn sample_batch<B: Backend>(
        &self,
        split: ShakespeareSplit,
        device: &B::Device,
    ) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>, Vec<bool>, Vec<usize>) {
        let (offset, span) = self.split_offset_and_span(split);

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
                inputs[batch_idx * self.block_size + t] = self.tokens[data_idx] as i64;
                targets[batch_idx * self.block_size + t] = self.tokens[data_idx + 1] as i64;
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

        let resets = vec![true; self.batch_size];
        let stream_ids: Vec<usize> = (0..self.batch_size).collect();

        (inputs_tensor, targets_tensor, resets, stream_ids)
    }

    pub fn decode(&self, tokens: &[i64]) -> String {
        let ids: Vec<u32> = tokens
            .iter()
            .filter_map(|&tok| (tok >= 0).then(|| tok as u32))
            .collect();
        self.tokenizer.decode(&ids)
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
}

#[derive(Clone)]
pub struct ShakespeareBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
    pub resets: Vec<bool>,
    pub stream_ids: Vec<usize>,
}

impl<B: Backend> ShakespeareBatch<B> {
    fn new(
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        resets: Vec<bool>,
        stream_ids: Vec<usize>,
    ) -> Self {
        let batch = inputs.shape().dims::<2>()[0];
        let stream_count = stream_ids.len();
        debug_assert_eq!(
            batch, stream_count,
            "tbptt stream id count must match batch rows"
        );
        debug_assert_eq!(
            stream_count,
            resets.len(),
            "tbptt reset count must match stream ids"
        );
        Self {
            inputs,
            targets,
            resets,
            stream_ids,
        }
    }
}

pub struct ShakespeareRandomDataLoader<B: Backend> {
    dataset: Arc<ShakespeareDataset>,
    split: ShakespeareSplit,
    device: B::Device,
    steps_per_epoch: usize,
    total_steps: Option<usize>,
    consumed_steps: Option<Arc<AtomicUsize>>,
}

impl<B: Backend> Clone for ShakespeareRandomDataLoader<B> {
    fn clone(&self) -> Self {
        Self {
            dataset: Arc::clone(&self.dataset),
            split: self.split,
            device: self.device.clone(),
            steps_per_epoch: self.steps_per_epoch,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
        }
    }
}

impl<B: Backend> ShakespeareRandomDataLoader<B> {
    pub fn new(
        dataset: Arc<ShakespeareDataset>,
        split: ShakespeareSplit,
        device: &B::Device,
        steps_per_epoch: usize,
        total_steps: Option<usize>,
    ) -> Self {
        let steps_per_epoch = steps_per_epoch.max(1);
        let total_steps = total_steps.filter(|value| *value > 0);
        let consumed_steps = total_steps.as_ref().map(|_| Arc::new(AtomicUsize::new(0)));

        Self {
            dataset,
            split,
            device: device.clone(),
            steps_per_epoch,
            total_steps,
            consumed_steps,
        }
    }
}

impl<B> DataLoader<B, ShakespeareBatch<B>> for ShakespeareRandomDataLoader<B>
where
    B: Backend + 'static,
    B::Device: Clone,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<ShakespeareBatch<B>> + 'a> {
        let steps_total =
            if let (Some(limit), Some(consumed)) = (self.total_steps, &self.consumed_steps) {
                let used = consumed.load(Ordering::Relaxed);
                if used >= limit {
                    0
                } else {
                    (limit - used).min(self.steps_per_epoch)
                }
            } else {
                self.steps_per_epoch
            };

        Box::new(ShakespeareRandomIterator {
            dataset: Arc::clone(&self.dataset),
            split: self.split,
            device: self.device.clone(),
            steps_total,
            step: 0,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.clone(),
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
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
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
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
        })
    }
}

struct ShakespeareRandomIterator<B: Backend> {
    dataset: Arc<ShakespeareDataset>,
    split: ShakespeareSplit,
    device: B::Device,
    steps_total: usize,
    step: usize,
    total_steps: Option<usize>,
    consumed_steps: Option<Arc<AtomicUsize>>,
}

impl<B: Backend> Iterator for ShakespeareRandomIterator<B> {
    type Item = ShakespeareBatch<B>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.step >= self.steps_total {
            return None;
        }
        self.step += 1;

        if let Some(counter) = &self.consumed_steps {
            if let Some(limit) = self.total_steps {
                let previous = counter.fetch_add(1, Ordering::Relaxed);
                if previous >= limit {
                    return None;
                }
            } else {
                counter.fetch_add(1, Ordering::Relaxed);
            }
        }

        let (inputs, targets, resets, stream_ids) =
            self.dataset.sample_batch::<B>(self.split, &self.device);

        Some(ShakespeareBatch::new(inputs, targets, resets, stream_ids))
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

pub struct ShakespeareStreamDataLoader<B: Backend> {
    dataset: Arc<ShakespeareDataset>,
    split: ShakespeareSplit,
    device: B::Device,
    steps_per_epoch: usize,
    total_steps: Option<usize>,
    consumed_steps: Option<Arc<AtomicUsize>>,
    positions: Arc<Mutex<Vec<usize>>>,
    tokens_since_reset: Arc<Mutex<Vec<usize>>>,
    pending_reset: Arc<Mutex<Vec<bool>>>,
    rng: Arc<Mutex<StdRng>>,
    initialized: Arc<AtomicBool>,
    context_window: Option<usize>,
}

impl<B: Backend> Clone for ShakespeareStreamDataLoader<B> {
    fn clone(&self) -> Self {
        Self {
            dataset: Arc::clone(&self.dataset),
            split: self.split,
            device: self.device.clone(),
            steps_per_epoch: self.steps_per_epoch,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
            positions: Arc::clone(&self.positions),
            tokens_since_reset: Arc::clone(&self.tokens_since_reset),
            pending_reset: Arc::clone(&self.pending_reset),
            rng: Arc::clone(&self.rng),
            initialized: Arc::clone(&self.initialized),
            context_window: self.context_window,
        }
    }
}

impl<B: Backend> ShakespeareStreamDataLoader<B> {
    pub fn new(
        dataset: Arc<ShakespeareDataset>,
        split: ShakespeareSplit,
        device: &B::Device,
        steps_per_epoch: usize,
        total_steps: Option<usize>,
        strategy: ContextStrategy,
    ) -> Self {
        let steps_per_epoch = steps_per_epoch.max(1);
        let total_steps = total_steps.filter(|value| *value > 0);
        let consumed_steps = total_steps.as_ref().map(|_| Arc::new(AtomicUsize::new(0)));
        let batch_size = dataset.batch_size;
        let block_size = dataset.block_size;
        let positions = Arc::new(Mutex::new(vec![0; batch_size]));
        let tokens_since_reset = Arc::new(Mutex::new(vec![0; batch_size]));
        let pending_reset = Arc::new(Mutex::new(vec![true; batch_size]));
        let rng = Arc::new(Mutex::new(StdRng::seed_from_u64(0xBAD5EED)));
        let initialized = Arc::new(AtomicBool::new(false));
        let context_window = match strategy {
            ContextStrategy::Infinite => None,
            ContextStrategy::Sliding { window } => Some(window.max(block_size).max(1)),
        };

        Self {
            dataset,
            split,
            device: device.clone(),
            steps_per_epoch,
            total_steps,
            consumed_steps,
            positions,
            tokens_since_reset,
            pending_reset,
            rng,
            initialized,
            context_window,
        }
    }
}

impl<B> DataLoader<B, ShakespeareBatch<B>> for ShakespeareStreamDataLoader<B>
where
    B: Backend + 'static,
    B::Device: Clone,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<ShakespeareBatch<B>> + 'a> {
        let steps_total =
            if let (Some(limit), Some(consumed)) = (self.total_steps, &self.consumed_steps) {
                let used = consumed.load(Ordering::Relaxed);
                if used >= limit {
                    0
                } else {
                    (limit - used).min(self.steps_per_epoch)
                }
            } else {
                self.steps_per_epoch
            };

        Box::new(ShakespeareStreamIterator {
            dataset: Arc::clone(&self.dataset),
            split: self.split,
            device: self.device.clone(),
            steps_total,
            step: 0,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.clone(),
            positions: Arc::clone(&self.positions),
            tokens_since_reset: Arc::clone(&self.tokens_since_reset),
            pending_reset: Arc::clone(&self.pending_reset),
            rng: Arc::clone(&self.rng),
            initialized: Arc::clone(&self.initialized),
            context_window: self.context_window,
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
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
            positions: Arc::clone(&self.positions),
            tokens_since_reset: Arc::clone(&self.tokens_since_reset),
            pending_reset: Arc::clone(&self.pending_reset),
            rng: Arc::clone(&self.rng),
            initialized: Arc::clone(&self.initialized),
            context_window: self.context_window,
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
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
            positions: Arc::clone(&self.positions),
            tokens_since_reset: Arc::clone(&self.tokens_since_reset),
            pending_reset: Arc::clone(&self.pending_reset),
            rng: Arc::clone(&self.rng),
            initialized: Arc::clone(&self.initialized),
            context_window: self.context_window,
        })
    }
}

struct ShakespeareStreamIterator<B: Backend> {
    dataset: Arc<ShakespeareDataset>,
    split: ShakespeareSplit,
    device: B::Device,
    steps_total: usize,
    step: usize,
    total_steps: Option<usize>,
    consumed_steps: Option<Arc<AtomicUsize>>,
    positions: Arc<Mutex<Vec<usize>>>,
    tokens_since_reset: Arc<Mutex<Vec<usize>>>,
    pending_reset: Arc<Mutex<Vec<bool>>>,
    rng: Arc<Mutex<StdRng>>,
    initialized: Arc<AtomicBool>,
    context_window: Option<usize>,
}

impl<B: Backend> Iterator for ShakespeareStreamIterator<B> {
    type Item = ShakespeareBatch<B>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.step >= self.steps_total {
            return None;
        }
        self.step += 1;

        if let Some(counter) = &self.consumed_steps {
            if let Some(limit) = self.total_steps {
                let previous = counter.fetch_add(1, Ordering::Relaxed);
                if previous >= limit {
                    return None;
                }
            } else {
                counter.fetch_add(1, Ordering::Relaxed);
            }
        }

        let batch_size = self.dataset.batch_size;
        let block_size = self.dataset.block_size;
        let (offset, span) = self.dataset.split_offset_and_span(self.split);

        {
            if !self.initialized.swap(true, Ordering::Relaxed) {
                let mut positions = self.positions.lock().expect("positions poisoned");
                let mut tokens_since_reset = self
                    .tokens_since_reset
                    .lock()
                    .expect("tokens_since_reset poisoned");
                let mut pending_reset = self.pending_reset.lock().expect("pending_reset poisoned");

                let stream_span = span.saturating_sub(block_size + 1);
                let stride = (stream_span / batch_size.max(1)).max(1);
                for (idx, pos) in positions.iter_mut().enumerate() {
                    let start_offset = (idx * stride).min(stream_span);
                    *pos = offset + start_offset;
                }
                for token_count in tokens_since_reset.iter_mut() {
                    *token_count = 0;
                }
                pending_reset.fill(true);
            }
        }

        let mut positions = self.positions.lock().expect("positions poisoned");
        let mut tokens_since_reset = self
            .tokens_since_reset
            .lock()
            .expect("tokens_since_reset poisoned");
        let mut pending_reset = self.pending_reset.lock().expect("pending_reset poisoned");
        let mut rng = self.rng.lock().expect("rng poisoned");

        let mut inputs = Vec::with_capacity(batch_size * block_size);
        let mut targets = Vec::with_capacity(batch_size * block_size);
        let mut resets = Vec::with_capacity(batch_size);
        let mut stream_ids = Vec::with_capacity(batch_size);

        for stream_idx in 0..batch_size {
            let mut reset_flag = pending_reset[stream_idx];
            if let Some(window) = self.context_window {
                if window > 0 && tokens_since_reset[stream_idx] >= window {
                    reset_flag = true;
                    tokens_since_reset[stream_idx] = 0;
                }
            }

            let start = positions[stream_idx];
            let end = start + block_size;

            for t in 0..block_size {
                let data_idx = start + t;
                inputs.push(self.dataset.tokens[data_idx] as i64);
                targets.push(self.dataset.tokens[data_idx + 1] as i64);
            }

            positions[stream_idx] = end;
            tokens_since_reset[stream_idx] += block_size;

            let limit = offset + span - (block_size + 1);
            let mut next_reset = false;
            if positions[stream_idx] > limit {
                let new_start = rng.gen_range(offset..=limit);
                positions[stream_idx] = new_start;
                tokens_since_reset[stream_idx] = 0;
                next_reset = true;
            } else if let Some(window) = self.context_window {
                if window > 0 && tokens_since_reset[stream_idx] >= window {
                    tokens_since_reset[stream_idx] = 0;
                    next_reset = true;
                }
            }

            pending_reset[stream_idx] = next_reset;
            resets.push(reset_flag);
            stream_ids.push(stream_idx);
        }

        let inputs_tensor = Tensor::<B, 2, Int>::from_data(
            TensorData::new(inputs, [batch_size, block_size]),
            &self.device,
        );
        let targets_tensor = Tensor::<B, 2, Int>::from_data(
            TensorData::new(targets, [batch_size, block_size]),
            &self.device,
        );

        Some(ShakespeareBatch::new(
            inputs_tensor,
            targets_tensor,
            resets,
            stream_ids,
        ))
    }
}

impl<B: Backend> DataLoaderIterator<ShakespeareBatch<B>> for ShakespeareStreamIterator<B> {
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
