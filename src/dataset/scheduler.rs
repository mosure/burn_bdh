use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use burn::data::dataloader::{DataLoader, DataLoaderIterator, Progress};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use rand::prelude::*;

use crate::tokenizer::SharedTokenizer;

use super::DatasetSplit;

/// Abstraction over text corpora that can be converted into BDH-compatible batches.
pub trait TokenSequenceDataset: Send + Sync {
    /// Return a shared tokenizer handle (cloned per call).
    fn tokenizer(&self) -> SharedTokenizer;

    /// Return the full sequence of token ids representing the corpus.
    fn tokens(&self) -> &[u32];

    /// Number of tokens reserved for the training split from the start of `tokens`.
    fn train_len(&self) -> usize;

    /// Maximum sequence length per sample.
    fn block_size(&self) -> usize;

    /// Number of sequences per batch.
    fn batch_size(&self) -> usize;

    /// Ratio used when determining train/validation split boundaries.
    fn train_split_ratio(&self) -> f32;

    /// Provide the offset and span of the requested split.
    fn split_offset_and_span(&self, split: DatasetSplit) -> (usize, usize) {
        match split {
            DatasetSplit::Train => (0, self.train_len()),
            DatasetSplit::Val => {
                let tokens = self.tokens();
                let train_len = self.train_len();
                let remaining = tokens.len().saturating_sub(train_len);
                if remaining <= self.block_size() + 1 {
                    (0, train_len)
                } else {
                    (train_len, remaining)
                }
            }
        }
    }

    /// Number of steps per epoch for a given split (defaults derived from token counts).
    fn steps_per_epoch(&self, split: DatasetSplit) -> usize {
        let (_offset, span) = self.split_offset_and_span(split);
        let tokens_per_step = self.block_size() * self.batch_size();
        if tokens_per_step == 0 {
            return 1;
        }
        let steps = span.div_ceil(tokens_per_step);
        steps.max(1)
    }

    /// Decode token ids back into text.
    fn decode(&self, tokens: &[i64]) -> String {
        let ids: Vec<u32> = tokens
            .iter()
            .filter_map(|&tok| (tok >= 0).then_some(tok as u32))
            .collect();
        self.tokenizer().decode(&ids)
    }
}

/// Sample a random batch from any dataset implementing [`TokenSequenceDataset`].
pub fn sample_batch<B: Backend, T: TokenSequenceDataset + ?Sized>(
    dataset: &T,
    split: DatasetSplit,
    device: &B::Device,
) -> SequenceBatch<B> {
    let tokens = dataset.tokens();
    let (offset, span) = dataset.split_offset_and_span(split);

    let mut rng = thread_rng();
    let mut inputs = vec![0i64; dataset.batch_size() * dataset.block_size()];
    let mut targets = vec![0i64; dataset.batch_size() * dataset.block_size()];

    for batch_idx in 0..dataset.batch_size() {
        let max_start = span.saturating_sub(dataset.block_size() + 1);
        let start_offset = if max_start == 0 {
            0
        } else {
            rng.gen_range(0..=max_start)
        };
        let start = offset + start_offset;
        for t in 0..dataset.block_size() {
            let data_idx = start + t;
            inputs[batch_idx * dataset.block_size() + t] = tokens[data_idx] as i64;
            targets[batch_idx * dataset.block_size() + t] = tokens[data_idx + 1] as i64;
        }
    }

    let inputs_tensor = Tensor::<B, 2, Int>::from_data(
        TensorData::new(inputs, [dataset.batch_size(), dataset.block_size()]),
        device,
    );
    let targets_tensor = Tensor::<B, 2, Int>::from_data(
        TensorData::new(targets, [dataset.batch_size(), dataset.block_size()]),
        device,
    );

    SequenceBatch::new(inputs_tensor, targets_tensor)
}

/// Batched token inputs and targets for language modeling.
#[derive(Clone)]
pub struct SequenceBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> SequenceBatch<B> {
    pub fn new(inputs: Tensor<B, 2, Int>, targets: Tensor<B, 2, Int>) -> Self {
        Self { inputs, targets }
    }
}

/// Data loader that produces random sequences from any `TokenSequenceDataset`.
pub struct RandomDataLoader<B: Backend> {
    dataset: Arc<dyn TokenSequenceDataset>,
    split: DatasetSplit,
    device: B::Device,
    steps_per_epoch: usize,
    total_steps: Option<usize>,
    consumed_steps: Option<Arc<AtomicUsize>>,
}

impl<B: Backend> Clone for RandomDataLoader<B> {
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

impl<B: Backend> RandomDataLoader<B> {
    pub fn new<T>(
        dataset: Arc<T>,
        split: DatasetSplit,
        device: &B::Device,
        steps_per_epoch: usize,
        total_steps: Option<usize>,
    ) -> Self
    where
        T: TokenSequenceDataset + 'static,
    {
        let dataset: Arc<dyn TokenSequenceDataset> = dataset;
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

impl<B> DataLoader<B, SequenceBatch<B>> for RandomDataLoader<B>
where
    B: Backend + 'static,
    B::Device: Clone,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<SequenceBatch<B>> + 'a> {
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

        Box::new(RandomIterator {
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

    fn to_device(&self, device: &B::Device) -> Arc<dyn DataLoader<B, SequenceBatch<B>>> {
        Arc::new(Self {
            dataset: Arc::clone(&self.dataset),
            split: self.split,
            device: device.clone(),
            steps_per_epoch: self.steps_per_epoch,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
        })
    }

    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<B, SequenceBatch<B>>> {
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

struct RandomIterator<B: Backend> {
    dataset: Arc<dyn TokenSequenceDataset>,
    split: DatasetSplit,
    device: B::Device,
    steps_total: usize,
    step: usize,
    total_steps: Option<usize>,
    consumed_steps: Option<Arc<AtomicUsize>>,
}

impl<B: Backend> Iterator for RandomIterator<B> {
    type Item = SequenceBatch<B>;

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

        Some(sample_batch::<B, _>(
            &*self.dataset,
            self.split,
            &self.device,
        ))
    }
}

impl<B: Backend> DataLoaderIterator<SequenceBatch<B>> for RandomIterator<B> {
    fn progress(&self) -> Progress {
        Progress::new(
            self.step * self.dataset.batch_size(),
            self.steps_total * self.dataset.batch_size(),
        )
    }
}
