mod factory;
mod huggingface;
pub mod scheduler;
mod shakespeare;

use crate::tokenizer::SharedTokenizer;

pub use factory::build_dataset;
pub use huggingface::HuggingFaceDataset;
pub use scheduler::{RandomDataLoader, SequenceBatch, TokenSequenceDataset};
pub use shakespeare::ShakespeareDataset;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DatasetSplit {
    Train,
    Val,
}

#[derive(Clone)]
pub enum Dataset {
    Shakespeare(ShakespeareDataset),
    HuggingFace(HuggingFaceDataset),
}

impl Dataset {
    pub fn from_shakespeare(dataset: ShakespeareDataset) -> Self {
        Self::Shakespeare(dataset)
    }

    pub fn from_huggingface(dataset: HuggingFaceDataset) -> Self {
        Self::HuggingFace(dataset)
    }

    pub fn tokenizer(&self) -> SharedTokenizer {
        TokenSequenceDataset::tokenizer(self)
    }

    pub fn train_split_ratio(&self) -> f32 {
        TokenSequenceDataset::train_split_ratio(self)
    }

    pub fn batch_size(&self) -> usize {
        TokenSequenceDataset::batch_size(self)
    }

    pub fn steps_per_epoch(&self, split: DatasetSplit) -> usize {
        TokenSequenceDataset::steps_per_epoch(self, split)
    }
}

impl TokenSequenceDataset for Dataset {
    fn tokenizer(&self) -> SharedTokenizer {
        match self {
            Dataset::Shakespeare(dataset) => dataset.tokenizer(),
            Dataset::HuggingFace(dataset) => dataset.tokenizer(),
        }
    }

    fn tokens(&self) -> &[u32] {
        match self {
            Dataset::Shakespeare(dataset) => dataset.tokens(),
            Dataset::HuggingFace(dataset) => dataset.tokens(),
        }
    }

    fn train_len(&self) -> usize {
        match self {
            Dataset::Shakespeare(dataset) => dataset.train_len(),
            Dataset::HuggingFace(dataset) => dataset.train_len(),
        }
    }

    fn block_size(&self) -> usize {
        match self {
            Dataset::Shakespeare(dataset) => dataset.block_size(),
            Dataset::HuggingFace(dataset) => dataset.block_size(),
        }
    }

    fn batch_size(&self) -> usize {
        match self {
            Dataset::Shakespeare(dataset) => dataset.batch_size(),
            Dataset::HuggingFace(dataset) => dataset.batch_size(),
        }
    }

    fn train_split_ratio(&self) -> f32 {
        match self {
            Dataset::Shakespeare(dataset) => dataset.train_split_ratio(),
            Dataset::HuggingFace(dataset) => dataset.train_split_ratio(),
        }
    }

    fn split_offset_and_span(&self, split: DatasetSplit) -> (usize, usize) {
        match self {
            Dataset::Shakespeare(dataset) => {
                TokenSequenceDataset::split_offset_and_span(dataset, split)
            }
            Dataset::HuggingFace(dataset) => {
                TokenSequenceDataset::split_offset_and_span(dataset, split)
            }
        }
    }

    fn steps_per_epoch(&self, split: DatasetSplit) -> usize {
        match self {
            Dataset::Shakespeare(dataset) => TokenSequenceDataset::steps_per_epoch(dataset, split),
            Dataset::HuggingFace(dataset) => TokenSequenceDataset::steps_per_epoch(dataset, split),
        }
    }

    fn decode(&self, tokens: &[i64]) -> String {
        match self {
            Dataset::Shakespeare(dataset) => TokenSequenceDataset::decode(dataset, tokens),
            Dataset::HuggingFace(dataset) => TokenSequenceDataset::decode(dataset, tokens),
        }
    }
}

pub type ShakespeareSplit = DatasetSplit;
pub type ShakespeareBatch<B> = SequenceBatch<B>;
pub type ShakespeareRandomDataLoader<B> = RandomDataLoader<B>;
