#![recursion_limit = "512"]

pub mod config;
pub mod dataset;
pub mod generation;
pub mod kernel;
pub mod model;
pub mod tokenizer;
pub mod wgpu;

pub use config::{
    ContextStrategyConfig, DatasetConfig, GenerationConfig, LearningRateScheduleConfig,
    ModelOverrides, OptimizerConfig, TrainingConfig, TrainingHyperparameters, load_training_config,
};
pub use dataset::{
    ShakespeareBatch, ShakespeareDataset, ShakespeareRandomDataLoader, ShakespeareSplit,
};
pub use generation::{
    ContextStrategy, generate_text, generate_tokens, prefill_state, resolve_context_strategy,
    sample_next_token,
};
pub use kernel::{BlockPattern1d, BlockPattern2d, BlockSparseConfig};
pub use model::{BDH, BDHConfig, ModelState, language_model_loss};
pub use tokenizer::char_vocab::CharVocab;
