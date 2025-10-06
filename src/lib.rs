#![recursion_limit = "512"]

pub mod config;
pub mod dataset;
pub mod kernel;
pub mod model;
pub mod wgpu;

pub use config::{
    DatasetConfig, GenerationConfig, ModelOverrides, OptimizerConfig, TrainingConfig,
    TrainingHyperparameters, load_training_config,
};
pub use dataset::{ShakespeareDataset, ShakespeareSplit};
pub use kernel::{BlockPattern1d, BlockPattern2d, BlockSparseConfig};
pub use model::{BDH, BDHConfig, language_model_loss};
