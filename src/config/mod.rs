use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;

use crate::tokenizer::TokenizerConfig;
use toml::Value;

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct DatasetConfig {
    pub cache_dir: PathBuf,
    #[serde(default = "default_train_split_ratio")]
    pub train_split_ratio: f32,
    #[serde(flatten)]
    pub source: DatasetSourceConfig,
    #[serde(default)]
    pub tokenizer: TokenizerConfig,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DatasetSourceConfig {
    Shakespeare {
        #[serde(default)]
        url: Option<String>,
    },
    HuggingFace(HuggingFaceDatasetConfig),
    DeepMath {
        #[serde(default)]
        revision: Option<String>,
        #[serde(default)]
        max_records: Option<usize>,
    },
    TinyChat {
        #[serde(default)]
        revision: Option<String>,
        #[serde(default)]
        max_records: Option<usize>,
    },
    WebscaleRl {
        #[serde(default)]
        revision: Option<String>,
        #[serde(default)]
        max_records: Option<usize>,
    },
}

impl Default for DatasetSourceConfig {
    fn default() -> Self {
        Self::Shakespeare { url: None }
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct HuggingFaceDatasetConfig {
    pub repo_id: String,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub format: HuggingFaceRecordFormat,
    #[serde(default = "default_hf_train_files")]
    pub train_files: Vec<String>,
    #[serde(default)]
    pub validation_files: Vec<String>,
    #[serde(default = "default_hf_text_fields")]
    pub text_fields: Vec<String>,
    #[serde(default = "default_hf_field_separator")]
    pub field_separator: String,
    #[serde(default)]
    pub template: Option<String>,
    #[serde(default)]
    pub max_records: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum HuggingFaceRecordFormat {
    Jsonl,
    Text,
    Parquet,
}

impl Default for HuggingFaceRecordFormat {
    fn default() -> Self {
        Self::Jsonl
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct TrainingHyperparameters {
    pub block_size: usize,
    pub batch_size: usize,
    pub max_iters: usize,
    pub log_frequency: usize,
    #[serde(default = "default_context_strategy")]
    pub context_strategy: ContextStrategyConfig,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct OptimizerConfig {
    pub learning_rate: f64,
    pub weight_decay: f32,
    #[serde(default)]
    pub lr_schedule: Option<LearningRateScheduleConfig>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LearningRateScheduleConfig {
    Constant {
        #[serde(default)]
        initial_lr: Option<f64>,
    },
    Cosine {
        #[serde(default)]
        initial_lr: Option<f64>,
        #[serde(default)]
        min_lr: Option<f64>,
        #[serde(default)]
        num_iters: Option<usize>,
    },
    Linear {
        #[serde(default)]
        initial_lr: Option<f64>,
        final_lr: f64,
        #[serde(default)]
        num_iters: Option<usize>,
    },
    Exponential {
        #[serde(default)]
        initial_lr: Option<f64>,
        gamma: f64,
    },
    Step {
        #[serde(default)]
        initial_lr: Option<f64>,
        #[serde(default = "default_step_gamma")]
        gamma: f64,
        #[serde(default)]
        step_size: Option<usize>,
    },
    Noam {
        #[serde(default)]
        initial_lr: Option<f64>,
        #[serde(default)]
        warmup_steps: Option<usize>,
        #[serde(default)]
        model_size: Option<usize>,
    },
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct GenerationConfig {
    pub prompt: String,
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default = "default_context_strategy")]
    pub context_strategy: ContextStrategyConfig,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContextStrategyConfig {
    #[default]
    Infinite,
    Sliding {
        window: usize,
    },
}

fn default_context_strategy() -> ContextStrategyConfig {
    ContextStrategyConfig::Infinite
}

#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
pub struct ModelOverrides {
    pub n_layer: Option<usize>,
    pub n_embd: Option<usize>,
    pub n_head: Option<usize>,
    pub mlp_internal_dim_multiplier: Option<usize>,
    pub dropout: Option<f64>,
    pub fused_kernels: Option<bool>,
    pub use_alibi: Option<bool>,
    pub block_size: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct TrainingConfig {
    pub dataset: DatasetConfig,
    pub training: TrainingHyperparameters,
    pub optimizer: OptimizerConfig,
    pub generation: GenerationConfig,
    #[serde(default)]
    pub model: ModelOverrides,
}

pub fn load_training_config(paths: &[PathBuf]) -> Result<TrainingConfig> {
    if paths.is_empty() {
        return Err(anyhow!("at least one configuration path is required"));
    }

    let mut iter = paths.iter();
    let first_path = iter
        .next()
        .ok_or_else(|| anyhow!("configuration iterator unexpectedly empty"))?;
    let mut value = load_value(first_path)?;

    for path in iter {
        let overlay = load_value(path)?;
        merge_values(&mut value, overlay);
    }

    value
        .try_into::<TrainingConfig>()
        .map_err(|err| anyhow!(err))
}

fn load_value(path: &Path) -> Result<Value> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read configuration file {}", path.display()))?;
    let table: toml::value::Table = toml::from_str(&content)
        .with_context(|| format!("failed to parse {} as TOML", path.display()))?;
    Ok(Value::Table(table))
}

fn merge_values(base: &mut Value, overlay: Value) {
    match (base, overlay) {
        (Value::Table(base_table), Value::Table(overlay_table)) => {
            if let Some(Value::String(overlay_type)) = overlay_table.get("type") {
                let type_changed = match base_table.get("type") {
                    Some(Value::String(base_type)) => base_type != overlay_type,
                    Some(_) => true,
                    None => !base_table.is_empty(),
                };
                if type_changed {
                    base_table.clear();
                }
            }
            for (key, overlay_value) in overlay_table {
                match base_table.get_mut(&key) {
                    Some(base_value) => merge_values(base_value, overlay_value),
                    None => {
                        base_table.insert(key, overlay_value);
                    }
                }
            }
        }
        (base_value, overlay_value) => {
            *base_value = overlay_value;
        }
    }
}

fn default_train_split_ratio() -> f32 {
    0.9
}

fn default_hf_train_files() -> Vec<String> {
    vec!["train.jsonl".to_string()]
}

fn default_hf_text_fields() -> Vec<String> {
    vec!["text".to_string()]
}

fn default_hf_field_separator() -> String {
    "\n".to_string()
}

fn default_temperature() -> f32 {
    1.0
}

fn default_step_gamma() -> f64 {
    0.1
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn write_config(dir: &Path, name: &str, contents: &str) -> PathBuf {
        let path = dir.join(name);
        let trimmed_lines: Vec<&str> = contents.lines().map(|line| line.trim_start()).collect();
        let mut formatted = trimmed_lines.join("\n");
        if formatted.starts_with('\n') {
            formatted = formatted.trim_start_matches('\n').to_string();
        }
        fs::write(&path, formatted).expect("write config");
        path
    }

    #[test]
    fn load_merges_in_order() {
        let dir = tempdir().expect("tempdir");

        let base_contents = [
            "[dataset]",
            "cache_dir = \"data\"",
            "train_split_ratio = 0.8",
            "type = \"shakespeare\"",
            "",
            "[training]",
            "block_size = 256",
            "batch_size = 16",
            "max_iters = 1000",
            "log_frequency = 50",
            "",
            "[optimizer]",
            "learning_rate = 0.001",
            "weight_decay = 0.05",
            "",
            "[optimizer.lr_schedule]",
            "type = \"cosine\"",
            "min_lr = 0.00005",
            "num_iters = 100",
            "",
            "[generation]",
            "prompt = \"Base prompt\"",
            "max_tokens = 64",
            "temperature = 0.9",
            "top_k = 4",
            "",
            "[model]",
            "n_layer = 6",
            "n_embd = 256",
            "n_head = 4",
            "mlp_internal_dim_multiplier = 128",
            "dropout = 0.1",
            "fused_kernels = false",
            "use_alibi = false",
        ]
        .join("\n");
        let base = write_config(dir.path(), "base.toml", &base_contents);

        let override_contents = [
            "[training]",
            "max_iters = 2000",
            "",
            "[optimizer]",
            "learning_rate = 0.0005",
            "",
            "[optimizer.lr_schedule]",
            "type = \"linear\"",
            "final_lr = 0.0002",
            "num_iters = 50",
            "",
            "[model]",
            "n_embd = 320",
            "fused_kernels = true",
            "block_size = 256",
        ]
        .join("\n");
        let override_cfg = write_config(dir.path(), "override.toml", &override_contents);

        let config = load_training_config(&[base, override_cfg]).expect("load config");

        assert_eq!(
            config.training,
            TrainingHyperparameters {
                block_size: 256,
                batch_size: 16,
                max_iters: 2000,
                log_frequency: 50,
                context_strategy: ContextStrategyConfig::Infinite,
            }
        );
        assert!((config.optimizer.learning_rate - 0.0005).abs() < f64::EPSILON);
        assert!((config.optimizer.weight_decay - 0.05).abs() < f32::EPSILON);
        assert_eq!(
            config.optimizer.lr_schedule,
            Some(LearningRateScheduleConfig::Linear {
                initial_lr: None,
                final_lr: 0.0002,
                num_iters: Some(50),
            })
        );
        assert_eq!(config.dataset.tokenizer, TokenizerConfig::default());
        assert!((config.dataset.train_split_ratio - 0.8).abs() < f32::EPSILON);
        assert_eq!(
            config.dataset.source,
            DatasetSourceConfig::Shakespeare { url: None }
        );
        assert_eq!(config.generation.max_tokens, 64);
        assert_eq!(
            config.training.context_strategy,
            ContextStrategyConfig::Infinite
        );
        assert_eq!(
            config.generation.context_strategy,
            ContextStrategyConfig::Infinite
        );
        assert_eq!(config.model.n_layer, Some(6));
        assert_eq!(config.model.n_embd, Some(320));
        assert_eq!(config.model.n_head, Some(4));
        assert_eq!(config.model.mlp_internal_dim_multiplier, Some(128));
        assert_eq!(config.model.dropout, Some(0.1));
        assert_eq!(config.model.fused_kernels, Some(true));
        assert_eq!(config.model.block_size, Some(256));
        assert_eq!(config.model.use_alibi, Some(false));
    }

    #[test]
    fn schedule_constant_round_trips() {
        let text = r#"
            learning_rate = 0.002
            weight_decay = 0.1

            [lr_schedule]
            type = "constant"
        "#;
        let optimizer: OptimizerConfig = toml::from_str(text).expect("parse optimizer config");
        assert_eq!(
            optimizer.lr_schedule,
            Some(LearningRateScheduleConfig::Constant { initial_lr: None })
        );
    }

    #[test]
    fn huggingface_dataset_config_parses() {
        let text = r#"
            cache_dir = "data"
            train_split_ratio = 0.75
            type = "hugging_face"
            repo_id = "zwhe99/DeepMath-103K"
            revision = "main"
            format = "parquet"
            train_files = [
                "data/train-00000-of-00010.parquet",
                "data/train-00001-of-00010.parquet",
            ]
            validation_files = []
            text_fields = ["question", "final_answer"]
            field_separator = "\n\n"
            template = "{question}\n{final_answer}"
            max_records = 1000
        "#;
        let dataset: DatasetConfig = toml::from_str(text).expect("parse dataset config");
        assert_eq!(dataset.train_split_ratio, 0.75);
        match &dataset.source {
            DatasetSourceConfig::HuggingFace(hf) => {
                assert_eq!(hf.repo_id, "zwhe99/DeepMath-103K");
                assert_eq!(hf.revision.as_deref(), Some("main"));
                assert_eq!(hf.format, HuggingFaceRecordFormat::Parquet);
                assert_eq!(
                    hf.train_files,
                    vec![
                        "data/train-00000-of-00010.parquet".to_string(),
                        "data/train-00001-of-00010.parquet".to_string()
                    ]
                );
                assert!(hf.validation_files.is_empty());
                assert_eq!(hf.text_fields, vec!["question", "final_answer"]);
                assert_eq!(hf.field_separator, "\n\n");
                assert_eq!(hf.template.as_deref(), Some("{question}\n{final_answer}"));
                assert_eq!(hf.max_records, Some(1000));
            }
            other => panic!("unexpected dataset source: {other:?}"),
        }
    }
}
