use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use toml::Value;

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct DatasetConfig {
    pub cache_dir: PathBuf,
    #[serde(default = "default_train_split_ratio")]
    pub train_split_ratio: f32,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct TrainingHyperparameters {
    pub block_size: usize,
    pub batch_size: usize,
    pub max_iters: usize,
    pub log_frequency: usize,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct OptimizerConfig {
    pub learning_rate: f64,
    pub weight_decay: f32,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct GenerationConfig {
    pub prompt: String,
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub top_k: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
pub struct ModelOverrides {
    pub n_layer: Option<usize>,
    pub n_embd: Option<usize>,
    pub n_head: Option<usize>,
    pub mlp_internal_dim_multiplier: Option<usize>,
    pub dropout: Option<f64>,
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

fn default_temperature() -> f32 {
    1.0
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
            "[model]",
            "n_embd = 320",
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
            }
        );
        assert!((config.optimizer.learning_rate - 0.0005).abs() < f64::EPSILON);
        assert!((config.optimizer.weight_decay - 0.05).abs() < f32::EPSILON);
        assert!((config.dataset.train_split_ratio - 0.8).abs() < f32::EPSILON);
        assert_eq!(config.generation.max_tokens, 64);
        assert_eq!(config.model.n_layer, Some(6));
        assert_eq!(config.model.n_embd, Some(320));
        assert_eq!(config.model.n_head, Some(4));
        assert_eq!(config.model.mlp_internal_dim_multiplier, Some(128));
        assert_eq!(config.model.dropout, Some(0.1));
    }
}
