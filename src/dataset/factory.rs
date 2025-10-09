use anyhow::{Context, Result};

use crate::config::{
    DatasetConfig, DatasetSourceConfig, HuggingFaceDatasetConfig, HuggingFaceRecordFormat,
    TrainingHyperparameters,
};

use super::{Dataset, HuggingFaceDataset, ShakespeareDataset};

pub fn build_dataset(
    cfg: &DatasetConfig,
    training: &TrainingHyperparameters,
) -> Result<(Dataset, String)> {
    let dataset = match &cfg.source {
        DatasetSourceConfig::Shakespeare { url } => Dataset::from_shakespeare(
            ShakespeareDataset::new_with_source(
                &cfg.cache_dir,
                training.block_size,
                training.batch_size,
                cfg.train_split_ratio,
                &cfg.tokenizer,
                url.as_deref(),
            )
            .with_context(|| "failed to prepare Shakespeare dataset")?,
        ),
        DatasetSourceConfig::HuggingFace(hf_cfg) => Dataset::from_huggingface(
            HuggingFaceDataset::new(
                &cfg.cache_dir,
                training.block_size,
                training.batch_size,
                cfg.train_split_ratio,
                &cfg.tokenizer,
                hf_cfg,
            )
            .with_context(|| {
                format!("failed to prepare Hugging Face dataset {}", hf_cfg.repo_id)
            })?,
        ),
        DatasetSourceConfig::DeepMath {
            revision,
            max_records,
        } => {
            let config = deepmath_config(revision, *max_records);
            Dataset::from_huggingface(
                HuggingFaceDataset::new(
                    &cfg.cache_dir,
                    training.block_size,
                    training.batch_size,
                    cfg.train_split_ratio,
                    &cfg.tokenizer,
                    &config,
                )
                .with_context(|| "failed to prepare DeepMath-103K dataset")?,
            )
        }
        DatasetSourceConfig::TinyChat {
            revision,
            max_records,
        } => {
            let config = tinychat_config(revision, *max_records);
            Dataset::from_huggingface(
                HuggingFaceDataset::new(
                    &cfg.cache_dir,
                    training.block_size,
                    training.batch_size,
                    cfg.train_split_ratio,
                    &cfg.tokenizer,
                    &config,
                )
                .with_context(|| "failed to prepare TinyChat dataset")?,
            )
        }
        DatasetSourceConfig::WebscaleRl {
            revision,
            max_records,
        } => {
            let config = webscale_rl_config(revision, *max_records);
            Dataset::from_huggingface(
                HuggingFaceDataset::new(
                    &cfg.cache_dir,
                    training.block_size,
                    training.batch_size,
                    cfg.train_split_ratio,
                    &cfg.tokenizer,
                    &config,
                )
                .with_context(|| "failed to prepare Webscale-RL dataset")?,
            )
        }
    };

    let description = match &dataset {
        Dataset::Shakespeare(ds) => format!(
            "Prepared Shakespeare dataset with batch_size={}, block_size={}, split_ratio={}",
            ds.batch_size(),
            ds.block_size(),
            ds.train_split_ratio()
        ),
        Dataset::HuggingFace(ds) => format!(
            "Prepared Hugging Face dataset {} (rev: {}) with batch_size={}, block_size={}, split_ratio={}",
            ds.repo_id(),
            ds.revision().unwrap_or("main"),
            ds.batch_size(),
            ds.block_size(),
            ds.train_split_ratio()
        ),
    };

    Ok((dataset, description))
}

fn deepmath_config(
    revision: &Option<String>,
    max_records: Option<usize>,
) -> HuggingFaceDatasetConfig {
    let train_files = (0..10)
        .map(|idx| format!("data/train-{idx:05}-of-00010.parquet"))
        .collect();

    HuggingFaceDatasetConfig {
        repo_id: "zwhe99/DeepMath-103K".to_string(),
        revision: revision.clone(),
        format: HuggingFaceRecordFormat::Parquet,
        train_files,
        validation_files: Vec::new(),
        text_fields: vec!["question".to_string(), "final_answer".to_string()],
        field_separator: "\n\n".to_string(),
        template: Some("Question:\n{question}\n\nAnswer:\n{final_answer}".to_string()),
        max_records,
    }
}

fn tinychat_config(
    revision: &Option<String>,
    max_records: Option<usize>,
) -> HuggingFaceDatasetConfig {
    HuggingFaceDatasetConfig {
        repo_id: "starhopp3r/TinyChat".to_string(),
        revision: revision.clone(),
        format: HuggingFaceRecordFormat::Text,
        train_files: vec!["tinychat.txt".to_string()],
        validation_files: Vec::new(),
        text_fields: vec!["text".to_string()],
        field_separator: "\n\n".to_string(),
        template: None,
        max_records,
    }
}

fn webscale_rl_config(
    revision: &Option<String>,
    max_records: Option<usize>,
) -> HuggingFaceDatasetConfig {
    let mut train_files = Vec::with_capacity(12);
    for idx in 0..12 {
        train_files.push(format!("data/part-{idx}.parquet"));
    }

    HuggingFaceDatasetConfig {
        repo_id: "Salesforce/Webscale-RL".to_string(),
        revision: revision.clone(),
        format: HuggingFaceRecordFormat::Parquet,
        train_files,
        validation_files: Vec::new(),
        text_fields: vec![
            "pretrain_text".to_string(),
            "question".to_string(),
            "answer".to_string(),
            "domain".to_string(),
            "persona".to_string(),
        ],
        field_separator: "\n\n".to_string(),
        template: Some(
            "Context:\n{pretrain_text}\n\nDomain: {domain}\nPersona: {persona}\n\nQuestion: \
             {question}\nAnswer: {answer}"
                .to_string(),
        ),
        max_records,
    }
}
