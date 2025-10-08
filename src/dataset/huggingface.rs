use std::collections::HashMap;
use std::fs;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use burn::tensor::backend::Backend;
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Cache, Repo, RepoType};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use serde_json::Value;
use tracing::warn;

use super::DatasetSplit;
use super::scheduler::{SequenceBatch, TokenSequenceDataset};
use crate::config::{HuggingFaceDatasetConfig, HuggingFaceRecordFormat};
use crate::tokenizer::{SharedTokenizer, TokenizerConfig};

const DEFAULT_RECORD_DELIMITER: &str = "\n";

#[derive(Clone)]
pub struct HuggingFaceDataset {
    tokens: Vec<u32>,
    train_len: usize,
    block_size: usize,
    batch_size: usize,
    train_split_ratio: f32,
    tokenizer: SharedTokenizer,
    repo_id: String,
    revision: Option<String>,
}

impl HuggingFaceDataset {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cache_dir: impl AsRef<Path>,
        block_size: usize,
        batch_size: usize,
        train_split_ratio: f32,
        tokenizer_cfg: &TokenizerConfig,
        hf_cfg: &HuggingFaceDatasetConfig,
    ) -> io::Result<Self> {
        if hf_cfg.text_fields.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "huggingface dataset requires at least one text field",
            ));
        }

        let cache_dir = cache_dir.as_ref();
        fs::create_dir_all(cache_dir)?;
        let hf_cache_dir = cache_dir.join("huggingface");
        fs::create_dir_all(&hf_cache_dir)?;

        let token = std::env::var("HF_TOKEN")
            .ok()
            .or_else(|| Cache::from_env().token());

        let mut api_builder = ApiBuilder::new().with_cache_dir(hf_cache_dir);
        if let Some(token) = token {
            api_builder = api_builder.with_token(Some(token));
        }
        let api = api_builder.build().map_err(io::Error::other)?;

        let repo = if let Some(revision) = &hf_cfg.revision {
            Repo::with_revision(hf_cfg.repo_id.clone(), RepoType::Dataset, revision.clone())
        } else {
            Repo::new(hf_cfg.repo_id.clone(), RepoType::Dataset)
        };
        let repo = api.repo(repo);

        let mut train_records = Vec::new();
        for file in &hf_cfg.train_files {
            if hf_cfg
                .max_records
                .is_some_and(|limit| train_records.len() >= limit)
            {
                break;
            }
            let path = repo
                .get(file)
                .map_err(|err| io::Error::other(format!("failed to download {file}: {err}")))?;
            collect_records(&path, hf_cfg, hf_cfg.max_records, &mut train_records)?;
        }

        let mut val_records = Vec::new();
        for file in &hf_cfg.validation_files {
            let path = repo
                .get(file)
                .map_err(|err| io::Error::other(format!("failed to download {file}: {err}")))?;
            collect_records(&path, hf_cfg, hf_cfg.max_records, &mut val_records)?;
        }

        if train_records.is_empty() && val_records.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "huggingface dataset contains no records",
            ));
        }

        let tokenizer_path = tokenizer_cfg.storage_path(cache_dir);
        let tokenizer = if let Some(path) = tokenizer_path {
            if path.is_file() {
                tokenizer_cfg.load(&path).map_err(io::Error::other)?
            } else {
                let tokenizer = tokenizer_cfg
                    .fit(record_iter(&train_records, &val_records))
                    .map_err(io::Error::other)?;
                tokenizer_cfg
                    .save(&*tokenizer, &path)
                    .map_err(io::Error::other)?;
                tokenizer
            }
        } else {
            tokenizer_cfg
                .fit(record_iter(&train_records, &val_records))
                .map_err(io::Error::other)?
        };

        for record in record_iter(&train_records, &val_records) {
            tokenizer_cfg
                .validate_corpus(&*tokenizer, record)
                .map_err(io::Error::other)?;
        }

        let mut tokens = Vec::new();
        let mut train_len = 0usize;

        for record in train_records.into_iter() {
            let mut encoded = tokenizer.encode(record.as_str(), false, false);
            if encoded.len() < 2 {
                warn!(
                    "skipping short training record from {} ({} tokens)",
                    hf_cfg.repo_id,
                    encoded.len()
                );
                continue;
            }
            train_len += encoded.len();
            tokens.append(&mut encoded);
        }

        let mut val_token_count = 0usize;
        for record in val_records.into_iter() {
            let mut encoded = tokenizer.encode(record.as_str(), false, false);
            if encoded.len() < 2 {
                warn!(
                    "skipping short validation record from {} ({} tokens)",
                    hf_cfg.repo_id,
                    encoded.len()
                );
                continue;
            }
            val_token_count += encoded.len();
            tokens.append(&mut encoded);
        }

        if tokens.len() <= block_size + 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "encoded huggingface dataset smaller than block size",
            ));
        }

        if val_token_count == 0 {
            let split_ratio = train_split_ratio.clamp(0.0, 1.0);
            let mut ratio_len = ((tokens.len() as f32) * split_ratio) as usize;
            let min_len = block_size + 1;
            let max_len = tokens.len().saturating_sub(1);
            if ratio_len < min_len {
                ratio_len = min_len;
            } else if ratio_len > max_len {
                ratio_len = max_len;
            }
            train_len = ratio_len;
        } else if train_len <= block_size {
            train_len = (block_size + 1).min(tokens.len().saturating_sub(1));
        }

        Ok(Self {
            tokens,
            train_len,
            block_size,
            batch_size,
            train_split_ratio: train_split_ratio.clamp(0.0, 1.0),
            tokenizer: tokenizer.clone(),
            repo_id: hf_cfg.repo_id.clone(),
            revision: hf_cfg.revision.clone(),
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

    pub fn repo_id(&self) -> &str {
        &self.repo_id
    }

    pub fn revision(&self) -> Option<&str> {
        self.revision.as_deref()
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

impl TokenSequenceDataset for HuggingFaceDataset {
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

fn collect_records(
    path: &Path,
    cfg: &HuggingFaceDatasetConfig,
    max_records: Option<usize>,
    records: &mut Vec<String>,
) -> io::Result<()> {
    match cfg.format {
        HuggingFaceRecordFormat::Jsonl => collect_jsonl_records(path, cfg, max_records, records),
        HuggingFaceRecordFormat::Text => collect_text_records(path, cfg, max_records, records),
        HuggingFaceRecordFormat::Parquet => {
            collect_parquet_records(path, cfg, max_records, records)
        }
    }
}

fn collect_jsonl_records(
    path: &Path,
    cfg: &HuggingFaceDatasetConfig,
    max_records: Option<usize>,
    records: &mut Vec<String>,
) -> io::Result<()> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);

    for line in reader.lines() {
        if max_records.is_some_and(|limit| records.len() >= limit) {
            break;
        }
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&line).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("failed to parse JSON record from {}: {err}", path.display()),
            )
        })?;

        match render_hf_record(cfg, extract_fields_from_json(cfg, &value)?)? {
            Some(rendered) => records.push(rendered),
            None => continue,
        }
    }

    Ok(())
}

fn collect_text_records(
    path: &Path,
    cfg: &HuggingFaceDatasetConfig,
    max_records: Option<usize>,
    records: &mut Vec<String>,
) -> io::Result<()> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let field_name = cfg
        .text_fields
        .first()
        .cloned()
        .unwrap_or_else(|| "text".to_string());

    for line in reader.lines() {
        if max_records.is_some_and(|limit| records.len() >= limit) {
            break;
        }
        let text = line?;
        if text.trim().is_empty() {
            continue;
        }
        let mut fields = HashMap::new();
        fields.insert(field_name.as_str(), text);
        match render_hf_record(cfg, fields)? {
            Some(rendered) => records.push(rendered),
            None => continue,
        }
    }

    Ok(())
}

fn collect_parquet_records(
    path: &Path,
    cfg: &HuggingFaceDatasetConfig,
    max_records: Option<usize>,
    records: &mut Vec<String>,
) -> io::Result<()> {
    let file = fs::File::open(path)?;
    let reader = SerializedFileReader::new(file).map_err(io::Error::other)?;
    let schema = reader.metadata().file_metadata().schema_descr();

    let mut index_map = HashMap::new();
    for (idx, column) in schema.columns().iter().enumerate() {
        index_map.insert(column.path().string(), idx);
    }

    let row_iter = reader.get_row_iter(None).map_err(io::Error::other)?;
    for row in row_iter {
        let row = row.map_err(io::Error::other)?;

        if max_records.is_some_and(|limit| records.len() >= limit) {
            break;
        }
        let mut field_values = HashMap::new();
        for field in &cfg.text_fields {
            let idx = *index_map.get(field.as_str()).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "missing field `{}` in parquet file {}",
                        field,
                        path.display()
                    ),
                )
            })?;

            let value = if let Ok(s) = row.get_string(idx) {
                s.clone()
            } else if let Ok(bytes) = row.get_bytes(idx) {
                String::from_utf8_lossy(bytes.data()).to_string()
            } else {
                row.get_column_iter()
                    .nth(idx)
                    .map(|(_, field)| field.to_string())
                    .ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "unable to render parquet field `{}` in {}",
                                field,
                                path.display()
                            ),
                        )
                    })?
            };

            field_values.insert(field.as_str(), value);
        }

        match render_hf_record(cfg, field_values)? {
            Some(rendered) => records.push(rendered),
            None => continue,
        }
    }

    Ok(())
}

fn extract_fields_from_json<'a>(
    cfg: &'a HuggingFaceDatasetConfig,
    value: &'a Value,
) -> io::Result<HashMap<&'a str, String>> {
    let mut map = HashMap::new();
    for field in &cfg.text_fields {
        let field_value = value.get(field).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("missing `{field}` in dataset record"),
            )
        })?;
        let text = match field_value {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        map.insert(field.as_str(), text);
    }
    Ok(map)
}

fn render_hf_record(
    cfg: &HuggingFaceDatasetConfig,
    fields: HashMap<&str, String>,
) -> io::Result<Option<String>> {
    if fields.is_empty() {
        return Ok(None);
    }

    let rendered = if let Some(template) = &cfg.template {
        render_template(template, &fields)?
    } else {
        let mut ordered = Vec::with_capacity(cfg.text_fields.len());
        for field in &cfg.text_fields {
            let value = fields.get(field.as_str()).cloned().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("template field `{field}` missing from record"),
                )
            })?;
            ordered.push(value);
        }

        let mut joined = if ordered.len() == 1 {
            ordered.into_iter().next().unwrap()
        } else {
            ordered.join(if cfg.field_separator.is_empty() {
                DEFAULT_RECORD_DELIMITER
            } else {
                cfg.field_separator.as_str()
            })
        };
        if !joined.ends_with('\n') {
            joined.push('\n');
        }
        joined
    };

    Ok(Some(rendered))
}

fn render_template(template: &str, fields: &HashMap<&str, String>) -> io::Result<String> {
    let mut result = String::with_capacity(template.len());
    let mut chars = template.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '{' {
            let mut key = String::new();
            let mut closed = false;
            for next in chars.by_ref() {
                if next == '}' {
                    closed = true;
                    break;
                }
                key.push(next);
            }
            if !closed {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "unclosed template placeholder",
                ));
            }
            if key.trim().is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "empty template placeholder {}",
                ));
            }
            let field_key = key.trim();
            let value = fields.get(field_key).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown template placeholder {{{field_key}}}"),
                )
            })?;
            result.push_str(value);
        } else {
            result.push(ch);
        }
    }

    if !result.ends_with('\n') {
        result.push('\n');
    }

    Ok(result)
}

fn record_iter<'a>(train: &'a [String], val: &'a [String]) -> impl Iterator<Item = &'a str> {
    train
        .iter()
        .map(String::as_str)
        .chain(val.iter().map(String::as_str))
}
