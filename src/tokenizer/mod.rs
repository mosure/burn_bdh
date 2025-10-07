pub mod byte;
pub mod char_vocab;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Result, anyhow};
use byte::ByteTokenizer;
use char_vocab::CharVocab;
use serde::Deserialize;

pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str, add_bos: bool, add_eos: bool) -> Vec<u32>;
    fn decode(&self, ids: &[u32]) -> String;
    fn len(&self) -> usize;
    fn bos_id(&self) -> Option<u32>;
    fn eos_id(&self) -> Option<u32>;
    fn pad_id(&self) -> Option<u32>;
    fn unk_id(&self) -> Option<u32>;
    fn as_any(&self) -> &dyn std::any::Any;
}

pub type SharedTokenizer = Arc<dyn Tokenizer>;

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct TokenizerConfig {
    #[serde(default)]
    pub vocab_path: Option<PathBuf>,
    #[serde(flatten)]
    pub kind: TokenizerKind,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            vocab_path: None,
            kind: TokenizerKind::Char(CharTokenizerConfig::default()),
        }
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TokenizerKind {
    Char(CharTokenizerConfig),
    Byte(ByteTokenizerConfig),
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct CharTokenizerConfig {
    #[serde(default = "default_true")]
    pub include_unknown: bool,
}

impl Default for CharTokenizerConfig {
    fn default() -> Self {
        Self {
            include_unknown: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ByteTokenizerConfig {
    #[serde(default = "default_true")]
    pub add_special_tokens: bool,
}

impl Default for ByteTokenizerConfig {
    fn default() -> Self {
        Self {
            add_special_tokens: true,
        }
    }
}

impl TokenizerConfig {
    pub fn storage_path(&self, cache_dir: &Path) -> Option<PathBuf> {
        match &self.kind {
            TokenizerKind::Char(_) => Some(match &self.vocab_path {
                Some(path) if path.is_absolute() => path.clone(),
                Some(path) => cache_dir.join(path),
                None => cache_dir.join("vocab.json"),
            }),
            TokenizerKind::Byte(_) => None,
        }
    }

    pub fn load(&self, path: &Path) -> Result<SharedTokenizer> {
        match &self.kind {
            TokenizerKind::Char(_) => {
                let vocab = CharVocab::load(path)?;
                Ok(Arc::new(vocab) as SharedTokenizer)
            }
            TokenizerKind::Byte(config) => {
                Ok(Arc::new(ByteTokenizer::new(config.add_special_tokens)) as SharedTokenizer)
            }
        }
    }

    pub fn fit<'a, I>(&self, texts: I) -> Result<SharedTokenizer>
    where
        I: Iterator<Item = &'a str>,
    {
        match &self.kind {
            TokenizerKind::Char(config) => {
                let vocab = CharVocab::fit(texts, config.include_unknown)?;
                Ok(Arc::new(vocab) as SharedTokenizer)
            }
            TokenizerKind::Byte(config) => {
                Ok(Arc::new(ByteTokenizer::new(config.add_special_tokens)) as SharedTokenizer)
            }
        }
    }

    pub fn save(&self, tokenizer: &dyn Tokenizer, path: &Path) -> Result<()> {
        match &self.kind {
            TokenizerKind::Char(_) => {
                let vocab = tokenizer
                    .as_any()
                    .downcast_ref::<CharVocab>()
                    .ok_or_else(|| anyhow!("expected char tokenizer"))?;
                vocab.save(path)
            }
            TokenizerKind::Byte(_) => Ok(()),
        }
    }

    pub fn requires_strict_coverage(&self) -> bool {
        matches!(&self.kind, TokenizerKind::Char(config) if !config.include_unknown)
    }

    pub fn validate_corpus(&self, tokenizer: &dyn Tokenizer, text: &str) -> Result<()> {
        match &self.kind {
            TokenizerKind::Char(config) if !config.include_unknown => {
                let vocab = tokenizer
                    .as_any()
                    .downcast_ref::<CharVocab>()
                    .ok_or_else(|| anyhow!("expected char tokenizer"))?;
                for ch in text.chars() {
                    if !vocab.contains(ch) {
                        return Err(anyhow!(
                            "vocabulary missing character {ch:?} found in dataset"
                        ));
                    }
                }
                Ok(())
            }
            TokenizerKind::Byte(_) => Ok(()),
            _ => Ok(()),
        }
    }

    pub fn kind_name(&self) -> &'static str {
        match &self.kind {
            TokenizerKind::Char(_) => "char",
            TokenizerKind::Byte(_) => "byte",
        }
    }
}

fn default_true() -> bool {
    true
}
