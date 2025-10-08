use std::any::Any;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use indexmap::IndexSet;
use serde::{Deserialize, Serialize};

const PAD_CHAR: char = '\u{0000}';
const BOS_CHAR: char = '\u{0001}';
const EOS_CHAR: char = '\u{0002}';
const UNK_CHAR: char = '\u{0003}';

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CharVocab {
    id2ch: Vec<char>,
    ch2id: HashMap<char, u32>,
    bos: u32,
    eos: u32,
    pad: u32,
    unk: Option<u32>,
}

impl CharVocab {
    pub fn fit<'a, I>(texts: I, include_unknown: bool) -> Result<Self>
    where
        I: Iterator<Item = &'a str>,
    {
        let mut chars = IndexSet::new();
        chars.insert(PAD_CHAR);
        chars.insert(BOS_CHAR);
        chars.insert(EOS_CHAR);
        if include_unknown {
            chars.insert(UNK_CHAR);
        }

        for text in texts {
            for ch in text.chars() {
                chars.insert(ch);
            }
        }

        Self::from_chars(chars.into_iter().collect(), include_unknown)
    }

    fn from_chars(chars: Vec<char>, include_unknown: bool) -> Result<Self> {
        if chars.is_empty() {
            return Err(anyhow!("vocabulary cannot be empty"));
        }

        let mut id2ch = Vec::with_capacity(chars.len());
        let mut ch2id = HashMap::with_capacity(chars.len());

        for (idx, ch) in chars.into_iter().enumerate() {
            let id = idx as u32;
            if ch2id.insert(ch, id).is_some() {
                return Err(anyhow!("duplicate character {ch:?}"));
            }
            id2ch.push(ch);
        }

        let bos = *ch2id
            .get(&BOS_CHAR)
            .ok_or_else(|| anyhow!("missing BOS character in vocabulary"))?;
        let eos = *ch2id
            .get(&EOS_CHAR)
            .ok_or_else(|| anyhow!("missing EOS character in vocabulary"))?;
        let pad = *ch2id
            .get(&PAD_CHAR)
            .ok_or_else(|| anyhow!("missing PAD character in vocabulary"))?;
        let unk = if include_unknown {
            ch2id.get(&UNK_CHAR).copied()
        } else {
            None
        };

        Ok(Self {
            id2ch,
            ch2id,
            bos,
            eos,
            pad,
            unk,
        })
    }

    fn to_record(&self) -> CharVocabRecord {
        CharVocabRecord {
            chars: self.id2ch.clone(),
            bos: self.bos,
            eos: self.eos,
            pad: self.pad,
            unk: self.unk,
        }
    }

    fn from_record(record: CharVocabRecord) -> Result<Self> {
        let include_unknown = record.unk.is_some();
        let mut vocab = Self::from_chars(record.chars, include_unknown)?;

        vocab.bos = record.bos;
        vocab.eos = record.eos;
        vocab.pad = record.pad;
        vocab.unk = record.unk;

        Ok(vocab)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create directory {}", parent.display()))?;
        }
        let json = serde_json::to_string_pretty(&self.to_record())
            .context("failed to serialize vocabulary")?;
        fs::write(path, json).with_context(|| format!("failed to write {}", path.display()))?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let data = fs::read_to_string(path)
            .with_context(|| format!("failed to read vocabulary {}", path.display()))?;
        let record: CharVocabRecord = serde_json::from_str(&data)
            .with_context(|| format!("failed to parse vocabulary {}", path.display()))?;
        Self::from_record(record)
    }

    pub fn encode(&self, s: &str, add_bos: bool, add_eos: bool) -> Vec<u32> {
        let mut tokens = Vec::with_capacity(s.chars().count() + 2);
        if add_bos {
            tokens.push(self.bos);
        }

        for ch in s.chars() {
            match self.ch2id.get(&ch) {
                Some(&id) => tokens.push(id),
                None => {
                    if let Some(unk) = self.unk {
                        tokens.push(unk);
                    } else {
                        panic!(
                            "character {ch:?} missing from vocabulary and no <unk> token configured"
                        );
                    }
                }
            }
        }

        if add_eos {
            tokens.push(self.eos);
        }

        tokens
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut text = String::new();
        for &id in ids {
            let idx = id as usize;
            let ch = *self
                .id2ch
                .get(idx)
                .unwrap_or_else(|| panic!("token id {id} out of range"));

            if id == self.pad || id == self.bos {
                continue;
            }

            if id == self.eos {
                break;
            }

            if Some(id) == self.unk {
                text.push('?');
            } else {
                text.push(ch);
            }
        }

        text
    }

    pub fn len(&self) -> usize {
        self.id2ch.len()
    }

    pub fn is_empty(&self) -> bool {
        self.id2ch.is_empty()
    }

    pub fn contains(&self, ch: char) -> bool {
        self.ch2id.contains_key(&ch)
    }

    pub fn bos(&self) -> u32 {
        self.bos
    }

    pub fn eos(&self) -> u32 {
        self.eos
    }

    pub fn pad(&self) -> u32 {
        self.pad
    }

    pub fn unk(&self) -> Option<u32> {
        self.unk
    }
}

#[derive(Serialize, Deserialize)]
struct CharVocabRecord {
    chars: Vec<char>,
    bos: u32,
    eos: u32,
    pad: u32,
    unk: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;
    use tempfile::tempdir;

    #[test]
    fn fit_encode_decode_round_trip() {
        let vocab = CharVocab::fit(["hello", "world"].into_iter(), true).expect("fit");
        assert!(vocab.len() >= 4);
        let encoded = vocab.encode("hello", true, true);
        assert_eq!(encoded.first().copied(), Some(vocab.bos()));
        assert_eq!(encoded.last().copied(), Some(vocab.eos()));
        let decoded = vocab.decode(&encoded);
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn encode_unknown_maps_to_unk() {
        let vocab = CharVocab::fit(["ab"].into_iter(), true).expect("fit");
        let tokens = vocab.encode("ac", false, false);
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0], vocab.ch2id[&'a']);
        assert_eq!(tokens[1], vocab.unk().expect("unk token"));
    }

    #[test]
    fn save_and_load_preserves_vocab() -> io::Result<()> {
        let vocab = CharVocab::fit(["abc"].into_iter(), true).expect("fit");
        let dir = tempdir()?;
        let path = dir.path().join("vocab.json");
        vocab.save(&path).expect("save vocab");
        let loaded = CharVocab::load(&path).expect("load vocab");
        assert_eq!(vocab, loaded);
        Ok(())
    }
}

impl super::Tokenizer for CharVocab {
    fn encode(&self, s: &str, add_bos: bool, add_eos: bool) -> Vec<u32> {
        Self::encode(self, s, add_bos, add_eos)
    }

    fn decode(&self, ids: &[u32]) -> String {
        Self::decode(self, ids)
    }

    fn len(&self) -> usize {
        Self::len(self)
    }

    fn is_empty(&self) -> bool {
        Self::is_empty(self)
    }

    fn bos_id(&self) -> Option<u32> {
        Some(self.bos())
    }

    fn eos_id(&self) -> Option<u32> {
        Some(self.eos())
    }

    fn pad_id(&self) -> Option<u32> {
        Some(self.pad())
    }

    fn unk_id(&self) -> Option<u32> {
        self.unk()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
