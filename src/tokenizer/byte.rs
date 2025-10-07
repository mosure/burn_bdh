use super::Tokenizer;

const BYTE_VOCAB_SIZE: usize = 256;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ByteTokenizer {
    add_special_tokens: bool,
    bos: Option<u32>,
    eos: Option<u32>,
    pad: Option<u32>,
    vocab_size: usize,
}

impl ByteTokenizer {
    pub fn new(add_special_tokens: bool) -> Self {
        let mut vocab_size = BYTE_VOCAB_SIZE;
        let mut bos = None;
        let mut eos = None;
        let mut pad = None;

        if add_special_tokens {
            bos = Some(vocab_size as u32);
            vocab_size += 1;
            eos = Some(vocab_size as u32);
            vocab_size += 1;
            pad = Some(vocab_size as u32);
            vocab_size += 1;
        }

        Self {
            add_special_tokens,
            bos,
            eos,
            pad,
            vocab_size,
        }
    }
}

impl Tokenizer for ByteTokenizer {
    fn encode(&self, text: &str, add_bos: bool, add_eos: bool) -> Vec<u32> {
        let mut tokens = Vec::with_capacity(text.len() + 2);
        if add_bos {
            if let Some(bos) = self.bos {
                tokens.push(bos);
            }
        }

        for byte in text.as_bytes() {
            tokens.push(*byte as u32);
        }

        if add_eos {
            if let Some(eos) = self.eos {
                tokens.push(eos);
            }
        }

        tokens
    }

    fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::with_capacity(ids.len());
        for &id in ids {
            if Some(id) == self.pad || Some(id) == self.bos {
                continue;
            }
            if Some(id) == self.eos {
                break;
            }
            if (id as usize) < BYTE_VOCAB_SIZE {
                bytes.push(id as u8);
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }

    fn len(&self) -> usize {
        self.vocab_size
    }

    fn bos_id(&self) -> Option<u32> {
        self.bos
    }

    fn eos_id(&self) -> Option<u32> {
        self.eos
    }

    fn pad_id(&self) -> Option<u32> {
        self.pad
    }

    fn unk_id(&self) -> Option<u32> {
        None
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_round_trip() {
        let tokenizer = ByteTokenizer::new(true);
        let encoded = tokenizer.encode("hello", true, true);
        assert_eq!(encoded.first().copied(), tokenizer.bos_id());
        assert_eq!(encoded.last().copied(), tokenizer.eos_id());
        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn decode_truncates_at_eos() {
        let tokenizer = ByteTokenizer::new(true);
        let mut ids = tokenizer.encode("abc", false, false);
        if let Some(eos) = tokenizer.eos_id() {
            ids.push(eos);
        }
        ids.extend([b'd' as u32, b'e' as u32]);
        assert_eq!(tokenizer.decode(&ids), "abc");
    }
}
