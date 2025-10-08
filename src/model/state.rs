use burn::tensor::backend::{AutodiffBackend, Backend};

use super::attention::AttentionCache;

#[derive(Debug, Clone)]
pub struct LayerState<B: Backend> {
    pub attention: AttentionCache<B>,
}

#[derive(Debug, Clone)]
pub struct ModelState<B: Backend> {
    pub layers: Vec<LayerState<B>>,
}

impl<B: Backend> ModelState<B> {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| LayerState {
                    attention: AttentionCache::new(),
                })
                .collect(),
        }
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.attention.reset_all();
        }
    }

    pub fn ensure_capacity(&mut self, capacity: usize) {
        for layer in &mut self.layers {
            layer.attention.ensure_streams(capacity);
        }
    }

    pub fn reset_stream(&mut self, stream: usize) {
        for layer in &mut self.layers {
            layer.attention.reset_stream(stream);
        }
    }

    pub fn trim_stream(&mut self, stream: usize, max_len: usize) {
        for layer in &mut self.layers {
            layer.attention.trim_stream(stream, max_len);
        }
    }

    pub fn trim_streams(&mut self, streams: &[usize], max_len: usize) {
        for &stream in streams {
            self.trim_stream(stream, max_len);
        }
    }
}

impl<B: AutodiffBackend> ModelState<B> {
    pub fn detach(&mut self) {
        for layer in &mut self.layers {
            layer.attention.detach();
        }
    }
}
