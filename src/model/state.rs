use burn::tensor::backend::Backend;

use super::attention::AttentionCache;

#[derive(Debug, Clone)]
pub struct LayerState<B: Backend> {
    pub attention: AttentionCache<B>,
}

#[derive(Debug, Clone)]
pub struct ModelState<B: Backend> {
    pub layers: Vec<LayerState<B>>,
    pub position: usize,
}

impl<B: Backend> ModelState<B> {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| LayerState {
                    attention: AttentionCache::new(),
                })
                .collect(),
            position: 0,
        }
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.attention.reset();
        }
        self.position = 0;
    }

    pub fn len(&self) -> usize {
        self.layers
            .first()
            .map(|layer| layer.attention.len())
            .unwrap_or(0)
    }

    pub fn trim(&mut self, max_len: usize) {
        for layer in &mut self.layers {
            layer.attention.retain_last(max_len);
        }
        self.position = self.len().min(max_len);
    }
}
