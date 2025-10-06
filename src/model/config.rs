#[derive(Clone, Debug)]
pub struct BDHConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub dropout: f64,
    pub n_head: usize,
    pub mlp_internal_dim_multiplier: usize,
    pub vocab_size: usize,
}

impl Default for BDHConfig {
    fn default() -> Self {
        Self {
            n_layer: 6,
            n_embd: 256,
            dropout: 0.1,
            n_head: 4,
            mlp_internal_dim_multiplier: 128,
            vocab_size: 256,
        }
    }
}

impl BDHConfig {
    pub(crate) fn latent_per_head(&self) -> usize {
        (self.mlp_internal_dim_multiplier * self.n_embd) / self.n_head
    }

    pub(crate) fn latent_total(&self) -> usize {
        self.latent_per_head() * self.n_head
    }
}
