use burn::module::{
    AutodiffModule, Devices, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor,
};
use burn::tensor::backend::{AutodiffBackend, Backend};

use crate::kernel::{BlockPattern1d, BlockPattern2d, BlockSparseConfig};

#[derive(Clone, Debug)]
pub struct FusedKernelConfig {
    pub enabled: bool,
    pub block_sparse: BlockSparseConfig,
    pub rope_theta: f32,
    pub relu_threshold: f32,
    pub alibi_slopes: Option<Vec<f32>>,
    pub use_alibi: bool,
}

impl Default for FusedKernelConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            block_sparse: BlockSparseConfig::dense(64, 64),
            rope_theta: 65_536.0,
            relu_threshold: 0.0,
            alibi_slopes: None,
            use_alibi: false,
        }
    }
}

impl FusedKernelConfig {
    pub fn with_block_sizes(mut self, latent: usize, time: usize) -> Self {
        self.set_block_sizes(latent, time);
        self
    }

    pub fn set_block_sizes(&mut self, latent: usize, time: usize) {
        self.block_sparse = BlockSparseConfig {
            latent: BlockPattern1d::dense(latent),
            time: BlockPattern2d::dense(time),
        };
    }

    pub fn set_alibi_slopes(&mut self, slopes: Vec<f32>) {
        self.alibi_slopes = Some(slopes);
    }

    pub fn set_use_alibi(&mut self, enabled: bool) {
        self.use_alibi = enabled;
    }
}

impl<B: Backend> Module<B> for FusedKernelConfig {
    type Record = ();

    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for FusedKernelConfig {
    type InnerModule = FusedKernelConfig;

    fn valid(&self) -> Self::InnerModule {
        self.clone()
    }
}

impl ModuleDisplayDefault for FusedKernelConfig {
    fn content(&self, _content: burn::module::Content) -> Option<burn::module::Content> {
        None
    }
}

impl ModuleDisplay for FusedKernelConfig {}

#[derive(Clone, Debug)]
pub struct BDHConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub dropout: f64,
    pub n_head: usize,
    pub mlp_internal_dim_multiplier: usize,
    pub vocab_size: usize,
    pub fused_kernels: FusedKernelConfig,
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
            fused_kernels: FusedKernelConfig::default(),
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
