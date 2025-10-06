use std::collections::HashSet;

use burn::module::{
    AutodiffModule, Devices, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor,
};
use burn::tensor::Tensor;
use burn::tensor::backend::{AutodiffBackend, Backend};

#[derive(Clone, Debug)]
pub struct BlockPattern1d {
    block_size: usize,
    active_blocks: Option<HashSet<usize>>,
}

impl BlockPattern1d {
    pub fn dense(block_size: usize) -> Self {
        let block_size = block_size.max(1);
        Self {
            block_size,
            active_blocks: None,
        }
    }

    pub fn from_blocks(block_size: usize, blocks: impl IntoIterator<Item = usize>) -> Self {
        Self {
            block_size,
            active_blocks: Some(blocks.into_iter().collect()),
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn is_sparse(&self) -> bool {
        self.active_blocks.is_some()
    }

    pub fn is_block_active(&self, block_idx: usize) -> bool {
        match &self.active_blocks {
            Some(set) => set.contains(&block_idx),
            None => true,
        }
    }

    pub fn mask<B: Backend>(&self, elements: usize, device: &B::Device) -> Tensor<B, 4> {
        let mut data = vec![0.0; elements];

        let block_size = self.block_size.max(1);
        let total_blocks = elements.div_ceil(block_size);

        for block_idx in 0..total_blocks {
            if self.is_block_active(block_idx) {
                let start = block_idx * block_size;
                let end = usize::min(start + block_size, elements);
                data[start..end].fill(1.0);
            }
        }

        Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([1, 1, 1, elements])
    }
}

#[derive(Clone, Debug)]
pub struct BlockPattern2d {
    block_size: usize,
    active_pairs: Option<HashSet<(usize, usize)>>,
}

impl BlockPattern2d {
    pub fn dense(block_size: usize) -> Self {
        let block_size = block_size.max(1);
        Self {
            block_size,
            active_pairs: None,
        }
    }

    pub fn from_pairs(block_size: usize, pairs: impl IntoIterator<Item = (usize, usize)>) -> Self {
        Self {
            block_size,
            active_pairs: Some(pairs.into_iter().collect()),
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn is_sparse(&self) -> bool {
        self.active_pairs.is_some()
    }

    pub fn is_active(&self, row: usize, col: usize) -> bool {
        match &self.active_pairs {
            Some(set) => set.contains(&(row, col)),
            None => col <= row,
        }
    }

    pub fn iter_cols(&self, row: usize, total_blocks: usize) -> Vec<usize> {
        match &self.active_pairs {
            Some(set) => set
                .iter()
                .filter_map(|(r, c)| {
                    if *r == row && *c < total_blocks {
                        Some(*c)
                    } else {
                        None
                    }
                })
                .collect(),
            None => (0..=row.min(total_blocks.saturating_sub(1))).collect(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BlockSparseConfig {
    pub latent: BlockPattern1d,
    pub time: BlockPattern2d,
}

impl BlockSparseConfig {
    pub fn dense(latent_block: usize, time_block: usize) -> Self {
        Self {
            latent: BlockPattern1d::dense(latent_block),
            time: BlockPattern2d::dense(time_block),
        }
    }
}

impl<B: Backend> Module<B> for BlockPattern1d {
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

impl<B: Backend> Module<B> for BlockPattern2d {
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

impl<B: Backend> Module<B> for BlockSparseConfig {
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

impl<B: AutodiffBackend> AutodiffModule<B> for BlockPattern1d {
    type InnerModule = BlockPattern1d;

    fn valid(&self) -> Self::InnerModule {
        self.clone()
    }
}

impl ModuleDisplayDefault for BlockPattern1d {
    fn content(&self, _content: burn::module::Content) -> Option<burn::module::Content> {
        None
    }
}

impl ModuleDisplay for BlockPattern1d {}

impl<B: AutodiffBackend> AutodiffModule<B> for BlockPattern2d {
    type InnerModule = BlockPattern2d;

    fn valid(&self) -> Self::InnerModule {
        self.clone()
    }
}

impl ModuleDisplayDefault for BlockPattern2d {
    fn content(&self, _content: burn::module::Content) -> Option<burn::module::Content> {
        None
    }
}

impl ModuleDisplay for BlockPattern2d {}

impl<B: AutodiffBackend> AutodiffModule<B> for BlockSparseConfig {
    type InnerModule = BlockSparseConfig;

    fn valid(&self) -> Self::InnerModule {
        self.clone()
    }
}

impl ModuleDisplayDefault for BlockSparseConfig {
    fn content(&self, _content: burn::module::Content) -> Option<burn::module::Content> {
        None
    }
}

impl ModuleDisplay for BlockSparseConfig {}
