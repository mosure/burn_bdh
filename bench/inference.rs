#![recursion_limit = "512"]

use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Int, Tensor, TensorData};
use burn_dragon_hatchling::{BDH, BDHConfig};
use burn_wgpu::{self, Wgpu};
use criterion::{Criterion, criterion_group, criterion_main};

fn inference_bench(c: &mut Criterion) {
    type Backend = Wgpu<f32>;
    <Backend as BackendTrait>::seed(42);
    let device = <Backend as BackendTrait>::Device::default();
    burn_wgpu::init_setup::<burn_wgpu::graphics::AutoGraphicsApi>(&device, Default::default());

    let model = BDH::<Backend>::new(BDHConfig::default(), &device);

    let batch_size = 8;
    let block_size = 128;
    let tokens: Vec<i64> = (0..(batch_size * block_size))
        .map(|idx| (idx % 255) as i64)
        .collect();
    let input = Tensor::<Backend, 2, Int>::from_data(
        TensorData::new(tokens, [batch_size, block_size]),
        &device,
    );

    c.bench_function("bdh_inference_forward", |b| {
        b.iter(|| {
            let _ = model.forward(input.clone());
        })
    });
}

criterion_group!(benches, inference_bench);
criterion_main!(benches);
