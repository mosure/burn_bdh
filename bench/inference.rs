use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Int, Tensor, TensorData};
use burn_dragon_hatchling::{BDH, BDHConfig};
use burn_ndarray::NdArray;
use criterion::{Criterion, criterion_group, criterion_main};

fn inference_bench(c: &mut Criterion) {
    type Backend = NdArray<f32>;
    <Backend as BackendTrait>::seed(42);
    let device = <Backend as BackendTrait>::Device::default();

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
