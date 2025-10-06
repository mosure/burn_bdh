use burn::LearningRate;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Int, Tensor, TensorData};
use burn_autodiff::Autodiff;
use burn_dragon_hatchling::{BDH, BDHConfig, language_model_loss};
use burn_ndarray::NdArray;
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};

fn training_step_bench(c: &mut Criterion) {
    type Backend = Autodiff<NdArray<f32>>;
    <Backend as BackendTrait>::seed(24);
    let device = <Backend as BackendTrait>::Device::default();

    let base_model = BDH::<Backend>::new(BDHConfig::default(), &device);
    let batch_size = 4;
    let block_size = 64;
    let input_tokens: Vec<i64> = (0..(batch_size * block_size))
        .map(|idx| (idx % 255) as i64)
        .collect();
    let target_tokens: Vec<i64> = input_tokens
        .iter()
        .map(|tok| ((tok + 1) % 255) as i64)
        .collect();

    let inputs = Tensor::<Backend, 2, Int>::from_data(
        TensorData::new(input_tokens, [batch_size, block_size]),
        &device,
    );
    let targets = Tensor::<Backend, 2, Int>::from_data(
        TensorData::new(target_tokens, [batch_size, block_size]),
        &device,
    );

    let optimizer_config = AdamWConfig::new().with_weight_decay(0.1);
    let lr: LearningRate = 1e-3;

    c.bench_function("bdh_single_train_step", |b| {
        b.iter_batched(
            || {
                let model = base_model.clone();
                let optimizer = optimizer_config.clone().init::<Backend, BDH<Backend>>();
                (model, optimizer)
            },
            |(mut model, mut optimizer)| {
                let logits = model.forward(inputs.clone());
                let loss = language_model_loss::<Backend>(logits, targets.clone());
                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &model);
                model = optimizer.step(lr, model, grads);
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, training_step_bench);
criterion_main!(benches);
