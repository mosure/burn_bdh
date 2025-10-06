use std::f32::consts::PI;

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use super::block_sparse::BlockPattern2d;

pub fn fused_state_aligned<B: Backend>(
    query: Tensor<B, 4>,
    value: Tensor<B, 4>,
    freqs: Tensor<B, 4>,
    alibi_slopes: Option<Tensor<B, 1>>,
    layout: &BlockPattern2d,
) -> Tensor<B, 4> {
    let device = query.device();
    let [batch, heads, time, _dim_q] = query.shape().dims::<4>();
    let dim_v = value.shape().dims::<4>()[3];

    let positions = Tensor::<B, 1, Int>::arange(0..time as i64, &device)
        .float()
        .reshape([1, 1, time, 1]);

    let raw = positions.clone() * freqs;
    let phases = (raw.clone() - raw.floor()) * (2.0 * PI);
    let (q_rot, k_rot) = apply_rope::<B>(phases, query.clone());

    let value = value.repeat_dim(1, heads);
    let mut outputs: Vec<Tensor<B, 4>> = Vec::new();

    let block_size = usize::max(layout.block_size(), 1);
    let total_blocks = (time + block_size - 1) / block_size;
    let (slopes, use_alibi) = match alibi_slopes {
        Some(tensor) => (tensor.reshape([1, heads, 1, 1]), true),
        None => (
            Tensor::<B, 1>::zeros([heads], &device).reshape([1, heads, 1, 1]),
            false,
        ),
    };

    for row in 0..total_blocks {
        let row_start = row * block_size;
        let row_end = usize::min(row_start + block_size, time);
        let row_len = row_end - row_start;
        let row_range = row_start..row_end;
        let q_block = q_rot.clone().slice_dim(2, row_range.clone());

        let mut block_acc = Tensor::<B, 4>::zeros([batch, heads, row_len, dim_v], &device);

        let cols = layout.iter_cols(row, total_blocks);
        if cols.is_empty() {
            outputs.push(block_acc);
            continue;
        }

        let pos_row = positions
            .clone()
            .slice_dim(2, row_range.clone())
            .reshape([1, 1, row_len, 1]);

        for col in cols {
            if !layout.is_active(row, col) {
                continue;
            }

            let col_start = col * block_size;
            let col_end = usize::min(col_start + block_size, time);
            let col_len = col_end - col_start;
            let col_range = col_start..col_end;

            let k_block = k_rot.clone().slice_dim(2, col_range.clone());
            let mut scores = q_block.clone().matmul(k_block.swap_dims(2, 3));

            if row == col {
                scores = scores.tril(-1);
            }

            let pos_col = positions
                .clone()
                .slice_dim(2, col_range.clone())
                .reshape([1, 1, 1, col_len]);

            if use_alibi {
                let alibi = slopes.clone() * (pos_col - pos_row.clone());
                scores = scores + alibi;
            }

            let v_block = value.clone().slice_dim(2, col_range);

            let contribution = scores.matmul(v_block);
            block_acc = block_acc + contribution;
        }

        outputs.push(block_acc);
    }

    let attn = Tensor::cat(outputs, 2);
    attn
}

fn apply_rope<B: Backend>(
    phases: Tensor<B, 4>,
    values: Tensor<B, 4>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let cos = phases.clone().cos();
    let sin = phases.sin();

    let [b, h, t, n] = values.shape().dims();
    let pairs = values.clone().reshape([b, h, t, n / 2, 2]);

    let even = pairs.clone().slice_dim(4, 0..1).squeeze::<4>(4);
    let odd = pairs.slice_dim(4, 1..2).squeeze::<4>(4);

    let rotated = Tensor::stack::<5>(vec![odd.clone().neg(), even], 4).reshape([b, h, t, n]);

    let rot = values * cos.clone() + rotated * sin;
    (rot.clone(), rot)
}

pub fn default_alibi_slopes(n_head: usize) -> Vec<f32> {
    if n_head == 0 {
        return Vec::new();
    }

    let mut slopes = Vec::with_capacity(n_head);
    for idx in 0..n_head {
        let ratio = idx as f32 / n_head as f32;
        slopes.push(1.0 / (2.0_f32.powf(ratio)));
    }
    slopes
}
