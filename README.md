# burn_dragon_hatchling 🔥🐉🐣

[![test](https://github.com/mosure/burn_dragon_hatchling/workflows/test/badge.svg)](https://github.com/Mosure/burn_dragon_hatchling/actions?query=workflow%3Atest)
[![GitHub License](https://img.shields.io/github/license/mosure/burn_dragon_hatchling)](https://raw.githubusercontent.com/mosure/burn_dragon_hatchling/main/LICENSE)
[![crates.io](https://img.shields.io/crates/v/burn_dragon_hatchling.svg)](https://crates.io/crates/burn_dragon_hatchling)


burn inference and training of [dragon hatchling](https://arxiv.org/abs/2509.26507) model


![Alt text](./docs/vocab.png)
![Alt text](./docs/bdh.png)


## features

- [x] cached inference
- [x] training benchmarks and reporting
- [x] multi-stream truncated backpropagation through time
- [ ] adaptive tool discovery
- [ ] conditional (deep) gating
- [ ] document-coherent dataloading and scale mixup
- [ ] episodic memory
- [ ] fused kernels
- [ ] hierarchical, memory-aware recurrent state
- [ ] mixture-of-expert routing
- [ ] multi-modal architecture
- [ ] neuromorphic backend
- [ ] rl reasoning training
- [ ] streaming, sparse synaptic backpropagation
- [ ] temporal neuron dampening


## training

- `cargo run --release` (defaults to the cuda backend)


## benchmarks

- `cargo bench` (executes both wgpu and cuda benchmarks)
- open `target/criterion/report/index.html`


## compatible burn versions

| `burn_dragon_hatchling` | `burn` |
| :--                     | :--    |
| `0.1`                   | `0.18` |


## license
licensed under either of

 - Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
 - MIT license (http://opensource.org/licenses/MIT)

at your option.


## contribution

unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
