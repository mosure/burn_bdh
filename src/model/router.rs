use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution as TensorDistribution, Tensor, activation};

#[derive(Module, Debug)]
pub struct Router<B: Backend> {
    experts: usize,
    n_head: usize,
    latent_per_head: usize,
    latent_per_expert: usize,
    weight: Param<Tensor<B, 3>>,
    bias: Param<Tensor<B, 2>>,
}

impl<B: Backend> Router<B> {
    pub fn new(
        experts: usize,
        n_head: usize,
        n_embd: usize,
        latent_per_head: usize,
        device: &B::Device,
    ) -> Self {
        assert!(experts >= 1, "router requires at least one expert");
        assert!(
            latent_per_head % experts == 0,
            "latent size {latent_per_head} must be divisible by experts {experts}"
        );

        let weight = Tensor::<B, 3>::random(
            [n_head, n_embd, experts],
            TensorDistribution::Normal(0.0, 0.02),
            device,
        );
        let bias = Tensor::<B, 2>::zeros([n_head, experts], device);

        Self {
            experts,
            n_head,
            latent_per_head,
            latent_per_expert: latent_per_head / experts,
            weight: Param::from_tensor(weight),
            bias: Param::from_tensor(bias),
        }
    }

    pub fn route(&self, gating_input: Tensor<B, 4>, activations: Tensor<B, 4>) -> Tensor<B, 4> {
        if self.experts == 1 {
            return activations;
        }

        let weight = self.weight.val().unsqueeze_dim::<4>(0);
        let mut logits = gating_input.matmul(weight);
        let bias = self.bias.val().reshape([1, self.n_head, 1, self.experts]);
        logits = logits + bias;

        let routing = activation::softmax(logits, 3);

        let [batch, heads, time, latent] = activations.shape().dims();
        debug_assert_eq!(heads, self.n_head);
        debug_assert_eq!(latent, self.latent_per_head);

        let routed = activations.reshape([
            batch,
            self.n_head,
            time,
            self.experts,
            self.latent_per_expert,
        ]);

        let routing = routing.unsqueeze_dim::<5>(4);
        let weighted = routing * routed;

        weighted.reshape([batch, self.n_head, time, self.latent_per_head])
    }

    pub fn experts(&self) -> usize {
        self.experts
    }
}
