import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Literal, Type, TypeVar, Optional
from enum import Enum
from .MoE import GumbelGatingNetwork


class GumbelArgmaxGatingNetwork(GumbelGatingNetwork):
    """
    MLP gating with BayesianLinear for each layer.
    We'll provide a method to sum the KL from both layers.
    """

    def __init__(self, in_features=784, num_experts=3, tau=1.0):
        super().__init__(in_features, num_experts, tau)

    def forward(self, x):
        # return argmax indices
        logits = self.lin(x)  # shape: (batch, num_experts) or (num_samples, batch, num_experts)
        if self.training:
            selected_experts = F.gumbel_softmax(logits, dim=-1, tau=self.tau, hard=True)
            # selected_experts = F.softmax(logits, dim=-1)
            counts = torch.bincount(selected_experts.argmax(dim=-1), minlength=logits.shape[-1])  # [num_experts]
            # print('train expert counts')
            # print(counts)
            print('--')
            # print("Weights:", self.lin.weight.data)
            # print("Bias:", self.lin.bias.data)
        else:
            indices = torch.argmax(logits, dim=-1)
            selected_experts = F.one_hot(indices, num_classes=logits.shape[-1]).float()
            # selected_experts = F.gumbel_softmax(logits, dim=-1, tau=self.tau, hard=True).argmax(dim=-1)
            counts = torch.bincount(selected_experts.argmax(dim=-1), minlength=logits.shape[-1])  # [num_experts]
            print(logits.shape)
            print('eval expert counts')
            print(counts)
        print(F.softmax(logits, dim=-1))
        return selected_experts


class SparseMoE(nn.Module):
    def __init__(
            self,
            *,
            d_in: Optional[int] = None,
            d_out: int,
            n_blocks: int,
            d_block_per_expert: int,
            dropout: Optional[float] = None,
            activation: str = 'ReLU',
            num_experts: Optional[int] = None,
            tau: float = 1.0,
            type: Literal['argmax'] = 'argmax',
    ) -> None:
        assert d_out is not None, "the output layer must be added to the MoE"
        super().__init__()

        self.num_experts = num_experts
        self.type = type
        self.d_out = d_out
        d_first = d_block_per_expert if d_in is None else d_in

        # self.experts = nn.ModuleList([
        #     nn.ModuleList([
        #         nn.Sequential(
        #             nn.Linear(d_first if i == 0 else d_block, d_block),
        #             getattr(nn, activation)() if (i < n_blocks) else nn.Identity(),
        #             nn.Dropout(dropout) if (i < n_blocks) and (dropout is not None) else nn.Identity()
        #         )
        #         for i in range(n_blocks + 1)
        #     ])
        #     for _ in range(self.num_experts)
        # ])
        self.experts = nn.ModuleList()

        for _ in range(self.num_experts):
            layers = []
            for i in range(n_blocks + 1):
                layers.append(nn.Linear(d_first if i == 0 else d_block_per_expert,
                                        d_out if i == n_blocks else d_block_per_expert))

                if i < n_blocks:
                    layers.append(getattr(nn, activation)())
                    if dropout is not None:
                        layers.append(nn.Dropout(dropout))

            self.experts.append(nn.Sequential(*layers))
        self.gate = GumbelArgmaxGatingNetwork(d_first, num_experts, tau=tau)

    def forward(self, x: Tensor) -> Tensor:
        selected_experts = self.gate(x)

        # Prepare output tensor
        output = torch.zeros(x.size(0), self.d_out, device=x.device, dtype=x.dtype)

        # Process only the required experts per sample
        # for expert_idx in range(self.num_experts):
        #     mask = (selected_experts == expert_idx)
        #     if mask.any():
        #         x_selected = x[mask]
        #         out_selected = self.experts[expert_idx](x_selected)
        #         output[mask] = out_selected
        # For each expert, process the inputs routed to it
        for expert_idx in range(self.num_experts):
            # Get mask of samples assigned to this expert
            mask = selected_experts[:, expert_idx] > 1e-5  # [batch]
            if mask.any():
                x_selected = x[mask]
                out_selected = self.experts[expert_idx](x_selected) * selected_experts[mask, expert_idx].unsqueeze(-1)
                output[mask] = out_selected
        return output
