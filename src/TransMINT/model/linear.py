from typing import Optional

import torch
from torch import Tensor, nn

from .base import ModelBase
from ..data_utils.spec import InputSpec


# ----------------------- Diagonal-ARX(1) Linear RNN ----------------------- #
class DiagLinearRNNCell(nn.Module):
    """
    h_t = diag(a) h_{t-1} + W_x x_t + b
    where
    * if rho is specified: a = rho * tanh(raw_a)
    * if rho is None     : a = tanh(raw_a)
    * rho must be within (0, 1), better to ~1.
    """

    def __init__(
            self,
            d_input: int,
            d_hidden: int,
            rho: Optional[float] = None,
            init_a: float = 0.95
    ) -> None:
        super().__init__()
        self.d_input, self.d_hidden = d_input, d_hidden
        self.rho = rho

        # a_i for each channel
        self.raw_a = nn.Parameter(torch.empty(d_hidden))
        self.W_x = nn.Linear(d_input, d_hidden, bias=True)

        # initialize a for a longer memory
        init_a = float(max(min(init_a, 0.999), -0.999))
        with torch.no_grad():
            from math import atanh
            self.raw_a.fill_(atanh(init_a))
            nn.init.xavier_uniform_(self.W_x.weight)
            nn.init.zeros_(self.W_x.bias)

    def a_vector(self) -> Tensor:
        a = torch.tanh(self.raw_a)  # keep a within (-1, 1)
        if self.rho is not None:
            a = a * self.rho
        return a  # (d_hidden,)

    def step(self, x_t: Tensor, h_prev: Tensor, a: Tensor) -> Tensor:
        # x_t: (B, d_input), h_prev: (B, d_hidden), a: (d_hidden,)
        return h_prev * a + self.W_x(x_t)

    def forward(self, x: Tensor, h0: Optional[Tensor] = None) -> Tensor:
        B, T, _ = x.shape
        H = self.d_hidden

        # 1) z_t = W_x x_t + b
        z = self.W_x(x)  # (B, T, H)

        a = self.a_vector()  # (H,)

        # 2) p[t] = a^(t+1) -> cumprod([a, a^2, ...])
        A = a.unsqueeze(0).expand(T, H)  # (T, H)
        p = torch.cumprod(A, dim=0).unsqueeze(0)  # (1, T, H)

        # 3) v_t = sum_{k<=t} z_k / p_k ；h = v * p
        v = torch.cumsum(z / p.clamp_min(1e-12), dim=1)  # (B, T, H)
        h = v * p  # (B, T, H)

        # 4) h0 contribution: + a^(t+1) * h0
        if h0 is not None:
            h = h + p * h0.view(B, 1, H)

        # 5) deal with |a|≈0: h_t ≈ z_t
        zero_mask = (a.abs() < 1e-6).view(1, 1, H)  # (1,1,H)
        h = torch.where(zero_mask, z, h)

        return h


class MinLinear(ModelBase):
    def __init__(
            self,
            input_spec: InputSpec,
            d_model: int,
            output_size: int = 1,
            rho: Optional[float] = None,
    ):
        super().__init__(input_spec)
        n_features = input_spec.count(feature_class='observed')
        self.output_size = output_size

        self.rnn_cell = DiagLinearRNNCell(n_features, d_model, rho)
        self.out_proj = nn.Linear(d_model, output_size)
        self.tanh = nn.Tanh()

    def forward(self, inputs) -> Tensor:
        rnn_cell = self.rnn_cell

        x = self._observed_features(inputs)  # B, T, n_features
        h_out = self.rnn_cell(x)
        return self.tanh(self.out_proj(h_out))  # bound to [-1, 1]
