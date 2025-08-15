import torch
import torch.nn as nn
from torch import Tensor

from .base import ModelBase
from ..data_utils.spec import InputSpec


class LinearRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, activation=None):
        super().__init__()
        self.ih = nn.Linear(input_size, hidden_size, bias=bias)
        self.hh = nn.Linear(hidden_size, hidden_size, bias=False)
        assert activation in ('tanh', 'relu', None)
        self.activation = activation

    def forward(self, x_t, h_prev=None):
        if h_prev is None:
            h_prev = x_t.new_zeros(x_t.size(0), self.hh.in_features)
        h = self.ih(x_t) + self.hh(h_prev)
        if self.activation == 'tanh':
            h = torch.tanh(h)
        elif self.activation == 'relu':
            h = torch.relu(h)
        # else: no activation
        return h


def _linear_rnn(cell: LinearRNNCell, x, h0=None):  # x: (B,T,In)
    B, T, _ = x.shape
    h = h0
    outs = []
    for t in range(T):
        h = cell(x[:, t, :], h)
        outs.append(h)
    return torch.stack(outs, dim=1), h  # (B,T,H), (B,H)


"""
# ---- PyTorch RNN ----
inp, hid = 5, 7
rnn = nn.RNN(inp, hid, num_layers=1, nonlinearity='tanh', bias=True, batch_first=True, dropout=0)

# ---- Create cell and copy initial parameters from PyTroch RNN ----
cell = LinearRNNCell(inp, hid, bias=True, activation='tanh')
with torch.no_grad():
    cell.ih.weight.copy_(rnn.weight_ih_l0)
    cell.hh.weight.copy_(rnn.weight_hh_l0)
    cell.ih.bias.copy_(rnn.bias_ih_l0 + rnn.bias_hh_l0)

# ---- Comparison ----
B, T = 3, 11
x  = torch.randn(B, T, inp)
h0 = torch.randn(1, B, hid)   # (num_layers * num_directions, B, H)

y_ref, hn_ref = rnn(x, h0)            # y_ref: (B,T,H), hn_ref: (1,B,H)
y_mine, hn_mine = linear_rnn(cell, x, h0.squeeze(0))

print(torch.allclose(y_ref, y_mine, rtol=1e-6, atol=1e-6))   # True
print(torch.allclose(hn_ref.squeeze(0), hn_mine, rtol=1e-6, atol=1e-6))  # True
"""


class MinLinear(ModelBase):
    def __init__(
            self,
            input_spec: InputSpec,
            d_model: int,
            output_size: int = 1,
    ):
        super().__init__(input_spec)
        n_features = input_spec.count(feature_class='observed')
        self.output_size = output_size

        self.rnn_cell = LinearRNNCell(n_features, d_model, activation=None)
        self.out_proj = nn.Linear(d_model, output_size)
        self.tanh = nn.Tanh()

    def forward(self, inputs) -> Tensor:
        x = self._observed_features(inputs)  # B, T, n_features
        h_out, _ = _linear_rnn(self.rnn_cell, x)
        return self.tanh(self.out_proj(h_out))  # bound to [-1, 1]
