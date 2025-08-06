import torch
from torch import nn

from .base import ModelBase


class MINLSTM(ModelBase):
    def __init__(
            self,
            input_spec,
            d_model: int = 16,
            dropout: float = 0.,
            num_layers: int = 1,
            output_size: int = 1
    ):
        super().__init__(input_spec)
        self.d_model = d_model
        n_features = input_spec.count(feature_class='observed')

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.out_proj = nn.Linear(d_model, output_size)
        self.tanh = nn.Tanh()

    def forward(self, inputs) -> torch.Tensor:
        obs = self._observed_features(inputs)
        dec, _ = self.lstm(obs)
        return self.tanh(self.out_proj(dec))  # (B, T, 1)
