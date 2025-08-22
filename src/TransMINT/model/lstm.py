from typing import Optional

import torch
from torch import nn

from .base import ModelBase
from .embedding import InputEmbedding
from .fusion import FusionLSTMModule, FusionVN, FusionVSN


class MinLSTM(ModelBase):
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
        return self.tanh(self.out_proj(dec))  # [B, T, output_size]


class FusionLSTM(ModelBase):
    def __init__(
            self,
            input_spec,
            d_model: int = 16,
            dropout: float = 0.,
            output_size: int = 1,
            trainable_skip_add: bool=False,
            d_static: Optional[int] = None,
            d_observed: Optional[int] = None,
            is_lite: bool = False,
    ):
        super().__init__(input_spec)
        self.d_model = embed_dim = d_model

        # 1) Input embedding
        self.input_embedding = embedding = InputEmbedding(
            input_spec,
            embed_dim,
            static_dim=d_static,
            observed_dim=d_observed,
        )

        # 2) Variable selection
        n_static = embedding.n_features['static']
        vsn_clz = FusionVN if is_lite else FusionVSN
        self.vsn = vsn_clz(
            n_static,
            embedding.n_features['observed'],
            d_model,
            dropout,
            trainable_skip_add,
            d_static=d_static,
            d_observed=d_observed,
        )

        # 3) Fusion LSTM
        self.lstm = FusionLSTMModule(
            d_model,
            dropout,
            n_static=n_static,
        )

        # 5) Output
        self.out_proj = nn.Linear(d_model, output_size)
        self.tanh = nn.Tanh()

    def forward(self, inputs) -> torch.Tensor:
        # ---- 0. input embedding ----
        # FIXME: currently do not process time_pos embedding
        static_embed, time_embed, obs_embed = self.input_embedding(inputs)

        # ---- 1. Variable selection ----
        obs_out, static_out = self.vsn(obs_embed, static_embed)

        # ---- 2. LSTM ----
        lstm_out = self.lstm(obs_out, static_out)

        # ---- 3. Output ----
        return self.tanh(self.out_proj(lstm_out))  # [B, T, output_size]
