from typing import Optional

import torch
import torch.nn as nn

from .base import ModelBase
from .embedding import InputEmbedding
from .fusion import FusionDecoderModule, FusionLSTMModule, FusionVN, FusionVSN
from ..data_utils.spec import InputSpec


class FusionTransformer(ModelBase):
    def __init__(
            self,
            input_spec: InputSpec,
            d_model: int,
            num_heads: int = 4,
            output_size: int = 1,
            dropout: float = 0.1,
            trainable_skip_add: bool=False,
            d_static: Optional[int] = None,
            d_observed: Optional[int] = None,
            is_lite: bool = False,
    ):
        super().__init__(input_spec)
        self.d_model = embed_dim = d_model
        self.num_heads = num_heads
        self.output_size = output_size

        # 1) Input embedding
        self.input_embedding = embedding = InputEmbedding(
            input_spec,
            embed_dim,
            static_dim=d_static,
            observed_dim=d_observed,
        )

        # 2) Variable Selection Network
        n_static = embedding.n_features['static']
        vsn_clz = FusionVN if is_lite else FusionVSN
        self.vsn = vsn_clz(
            n_static=n_static,
            n_observed=embedding.n_features['observed'],
            d_model=d_model,
            dropout=dropout,
            trainable_skip_add=trainable_skip_add,
            d_static=d_static,
            d_observed=d_observed,
        )

        # 3) Decoder
        self.decoder = FusionDecoderModule(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            n_static=n_static,
        )

        # 4) Output
        self.out_proj = nn.Linear(d_model, output_size)
        self.tanh = nn.Tanh()

    def forward(self, inputs) -> torch.Tensor:
        # ---- 0. input embedding ----
        # FIXME: currently do not process time_pos embedding
        static_embed, time_embed, obs_embed = self.input_embedding(inputs)

        # ---- 1. Variable selection ----
        obs_out, static_out = self.vsn(obs_embed, static_embed)

        # ---- 2. Decoder self-attention ----
        dec_out = self.decoder(obs_out, static_out)

        # ---- 3. Output ----
        return self.tanh(self.out_proj(dec_out))  # [B, T, output_size]


class MINTransformer(ModelBase):
    def __init__(
            self,
            input_spec: InputSpec,
            d_model: int,
            num_heads: int = 4,
            output_size: int = 1,
            dropout: float = 0.1,
            trainable_skip_add: bool=False,
            d_static: Optional[int] = None,
            d_observed: Optional[int] = None,
            is_lite: bool = False,
    ):
        super().__init__(input_spec)
        self.d_model = embed_dim = d_model
        self.num_heads = num_heads
        self.output_size = output_size

        # 1) Input embedding
        self.input_embedding = embedding = InputEmbedding(
            input_spec,
            embed_dim,
            static_dim=d_static,
            observed_dim=d_observed,
        )

        # 2) Variable Selection Network
        n_static = embedding.n_features['static']
        vsn_clz = FusionVN if is_lite else FusionVSN
        self.vsn = vsn_clz(
            n_static=n_static,
            n_observed=embedding.n_features['observed'],
            d_model=d_model,
            dropout=dropout,
            trainable_skip_add=trainable_skip_add,
            d_static=d_static,
            d_observed=d_observed,
        )

        # 3) LSTM
        self.lstm = FusionLSTMModule(
            d_model=d_model,
            dropout=dropout,
            n_static=n_static,
        )

        # 4) Decoder
        self.decoder = FusionDecoderModule(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
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

        # ---- 3. Decoder self-attention ----
        dec_out = self.decoder(lstm_out, static_out)

        # ---- 4. Output ----
        return self.tanh(self.out_proj(dec_out))  # [B, T, output_size]
