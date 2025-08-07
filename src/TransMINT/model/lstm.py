import torch
from torch import nn

from .base import ModelBase
from .embedding import InputEmbedding
from .layer import GatedAddNorm, GatedResidualNetwork, VariableSelectionNetwork


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
        return self.tanh(self.out_proj(dec))  # (B, T, 1)


class MinFusionLSTM(ModelBase):
    def __init__(
            self,
            input_spec,
            d_model: int = 16,
            dropout: float = 0.,
            output_size: int = 1,
            trainable_skip_add: bool=False,
    ):
        super().__init__(input_spec)
        self.d_model = embed_dim = d_model

        # 1) Input embedding
        self.input_embedding = embedding = InputEmbedding(input_spec, embed_dim)

        # 2) Variable Selection Network
        self.vsn_static = VariableSelectionNetwork(
            num_vars=embedding.n_features['static'],
            input_dim=d_model,
            hidden_size=d_model,
            dropout=dropout,
            context_size=None,  # no context for static variables
            trainable_skip_add=trainable_skip_add,
        )
        self.vsn_observed = VariableSelectionNetwork(
            num_vars=embedding.n_features['observed'],
            input_dim=d_model,
            hidden_size=d_model,
            dropout=dropout,
            context_size=d_model,  # d_model context for observed variables
            trainable_skip_add=trainable_skip_add,
        )

        # 3) Static context GRN
        self.static_ctx_varsel = GatedResidualNetwork(
            input_size=d_model,
            hidden_size=d_model,
            output_size=d_model,
            dropout=dropout,
        )
        self.state_h = GatedResidualNetwork(
            input_size=d_model,
            hidden_size=d_model,
            output_size=d_model,
            dropout=dropout,
        )
        self.state_c = GatedResidualNetwork(
            input_size=d_model,
            hidden_size=d_model,
            output_size=d_model,
            dropout=dropout,
        )

        # 4) LSTM
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            batch_first=True,
        )
        self.lstm_gate_addnorm = GatedAddNorm(
            input_size=d_model,
            dropout=dropout,
            trainable_add=False
        )

        # 5) Output
        self.out_proj = nn.Linear(d_model, output_size)
        self.tanh = nn.Tanh()

    def forward(self, inputs) -> torch.Tensor:
        # ---- 0. input embedding ----
        # FIXME: currently do not process time_pos embedding
        static_embed, time_embed, obs_embed = self.input_embedding(inputs)

        # ---- 1. Static variable selection ----
        static_out, static_weights = self.vsn_static(static_embed)      # [B,D]

        c_varsel = self.static_ctx_varsel(static_out)                   # [B,D]
        h0 = self.state_h(static_out).unsqueeze(0)                      # [1,B,d]
        c0 = self.state_c(static_out).unsqueeze(0)                      # [1,B,d]

        # ---- 2. Temporal variable selection ----
        obs_out, obs_weights = self.vsn_observed(obs_embed, c_varsel)

        # ---- 3. LSTM ----
        lstm_out, _ = self.lstm(obs_out, (h0, c0))               # [B,T,d]
        lstm_out = self.lstm_gate_addnorm(lstm_out, obs_out)

        return self.tanh(self.out_proj(lstm_out))  # (B, T, 1)
