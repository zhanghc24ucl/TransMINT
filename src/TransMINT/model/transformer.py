from typing import Dict, Tuple

import torch
import torch.nn as nn

from .embedding import InputEmbedding
from .layer import GatedAddNorm, GatedResidualNetwork, InterpretableMultiHeadAttention, VariableSelectionNetwork
from ..data_utils.spec import InputSpec


class MINTransformer(nn.Module):
    def __init__(
            self,
            input_spec: InputSpec,
            d_model: int,
            num_heads: int = 4,
            output_size: int = 1,
            dropout: float = 0.1,
            trainable_skip_add: bool=False,
    ):
        super().__init__()
        self.d_model = embed_dim = d_model
        self.num_heads = num_heads
        self.output_size = output_size

        # 1) Input embedding
        self.input_embedding = embedding = InputEmbedding(input_spec, embed_dim)

        # 2) Variable Selection Network
        self.vsn_static = VariableSelectionNetwork(
            num_vars=embedding.n_static,
            input_dim=d_model,
            hidden_size=d_model,
            dropout=dropout,
            context_size=None,  # no context for static variables
            trainable_skip_add=trainable_skip_add,
        )
        self.vsn_observed = VariableSelectionNetwork(
            num_vars=embedding.n_observed,
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
        self.static_ctx_enrich = GatedResidualNetwork(
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
            batch_first=True
        )
        self.lstm_gate_addnorm = GatedAddNorm(
            input_size=d_model,
            dropout=dropout,
            trainable_add=False
        )

        # 5) Decoder
        self.enrich_grn = GatedResidualNetwork(
            input_size=d_model,
            hidden_size=d_model,
            output_size=d_model,
            dropout=dropout,
            context_size=d_model,
        )
        self.attention = InterpretableMultiHeadAttention(d_model, num_heads, dropout)
        self.attn_gate_addnorm = GatedAddNorm(
            input_size=d_model,
            dropout=dropout,
            trainable_add=False
        )
        self.post_grn = GatedResidualNetwork(
            input_size=d_model,
            hidden_size=d_model,
            output_size=d_model,
            dropout=dropout,
        )

        # 6) Output
        self.out_proj = nn.Linear(d_model, output_size)
        self.tanh = nn.Tanh()

    def _causal_mask(self, T: int, device):
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        return ~mask  # True for visible

    def forward(self, inputs) -> Tuple[torch.Tensor, Dict]:
        device = inputs.device
        B, T = inputs.batch_size, inputs.time_step

        # ---- 0. input embedding ----
        static_embed, pos_embed, obs_embed, known_embed = self.input_embedding(inputs)

        assert pos_embed is None, 'currently not supported'
        assert known_embed is None, 'currently not supported'

        print('static_shape', static_embed.shape)
        print('observed_shape', obs_embed.shape)

        # ---- 1. Static variable selection ----
        static_out, static_weights = self.vsn_static(static_embed)      # [B,D]

        c_varsel = self.static_ctx_varsel(static_out)                   # [B,D]
        c_enrich = self.static_ctx_enrich(static_out)                   # [B,D]
        h0 = self.state_h(static_out).unsqueeze(0)                      # [1,B,d]
        c0 = self.state_c(static_out).unsqueeze(0)                      # [1,B,d]

        # ---- 2. Temporal variable selection ----
        obs_out, obs_weights = self.vsn_observed(obs_embed, c_varsel)

        # ---- 3. LSTM ----
        lstm_out, _ = self.lstm(obs_out, (h0, c0))               # [B,T,d]
        lstm_out = self.lstm_gate_addnorm(lstm_out, obs_out)

        # ---- 4. Static enrichment ----
        c_enrich = c_enrich.unsqueeze(1)    # [B,D]->[B,1,D]
        enriched = self.enrich_grn(lstm_out, c_enrich)

        # ---- 5. Decoder self-attention ----
        mask = self._causal_mask(T, device).unsqueeze(0).expand(B, -1, -1)
        attn_out, attn_maps = self.attention(enriched, enriched, enriched, mask)
        attn_out = self.attn_gate_addnorm(attn_out, enriched)
        dec = self.post_grn(attn_out)

        # ---- 6. Output ----
        out = self.tanh(self.out_proj(dec))  # [B,T,output_size]

        verbose = {
            'var_static_weights': static_weights, # [B,n_static]
            'var_observed_weights': obs_weights,  # [B,T,n_observed]
            'attn_maps': attn_maps,               # [H,B,T,T]
        }
        return out, verbose
