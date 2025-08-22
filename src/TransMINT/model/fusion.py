from typing import Optional, Tuple

from torch import Tensor, nn

from .base import causal_mask
from .layer import GatedAddNorm, GatedResidualNetwork, InterpretableMultiHeadAttention, VariableNetwork, \
    VariableSelectionNetwork


class FusionLSTMModule(nn.Module):
    def __init__(
            self,
            d_model: int,
            dropout: float,
            n_static: int = 0,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            batch_first=True,
        )
        if n_static > 0:
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
        else:
            self.state_h = None
            self.state_c = None
        self.gate_addnorm = GatedAddNorm(
            input_size=d_model,
            dropout=dropout,
            trainable_add=False
        )

    def forward(
            self,
            obs_input: Tensor,
            static_input: Optional[Tensor] = None
    ) -> Tensor:
        if static_input is None:
            # no static features
            assert self.state_c is None  # state_h is also None
            lstm_out, _ = self.lstm(obs_input)
        else:
            h0 = self.state_h(static_input).unsqueeze(0)  # [B, D] -> [1, B, D]
            c0 = self.state_c(static_input).unsqueeze(0)  # [B, D] -> [1, B, D]
            lstm_out, _ = self.lstm(obs_input, (h0, c0))
        return self.gate_addnorm(lstm_out, obs_input)


class FusionDecoderModule(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            dropout: float,
            n_static: int = 0,
    ):
        super().__init__()

        if n_static > 0:
            self.static_ctx_enrich = GatedResidualNetwork(
                input_size=d_model,
                hidden_size=d_model,
                output_size=d_model,
                dropout=dropout,
            )
        else:
            self.static_ctx_enrich = None

        # Decoder
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
        self.decoder_gate_addnorm = GatedAddNorm(
            input_size=d_model,
            dropout=0.0,  # shut down dropout for decoder output
            trainable_add=False
        )

    def forward(
            self,
            obs_input: Tensor,
            static_input: Optional[Tensor] = None
    ) -> Tensor:
        B, T, D = obs_input.shape
        device = obs_input.device

        if static_input is None:
            assert self.static_ctx_enrich is None
            enriched = self.enrich_grn(obs_input)
        else:
            c_enrich = self.static_ctx_enrich(static_input)
            c_enrich = c_enrich.unsqueeze(1)  # [B, D] -> [B, 1, D]
            enriched = self.enrich_grn(obs_input, c_enrich)

        mask = causal_mask(T, device).unsqueeze(0).expand(B, -1, -1)
        attn_out, attn_maps = self.attention(enriched, enriched, enriched, mask)
        attn_out = self.attn_gate_addnorm(attn_out, enriched)
        dec = self.post_grn(attn_out)

        return self.decoder_gate_addnorm(dec, obs_input)



class FusionVSN(nn.Module):
    def __init__(
            self,
            n_static: int,
            n_observed: int,
            d_model: int,
            dropout: float,
            trainable_skip_add: bool = False,
            d_static: Optional[int] = None,
            d_observed: Optional[int] = None,
    ):
        super().__init__()

        if n_static > 0:
            self.vsn_static = VariableSelectionNetwork(
                num_vars=n_static,
                input_dim=d_static or d_model,
                hidden_size=d_model,
                dropout=dropout,
                context_size=None,  # no context for static variables
                trainable_skip_add=trainable_skip_add,
            )
            self.static_ctx_varsel = GatedResidualNetwork(
                input_size=d_model,
                hidden_size=d_model,
                output_size=d_model,
                dropout=dropout,
            )
        else:
            self.vsn_static = None
            self.static_ctx_varsel = None

        assert n_observed > 0
        self.vsn_observed = VariableSelectionNetwork(
            num_vars=n_observed,
            input_dim=d_observed or d_model,
            hidden_size=d_model,
            dropout=dropout,
            context_size=d_model,  # d_model context for observed variables
            trainable_skip_add=trainable_skip_add,
        )

    def forward(
            self,
            embedded_obs_input: Tensor,
            embedded_static_input: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if embedded_static_input is None:
            assert self.vsn_static is None  # static_ctx_varsel is also None
            static_out, static_weights, c_varsel = None, None, None
        else:
            static_out, static_weights = self.vsn_static(embedded_static_input)  # [B, D]
            c_varsel = self.static_ctx_varsel(static_out)                        # [B, D]

        obs_out, obs_weights = self.vsn_observed(embedded_obs_input, c_varsel)
        return obs_out, static_out


class FusionVN(nn.Module):
    def __init__(
            self,
            n_static: int,
            n_observed: int,
            d_model: int,
            dropout: float,
            trainable_skip_add: bool = False,
            d_static: Optional[int] = None,
            d_observed: Optional[int] = None,
    ):
        super().__init__()

        if n_static > 0:
            self.vn_static = VariableNetwork(
                num_vars=n_static,
                input_dim=d_static or d_model,
                hidden_size=d_model,
                dropout=dropout,
                context_size=None,  # no context for static variables
                trainable_skip_add=trainable_skip_add,
            )
            self.static_ctx = GatedResidualNetwork(
                input_size=d_model,
                hidden_size=d_model,
                output_size=d_model,
                dropout=dropout,
            )
        else:
            self.vn_static = None
            self.static_ctx = None

        assert n_observed > 0
        self.var_network = VariableNetwork(
            num_vars=n_observed,
            input_dim=d_observed or d_model,
            hidden_size=d_model,
            dropout=dropout,
            context_size=d_model,
            trainable_skip_add=trainable_skip_add,
        )

    def forward(
            self,
            embedded_obs_input: Tensor,
            embedded_static_input: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if embedded_static_input is None:
            assert self.vn_static is None
            static_out, c_static = None, None
        else:
            static_out = self.vn_static(embedded_static_input)  # [B, D]
            c_static = self.static_ctx(static_out)              # [B, D]

        obs_out = self.var_network(embedded_obs_input, c_static)
        return obs_out, static_out
