from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit"""

    def __init__(
            self,
            input_size: int,
            hidden_size: Optional[int] = None,
            dropout: float = 0.0
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.hidden_size = hidden_size or input_size
        self.fc = nn.Linear(input_size, self.hidden_size * 2)

        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if "bias" in n:
                torch.nn.init.zeros_(p)
            elif "fc" in n:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(x)
        x = self.fc(x)

        from torch.nn.functional import glu
        x = glu(x, dim=-1)
        return x


class SkipLayer(nn.Module):
    """Projects (or passes through) the residual branch so it matches *output_size*."""
    def __init__(
            self,
            input_size: int,
            output_size: int
    ):
        super().__init__()
        self.proj = nn.Linear(input_size, output_size) \
            if input_size != output_size else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        if isinstance(self.proj, nn.Linear):
            nn.init.xavier_uniform_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class AddNorm(nn.Module):
    def __init__(
            self,
            input_size: int,
            trainable_add: bool = False
    ):
        super().__init__()

        # Force skip_size == input_size
        self.input_size = self.skip_size = input_size
        self.trainable_add = trainable_add

        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.input_size, dtype=torch.float))
        else:
            self.register_parameter('mask', None)
        self.norm = nn.LayerNorm(self.input_size)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        if self.trainable_add:
            skip = skip * torch.sigmoid(self.mask) * 2.0

        output = self.norm(x + skip)
        return output


class GatedAddNorm(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: Optional[int] = None,
        trainable_add: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_size = input_size
        # Force skip_size == hidden_size
        self.skip_size = self.hidden_size = hidden_size or input_size
        self.dropout = dropout

        self.glu = GatedLinearUnit(
            self.input_size, hidden_size=self.hidden_size, dropout=self.dropout
        )
        self.add_norm = AddNorm(
            self.hidden_size, trainable_add=trainable_add
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        output = self.glu(x)
        output = self.add_norm(output, skip)
        return output


class GatedResidualNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        dropout: float = 0.0,
        context_size: Optional[int] = None,
        trainable_skip_add: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size or hidden_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.skip_layer = SkipLayer(self.input_size, self.output_size)
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.activation = nn.ELU()

        self.context_fc = nn.Identity() if context_size is None else \
            nn.Linear(self.context_size, self.hidden_size, bias=False)

        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.init_weights()

        self.glu_addnorm = GatedAddNorm(
            input_size=self.hidden_size,
            hidden_size=self.output_size,
            dropout=self.dropout,
            trainable_add=trainable_skip_add,
        )

    def init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, a=0.0, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.zeros_(self.fc1.bias)

        nn.init.kaiming_normal_(self.fc2.weight, a=0.0, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.zeros_(self.fc2.bias)

        if isinstance(self.context_fc, nn.Linear):
            nn.init.xavier_uniform_(self.context_fc.weight)

    def forward(
            self,
            x: Tensor,
            context: Optional[Tensor]=None
    ):
        skip = self.skip_layer(x)

        hidden = self.fc1(x)
        if context is not None:
            hidden = hidden + self.context_fc(context)  # type: ignore[attr-defined]
        hidden = self.activation(hidden)
        hidden = self.fc2(hidden)
        gated_output = self.glu_addnorm(hidden, skip)
        return gated_output

class VariableSelectionNetwork(nn.Module):
    def __init__(
        self,
        num_vars: int,
        input_dim: int,
        hidden_size: int,
        dropout: float = 0.0,
        context_size: Optional[int] = None,
        trainable_skip_add: bool = False,
    ):
        super().__init__()

        self.num_vars = num_vars
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.context_size = context_size

        # GRN that converts the *flattened* embedding tensor into selection logits
        self.flatten_grn = GatedResidualNetwork(
            input_size=num_vars * input_dim,
            hidden_size=hidden_size,
            output_size=num_vars,  # one logit per variable
            dropout=dropout,
            context_size=context_size,
            trainable_skip_add=trainable_skip_add,
        )

        # One GRN per variable to transform its embedding
        self.var_grns = nn.ModuleList(
            [
                GatedResidualNetwork(
                    input_size=input_dim,
                    hidden_size=hidden_size,
                    output_size=hidden_size,
                    dropout=dropout,
                    context_size=None,
                    trainable_skip_add=trainable_skip_add,
                )
                for _ in range(num_vars)
            ]
        )
        self.softmax = nn.Softmax(dim=-1)

    def _broadcast_context(self, context: Tensor, repeat: int) -> Optional[Tensor]:
        if context.dim() == 2:  # (B, C)
            context = context.unsqueeze(1).expand(-1, repeat, -1)  # (B, repeat, C)
        elif context.dim() == 3:  # (B, repeat, C)
            assert context.size(-2) == repeat
        return context  # (B, repeat, C)

    def forward(
        self,
        embedding: Tensor,
        context: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if embedding.dim() == 3:  # (B, N, D) – static
            # static features accept no context
            assert context is None, "Static embedding does not accept context"
            B, N, D = embedding.shape
        elif embedding.dim() == 4:  # (B, T, N, D) – temporal/known
            B, T, N, D = embedding.shape
            if context is not None:
                # Handle context broadcasting if necessary
                context = self._broadcast_context(context, repeat=T)
        else:
            raise ValueError(
                "embedding must have shape (B, N, D) or (B, T, N, D), got"
                f" {embedding.shape}"
            )

        flat = embedding.reshape(*embedding.shape[:-2], N * D)  # (..., N*D)
        logits = self.flatten_grn(flat, context)                # (..., N)
        weights = self.softmax(logits)                          # (..., N)

        transformed_list: List[Tensor] = []
        for i in range(N):
            var_emb = embedding[..., i, :]                      # (..., D)
            trans = self.var_grns[i](var_emb)                   # (..., H)
            transformed_list.append(trans.unsqueeze(-2))        # (..., 1, H)

        transformed = torch.cat(transformed_list, dim=-2)       # (..., N, H)
        combined = weights.unsqueeze(-1) * transformed          # (..., N, H)
        outputs = combined.sum(dim=-2)                          # (..., H)

        return outputs, weights

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.0, mask_bias=-1e9):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.softmax = nn.Softmax(dim=2)
        self.mask_bias = mask_bias

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.permute(0, 2, 1))  # query-key overlap

        temper = torch.as_tensor(
            k.size(-1), dtype=attn.dtype, device=attn.device
        ).sqrt()
        attn = attn / temper

        if mask is not None:
            attn = attn.masked_fill(mask, self.mask_bias)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn

class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable MultiHead Attention with shared Value projection matrix.
    """
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.d_q = self.d_k = self.d_v = d_model // n_head
        self.n_head = n_head

        self.q_layers = nn.ModuleList(
            [nn.Linear(d_model, self.d_q, bias=False) for _ in range(n_head)]
        )
        self.k_layers = nn.ModuleList(
            [nn.Linear(d_model, self.d_k, bias=False) for _ in range(n_head)]
        )
        self.v_layer = nn.Linear(d_model, self.d_v, bias=False)

        self.attention = ScaledDotProductAttention()
        self.w_o = nn.Linear(self.d_v, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            q: Tensor,
            k: Tensor,
            v: Tensor,
            mask: Optional[Tensor]=None
    )-> tuple[Tensor, Tensor]:
        heads = []
        attns = []
        vs = self.v_layer(v)
        for i in range(self.n_head):
            qs = self.q_layers[i](q)
            ks = self.k_layers[i](k)

            head, attn = self.attention(qs, ks, vs, mask)
            head_dropout = self.dropout(head)

            heads.append(head_dropout)
            attns.append(attn)

        head = torch.stack(heads, dim=2) if self.n_head > 1 else heads[0]
        attn = torch.stack(attns, dim=2)

        # take the average of all heads as output
        outputs = torch.mean(head, dim=2) if self.n_head > 1 else head
        outputs = self.w_o(outputs)
        outputs = self.dropout(outputs)
        return outputs, attn  # (H, B, T, T)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=160):
        super().__init__()
        assert (
            d_model % 2 == 0
        ), "model dimension has to be multiple of 2 (encode sin(pos) and cos(pos))"
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = torch.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = torch.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        with torch.no_grad():
            x = x * torch.sqrt(self.d_model)
            seq_len = x.size(0)
            pe = self.pe[:, :seq_len].view(seq_len, 1, self.d_model)
            x = x + pe
            return x
