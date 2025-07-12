import torch
import torch.nn as nn

from ..data_utils.spec import FeatureSpec, InputSpec, NamedInput


class FeatureEmbedding(nn.Module):
    def __init__(self, feature_spec: FeatureSpec, d_model: int):
        super().__init__()
        self.name = feature_spec.name
        self.feature_spec = feature_spec

        ftype = feature_spec.feature_type

        if ftype == 'real':
            self.embedding = nn.Linear(1, d_model)
        elif ftype == 'categorical':
            self.embedding = nn.Embedding(feature_spec.category_size, d_model)
        else:
            raise ValueError(f'Unsupported feature type: {feature_spec.name}@{feature_spec.feature_type}')

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """
        value:  (B, T)   for time-varying features
                (B,)     for static features
        returns: (B, T, d_model) or (B, 1, d_model)
        """
        ftype = self.feature_spec.feature_type

        if self.feature_spec.feature_class == 'static':
            assert value.dim() == 1, 'static feature has no time step'
            value = value.unsqueeze(-1)  # (B,) -> (B, 1)
        else:
            # for temporal features
            assert value.dim() == 2

        if ftype == 'real':  # real
            value = value.float()  # float input for Linear
            return self.embedding(value.unsqueeze(-1))  # -> (..., d_model)

        elif ftype == 'categorical':  # categorical
            value = value.long()  # long input for Embedding
            return self.embedding(value)  # -> (..., d_model)

class InputEmbedding(nn.Module):
    def __init__(self, input_spec: InputSpec, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

        self.static_embeds = nn.ModuleList(
            [FeatureEmbedding(f, embed_dim) for f in input_spec.get(feature_class='static')]
        )
        self.n_static = len(self.static_embeds)

        self.observed_embeds = nn.ModuleList(
            [FeatureEmbedding(f, embed_dim) for f in input_spec.get(feature_class='observed')]
        )
        self.n_observed = len(self.observed_embeds)

    def _embed_group(self, inputs, embeds):
        reps = []
        for emb in embeds:
            reps.append(emb(inputs[emb.name]))         # each column -> (..., d)
        return torch.stack(reps, dim=-2)                # -> (B, T, n_vars, d)

    def forward(self, inputs: NamedInput):
        """
        inputs: dict[str, Tensor], key = feature.name
          s_embed: (B, n_static, d)
          * time-varying = (B, T)x
        returns:
          s_embed: (B, n_static, d)
          p_embed: (B, T, n_timepos, d)
          o_embed: (B, T, n_observed, d)
          k_embed: (B, T, n_known, d)
        """
        # static: (B, n_static, d)
        s = self._embed_group(inputs, self.static_embeds)    # (B,1,n_s,d)
        s = s.squeeze(1)                                     # (B,n_s,d)

        # observed
        o = self._embed_group(inputs, self.observed_embeds)  # (B,T,n_obs,d)

        p, k = None, None  # currently not implemented
        return s, p, o, k
