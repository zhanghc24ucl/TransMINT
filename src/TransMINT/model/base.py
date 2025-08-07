import torch
from torch import nn

from ..data_utils.spec import InputSpec


class ModelBase(nn.Module):
    def __init__(self, input_spec: InputSpec):
        super().__init__()
        self.input_spec = input_spec

    def _observed_features(self, inputs):
        obs = []
        for k in self.input_spec.features:
            if k.feature_class != 'observed':
                continue
            obs.append(inputs[k.name].float())
        return torch.cat(obs, dim=-1)  # (B, T, k)


class MinLinear(ModelBase):
    def __init__(self, input_spec: InputSpec, output_size: int = 1):
        super().__init__(input_spec)
        n_features = input_spec.count(feature_class='observed')
        self.output_size = output_size
        self.out_proj = nn.Linear(n_features, output_size)
        self.tanh = nn.Tanh()

    def forward(self, inputs) -> torch.Tensor:
        obs = self._observed_features(inputs)
        return self.tanh(self.out_proj(obs))  # bound to [-1, 1]
