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
