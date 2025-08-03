import torch
from torch import nn

from ..data_utils.spec import InputSpec


class ModelBase(nn.Module):
    def __init__(self, input_spec: InputSpec):
        super().__init__()
        self.input_spec = input_spec

    def get_l1_penalty(self):
        return None


class MINLinear(ModelBase):
    def __init__(self, input_spec: InputSpec, output_size: int = 1, time_step: int = 1):
        super().__init__(input_spec)
        n_features = input_spec.count(feature_class='observed')
        self.time_step = time_step
        self.output_size = output_size
        self.out_proj = nn.Linear(time_step * n_features, time_step * output_size)
        self.tanh = nn.Tanh()

    def forward(self, inputs) -> torch.Tensor:
        assert inputs.time_step == self.time_step
        obs = []
        for k in self.input_spec.features:
            if k.feature_class != 'observed':
                continue
            obs.append(inputs[k.name].float())
        obs = torch.cat(obs, dim=-1)  # (B, T, k)
        B, T, k = obs.shape
        obs_flat = obs.view(B, T * k)  # (B, T * k)

        out_flat = self.out_proj(obs_flat)  # (B, T * output)
        out = out_flat.view(B, T, self.output_size)  # (B, T, output_size)
        return self.tanh(out)  # bound to [-1, 1]

    def get_l1_penalty(self):
        return torch.linalg.vector_norm(self.out_proj.weight, ord=1)
