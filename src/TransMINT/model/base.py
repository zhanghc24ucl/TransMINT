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
        return torch.cat(obs, dim=-1)  # (B, T, N)


def causal_mask(T: int, device):
    """
    Build a causal attention mask of shape (T, T).

    Semantics:
      - dtype: bool
      - True  -> masked (disallowed), will be filled with -inf in attention scores
      - False -> allowed

    For query time i and key time j, we mask j > i (future positions):

        i\j    0      1      2
        0    False   True   True
        1    False  False   True
        2    False  False  False
    """
    return torch.triu(torch.ones((T, T), dtype=torch.bool, device=device), diagonal=1)
