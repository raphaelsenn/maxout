import torch
import torch.nn as nn


class MaxNorm(object):
    def __init__(self, max_value: int=2, dim: int=0) -> None:
        self.max_value = max_value
        self.dim = dim

    def __call__(self, m: nn.Module) -> None:
        if hasattr(m, 'weight'):
            norms = torch.norm(m.weight.data, dim=self.dim, keepdim=True)
            w = m.weight.data.clamp(norms, self.max_value)
            m.weight.data = w