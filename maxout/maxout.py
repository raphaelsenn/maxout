"""
Author: Raphael Senn <raphaelsenn@gmx.de>
Initial coding: 2025-08-01
"""

from typing import Any

import torch
import torch.nn as nn


class Maxout(nn.Module):
    """
    Implementation of the fully-connected maxout layer.

    Reference:
    Maxout Networks; Goodfellow et al. 2013
    https://arxiv.org/abs/1302.4389
    """ 
    def __init__(
            self,
            in_features: int,
            num_units: int,
            num_pieces: int,
            bias: bool=True,
            device: Any | None = None,
            dtype: Any | None = None
        ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_units = num_units
        self.num_pieces = num_pieces
        self.bias = bias

        self.layer = nn.Linear(
            in_features, 
            num_pieces * num_units, 
            bias,
            device,
            dtype
        )        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)               # [N, num_pieces * num_units]
        x = x.view(
            -1, 
            self.num_units, 
            self.num_pieces
        )                               # [N, num_units, num_pieces]
        x, _ = torch.max(x, dim=2)      # [N, num_units]
        return x