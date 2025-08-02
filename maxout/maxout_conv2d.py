"""
Author: Raphael Senn <raphaelsenn@gmx.de>
Initial coding: 2025-08-01

Syntax:
    - N : batch_size
    - C : out_channels
    - Hc : height after convolution
    - Wc : width after convolution
    """

from typing import Any

import torch
import torch.nn as nn


class MaxoutConv2d(nn.Module):
    """
    Implementation of the convolutional maxout layer.
    
    Reference:
    Maxout Networks; Goodfellow et al. 2013
    https://arxiv.org/abs/1302.4389 
    """ 
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_pieces: int,
            kernel_size: int,
            stride: int=1,
            padding: int = 0,
            bias: bool=True,
            padding_mode: str="zeros",
            device: Any | None = None,
            dtype: Any | None = None
            ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.num_pieces = num_pieces
        
        self.layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_pieces * out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)           # [N, num_pieces * C, Hc, Wc]
        _, _, Hc, Wc = x.shape
        x = x.view(
            -1, 
            self.out_channels,
            self.num_pieces, 
            Hc,
            Wc
        )                           # [N, C, num_pieces, Hc, Wc]
        x, _ = torch.max(x, dim=2)  # [N, C, Hc, Wc]
        return x