import torch
import torch.nn as nn


class MaxOut(nn.Module):
    """
    Applies the maxout unit function (for multiplayer perceptrons),
    instead of applying an element-wise function,
    maxout units devide x into groups of k values.
    Each maxout unit then outputs the maximum element of one of these groups.

    Args:
        in_featuares: size of each input sample
        out_features: size of each output sample
        k_groups: number of groups to divide input
    
    Attributes:

    
    """ 
    def __init__(self,
                 in_features: int,
                 out_features: int, 
                 k_groups: int) -> None:
        super().__init__() 

        # transforms the input x into groups of k values, then take the maximum.
        self.linear_layers = nn.ModuleList(
            [nn.Linear(in_features=in_features, out_features=k_groups, bias=True)
             for _ in range(out_features)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [torch.max(layer(x), dim=1).values
             for layer in self.linear_layers] ,dim=1)


class MaxOutConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            k_groups: int,
            kernel_size: int,
            padding: int,
            stride: int,
            pool_size: int,
            pool_stride: int
        ) -> None:
        super().__init__()

        # assert out_channels % k_groups == 0, "Number of out_channels must be divisable by k_groups"
        self.C = out_channels * k_groups
        self.k_groups = k_groups

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=k_groups*out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride
        )
        
        self.pool = nn.MaxPool3d(
            kernel_size=(k_groups, pool_size, pool_size),
            stride=pool_stride
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.shape[0], self.C // self.k_groups, self.k_groups, x.shape[2], x.shape[3])
        x = self.pool(x)
        x = x.view(x.shape[0], self.C // self.k_groups,x.shape[2], x.shape[3]) 
        return x