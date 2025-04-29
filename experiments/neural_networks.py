import torch
import torch.nn as nn

from maxout.maxout import MaxOut
from maxout.maxout import MaxOutConv2d


class MaxoutMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.dropout_input = nn.Dropout(p=0.2)
        self.maxout1 = MaxOut(in_features=28 * 28, out_features=240, k_groups=5)

        self.dropout1 = nn.Dropout(p=0.5)
        self.maxout2 = MaxOut(in_features=240, out_features=240, k_groups=5)
        
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features=240, out_features=10, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout_input(x) 
        x = self.maxout1(x)

        x = self.dropout1(x)
        x = self.maxout2(x)

        x = self.dropout2(x)
        x = self.fc(x)
        return x


class MaxoutConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__() 

        self.maxout1 = MaxOutConv2d(
            in_channels=3,
            out_channels=96,
            k_groups=2,
            kernel_size=8,
            stride=1,
            padding=4,
            pool_size=4,
            pool_stride=2)


        self.maxout2 = MaxOutConv2d(
            in_channels=96,
            out_channels=192,
            k_groups=2,
            kernel_size=8,
            stride=2,
            padding=3,
            pool_size=4,
            pool_stride=2)

        self.maxout3 = MaxOutConv2d(
            in_channels=192,
            out_channels=192,
            k_groups=2,
            kernel_size=5,
            stride=2,
            padding=3,
            pool_size=2,
            pool_stride=2
        )

        self.maxout_fc = MaxOut(in_features=192, out_features=500, k_groups=5)

        self.out = nn.Linear(in_features=500, out_features=10, bias=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxout1(x)
        x = self.maxout2(x)
        x = self.maxout3(x)
        x = x.flatten(start_dim=1)
        x = self.maxout_fc(x)
        return self.out(x)