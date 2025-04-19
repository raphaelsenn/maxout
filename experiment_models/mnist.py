import torch
import torch.nn as nn
import torch.nn.functional as F

from maxout.maxout import MaxOutMLP


class NetMNIST(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.dropout_input = nn.Dropout(p=0.2)
        self.maxout1 = MaxOutMLP(in_features=28 * 28, out_features=5, k_groups=240)

        self.dropout1 = nn.Dropout(p=0.5) 
        self.maxout2 = MaxOutMLP(in_features=5, out_features=5, k_groups=240)
        
        self.dropout2 = nn.Dropout(p=0.5) 
        self.fc      = nn.Linear(in_features=5, out_features=10, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout_input(x) 
        x = self.maxout1(x)

        x = self.dropout1(x)
        x = self.maxout2(x)

        x = self.dropout2(x)
        x = self.fc(x)
        return x 