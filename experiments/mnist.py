import torch
import torch.nn as nn

from maxout.maxout import Maxout


class MaxoutMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Dropout(p=0.2),
            Maxout(in_features=28 * 28, out_features=240, k_groups=5),
            nn.Dropout(p=0.5),
            Maxout(in_features=240, out_features=240, k_groups=5),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=240, out_features=10)
        )
        # self.initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def initialize_weights(self) -> None:
        for m in self.parameters():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.005, 0.005)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)