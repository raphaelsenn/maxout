import torch
import torch.nn as nn


class Maxout(nn.Module):
    """
    Applies the maxout unit function (for multiplayer perceptrons),
    instead of applying an element-wise function,
    maxout units devide x into groups of k values.
    Each maxout unit then outputs the maximum element of one of these groups.

    Args:
        in_featuares: size of each input sample
        out_features: size of each output sample
        k_groups: number of groups to divide input
    """ 
    def __init__(self,
                 in_features: int,
                 out_features: int, 
                 k_groups: int 
    ) -> None:
        super().__init__() 

        self.linear = nn.Linear(in_features, k_groups * out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.k_groups = k_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = self.linear(x)
        outs = outs.view(x.shape[0], self.k_groups, self.out_features)
        outs, _ = torch.max(outs, dim=1)
        return outs