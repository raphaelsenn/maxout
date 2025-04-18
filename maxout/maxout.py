import torch
import torch.nn as nn


class MaxOutMLP(nn.Module):
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


# TODO: implemnt maxout for convolutional neural networks
class MaxOutConv(nn.Module):
    def __init__(self) -> None:
        pass