# maxout
PyTorch implementation of the "Maxout unit/activation" described in the paper [Maxout Networks](https://arxiv.org/abs/1302.4389).

## The Maxout Unit
Maxout units devide the input $x \in \mathbb{R}^d, d\in \mathbb{N}$ into groups of $k \in \mathbb{N}$ values. Each maxout unit then outputs the maximum element of one of these groups:

![image](res/maxout_unit.png)

Taken from *Ian J. Goodfellow at al., 2013*,
"Max Networks",
*Proceedings of Machine Learning Research*, 2013


## Usage

Multilayer perceptrons
```python
import torch
from maxout.maxout import Maxout


input = torch.rand(64, 5)   # N=64 samples, 5 inputs

# creating a maxout layer with 5 input and 3 output neurons
maxout = Maxout(in_features=5, out_features=3, k_groups = 7)

output = maxout(input)       # shape [64, 3]
```

Convolutional neural networks

```python
import torch
from maxout.maxout import Conv2dMaxout


input = torch.rand(64, 3, 32, 32)   # N=64 samples, 3 channels, 32 x 32 input

# creating a maxout convolution
convmax = Conv2dMaxput(
  in_channels=3,
  out_channels=96,
  k_groups=2,
  kernel_size=8,
  padding=4,
  pool_size=4,
  pool_stride=2)

output = convmax(input)            # shape [64, 96, 15, 15]
```
## Visualization of Maxout Networks

![image](/res/2maxout_units.png)
Taken from *Gabriel Castaneda at al., 2019*,
"Evaluation of maxout activations in deep learning across several big data domains",
*Journal of Big Data*, 2019

![image](/res/cnn_maxout.png)
Taken from *Gabriel Castaneda at al., 2019*,
"Evaluation of maxout activations in deep learning across several big data domains",
*Journal of Big Data*, 2019

## Experiments

### MNIST (permutation invariant setting)
The model they used consisted of two densly connected maxout layers, followed by a softmax layer.
Sadly, they did not mention how many max-out units, and which k (to group and divide the input) was used. But the results from this architecture was more detailed described in the paper [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) (Table 2).


```bibtex
@InProceedings{pmlr-v28-goodfellow13,
  title = 	 {Maxout Networks},
  author = 	 {Goodfellow, Ian and Warde-Farley, David and Mirza, Mehdi and Courville, Aaron and Bengio, Yoshua},
  booktitle = 	 {Proceedings of the 30th International Conference on Machine Learning},
  pages = 	 {1319--1327},
  year = 	 {2013},
  editor = 	 {Dasgupta, Sanjoy and McAllester, David},
  volume = 	 {28},
  number =       {3},
  series = 	 {Proceedings of Machine Learning Research},
  address = 	 {Atlanta, Georgia, USA},
  month = 	 {17--19 Jun},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v28/goodfellow13.pdf},
  url = 	 {https://proceedings.mlr.press/v28/goodfellow13.html},
}
```