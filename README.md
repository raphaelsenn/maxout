# maxout
PyTorch implementation of the "Maxout unit/activation" described in the paper "Maxout Networks".

## The Maxout Unit
Maxout units devide the input $x \in \mathbb{R}^d$ into groups of $k \in \mathbb{N}$ values. Each maxout unit then outputs the maximum element of one of these groups:

![image](res/maxout_unit.png)

## Experiments (replicated)

Original
![image](/res/figure1.png)

Replication
![image](/res/figure1_replica.png)
