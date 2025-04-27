import os

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10, MNIST

###############################################################################
# MaxNorm weight constraint:
###############################################################################


class MaxNorm(object):
    def __init__(self, max_value: int=2, dim: int=0) -> None:
        self.max_value = max_value
        self.dim = dim

    def __call__(self, m: nn.Module) -> None:
        if hasattr(m, 'weight'):
            norms = torch.norm(m.weight.data, dim=self.dim, keepdim=True)
            w = m.weight.data.clamp(norms, self.max_value)
            m.weight.data = w


###############################################################################
# Contrast normalization and ZCA whitening for CIFAR10:
###############################################################################


def zca_whitening(X: torch.Tensor, epsilon: float=10e-7) -> None:
    """
    Applies contrast normalization and ZCA whitening on the CIFAR-10 dataset.
    """
    # contrast normalization 
    X = X / 255.0
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std

    # [N, 32, 32, 3] -> [N, 3072]
    X = X.reshape(-1, 3 * 32 * 32)

    # calculate mean for each image
    mean = X.mean(axis=0)
    
    # mean subtraction per-pixel for all images
    X_ = X - mean

    # calculate covariance-matrix
    cov = np.cov(X_, rowvar=False)

    # apply magic (linear algebra 1+2 at university was perfect <3)
    U, S, _ = np.linalg.svd(cov)

    # final step of zca
    X_ZCA = U.dot(np.diag(1.0/np.sqrt(S + epsilon))).dot(U.T).dot(X_.T).T 
 
    return torch.from_numpy(X_ZCA.reshape(X_ZCA.shape[0], 32, 32, 3).transpose(0, 3, 1, 2))


###############################################################################
# Load datasets (MNIST and CIFAR10):
###############################################################################


def load_cifar10(root: str, batch_size: int=64) -> tuple[DataLoader, DataLoader]:
    """
    Loads the CIFAR-10 dataset (with ZCA whitening) and returns a tuple of DataLoaders 
    for the training and test sets.
    """ 
    
    cifar10_train = CIFAR10(
        root=root,
        train=True,
        download=True)
    
    cifar10_test = CIFAR10(
        root=root,
        train=False,
        download=True)

    if not (os.path.isfile('cifar10_train.pt') and os.path.isfile('cifar10_test.pt')):
        print(f'Applying ZCA whitening to CIFAR10_train') 
        X_train = zca_whitening(cifar10_train.data)
        y_train = torch.tensor(cifar10_train.targets).long()
        torch.save((X_train, y_train), 'cifar10_train.pt')

        print(f'Applying ZCA whitening to CIFAR10_test') 
        X_test = zca_whitening(cifar10_test.data)
        y_test = torch.tensor(cifar10_test.targets).long()
        torch.save((X_test, y_test), 'cifar10_test.pt')

    X_train, y_train = torch.load('cifar10_train.pt', weights_only=False)
    X_test, y_test = torch.load('cifar10_test.pt', weights_only=False)
    
    dataset_train = TensorDataset(X_train.float(), y_train)
    dataset_test = TensorDataset(X_test.float(), y_test)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    
    return dataloader_train, dataloader_test


def load_mnist(root: str, batch_size: int=64) -> tuple[DataLoader, DataLoader]:
    """
    Loads the MNIST dataset and returns a tuple of DataLoaders 
    for the training and test sets.
    """ 

    transform = transforms.Compose(
        [transforms.ToTensor(),                     # greyscale [0, 255] -> [0, 1]
        transforms.Lambda(lambda x: x.view(-1))])   # shape [1, 28, 28] -> [1, 784]

    mnist_train = MNIST(
        root=root,
        train=True,
        download=True,
        transform=transform)
        
    mnist_test = MNIST(
        root=root,
        train=False,
        download=True,
        transform=transform)
    
    dataloader_train = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)
    return dataloader_train, dataloader_test