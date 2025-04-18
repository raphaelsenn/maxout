import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10

from experiment_models.mnist import NetMNIST

from experiment_models.helpers import MaxNorm   # max-norm constraint


# -----------------------------------------------------------------------------
# settings
# -----------------------------------------------------------------------------
DATASET = 'mnist'                 # mnist or cifar-10

lr = 0.001                         # learning rate
momentum = 0.95                     # momentum
epochs = 4                          # number of iterations
batch_size = 64                     # batch size
seed = 42                           # random seed for reproducability

verbose = True                      # printing error/acc while training
num_threads = 10                    # number of threads 
device = torch.device('cpu')        # computing device i.e. cpu or cuda

torch.manual_seed(seed)
torch.set_num_threads(num_threads)


def evaluate(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device
        ) -> tuple[float, float]:
    model.eval() 
    correct = 0 
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device) 
            preds = model(inputs)
            correct += torch.sum(targets == torch.argmax(preds, dim=1))
    model.train()
    acc = correct / len(dataloader.dataset)
    error = 1 - acc
    return acc, error


def train(
        model: nn.Module,
        dataloader: DataLoader,
        epochs: int,
        lr: float,
        momentum: float,
        device: torch.device,
        verbose: bool
        ) -> None:
    model.train()
    
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    maxnorm = MaxNorm(max_value=4)

    # for epoch in range(epochs):
    for epoch in range(epochs):

        total_loss, error = 0.0, 0.0

        start_time = time.monotonic()
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # reset gradients 
            optimizer.zero_grad() 
            
            # make predictions
            pred = model.forward(X_batch)
            
            # calculate cross-entropy
            loss = criterion(pred, y_batch)
            
            # error measures (+loss) 
            total_loss += loss.item()
            error += torch.sum(y_batch != torch.argmax(pred, dim=1))

            # backpropagation
            loss.backward()

            # update parameters
            optimizer.step()

            # apply max-norm constraint
            model.apply(maxnorm)

        end_time = time.monotonic()
        
        if verbose:
            total_loss = total_loss / len(dataloader.dataset) 
            total_error = error / len(dataloader.dataset)
            total_acc = 1 - total_error 
            epoch_time = end_time - start_time
            print(f'epoch: {epoch}\ttime: {epoch_time:.02f}s\tloss: {total_loss:.04f}\terror: {total_error:.04f}\tacc: {total_acc:.04f}')


if __name__ == '__main__':

    if torch.backends.mps.is_available():
        device = torch.device('mps')
 
    if DATASET == 'mnist':  # training on minst
        model = NetMNIST()

        # loading and transforming the mnist dataset
        transform = transforms.Compose(
            [transforms.ToTensor(),                     # greyscale [0, 255] -> [0, 1]
            transforms.Lambda(lambda x: x.view(-1))])   # shape [1, 28, 28] -> [1, 784]

        mnist_train = MNIST(
            root='mnist/',
            train=True,
            download=True,
            transform=transform)
        
        mnist_test = MNIST(
            root='mnist/',
            train=False,
            download=True,
            transform=transform)
        
        dataloader_train = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
        dataloader_test = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

    else: # training on cifar10
        
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) 

        cifar10_train = CIFAR10(
            root='cifar10/',
            train=True,
            download=True,
            transform=transform)
        
        cifar10_test = CIFAR10(
            root='cifar10/',
            train=False,
            download=True,
            transform=transform)
        
        dataloader_train = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
        dataloader_test = DataLoader(cifar10_test, batch_size=batch_size, shuffle=True)

    model.to(device)


    # finally start training on mnist
    train(
        model=model,
        dataloader=dataloader_train,
        epochs=epochs,
        lr=lr,
        momentum=momentum,
        device=device,
        verbose=True)

    # evaluating 
    acc_train, error_train = evaluate(model, dataloader_train, device=device)
    acc_test, error_test = evaluate(model, dataloader_test, device=device)

    # printing results
    print(f'(train report)\terror: {error_train:04f}\tacc: {acc_train:04f}')
    print(f'(test report)\t error: {error_test:04f}\tacc: {acc_test:04f}')