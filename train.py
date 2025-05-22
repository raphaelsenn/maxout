import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.neural_networks import MaxoutMLP
from experiments.neural_networks import MaxoutConvNet

from experiments.helpers import MaxNorm             # max-norm constraint
from experiments.helpers import MomentumAdjuster    # adjust momentum while training
from experiments.helpers import load_cifar10, load_mnist

"""
Potential hyperparameters of the original paper:

Max norm constant: 1.9365
batch size: 100
learning rate: 0.1
start momemtun: 0.5
init momentum: 0.7
"""


# -----------------------------------------------------------------------------
# settings
# -----------------------------------------------------------------------------
DATASET = 'mnist'                 # mnist or cifar-10
ROOT_DATA = DATASET + '/'

lr = 0.1                            # init learning rate
momentum = 0.5                      # init momentum
epochs = 4                          # number of iterations
c = 1.9365                          # max norm constraint
batch_size = 64                     # batch size
seed = 42                           # random seed for reproducability

verbose = True                      # printing error/acc while training
num_threads = 10                    # number of threads
device = torch.device('mps')        # computing device i.e. cpu, cuda or mps

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
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.999996) 
    momentum_adjuster = MomentumAdjuster(optimizer=optimizer, init_momentum=momentum, final_momentum=0.7, saturate=5)
    maxnorm = MaxNorm(max_value=c)

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
        lr_scheduler.step()
        momentum_adjuster.step()

        if verbose:
            total_loss = total_loss / len(dataloader.dataset) 
            total_error = error / len(dataloader.dataset)
            total_acc = 1 - total_error 
            epoch_time = end_time - start_time
            print(f'epoch: {epoch}\t'
                  f'time: {epoch_time:.02f}s\t'
                  f'loss: {total_loss:.04f}\t'
                  f'error: {total_error:.04f}\t'
                  f'acc: {total_acc:.04f}'
            )


if __name__ == '__main__':
    if DATASET == 'mnist':  # training on minst
        model = MaxoutMLP()
        dataloader_train, dataloader_test = load_mnist(ROOT_DATA)

    else: # training on cifar10
        model = MaxoutConvNet() 
        dataloader_train, dataloader_test = load_cifar10(ROOT_DATA)
    
    model.to(device)
    print(f'Start training on {DATASET} for {epochs} epochs\n'
          f'Using device: {device}\n'
          f'Initial lr: {lr}\n'
          f'Initial momentum: {momentum}\n'
          f'Batch size: {batch_size}'
    )
    
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