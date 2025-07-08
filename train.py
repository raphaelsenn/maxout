import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.mnist import MaxoutMLP
from experiments.helpers import MaxNorm
from experiments.helpers import load_mnist


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
        optimizer,
        criterion,
        dataloader: DataLoader,
        epochs: int,
        lamb: float,
        c: float,
        device: torch.device,
        verbose: bool,
    ) -> None:

    model.train()
    maxnorm = MaxNorm(max_value=c)

    for epoch in range(epochs):
        total_loss, error = 0.0, 0.0
        start_time = time.monotonic()
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad() 
            pred = model.forward(X_batch)
            loss = criterion(pred, y_batch)

            # L2 weight decay 
            l2_reg = torch.tensor(0.).to(device)
            for param in model.parameters(): l2_reg += torch.sum(param**2)
            loss += lamb * l2_reg

            loss.backward()
            optimizer.step()
            model.apply(maxnorm)
            
            total_loss += loss.item() * X_batch.shape[0]
            error += torch.sum(y_batch != torch.argmax(pred, dim=1))
        end_time = time.monotonic()

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
    DATASET = 'mnist'
    ROOT_DATA = DATASET + '/'

    hyper = {'lr': 0.0009261126249687733, 'beta1': 0.9177696863739093, 'beta2': 0.9677753053312056, 'lamb': 0.002797628859988094, 'c': 1.6648941117216467}
    batch_size = 128
    epochs = 50
    
    model = MaxoutMLP()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=hyper['lr'], betas=(hyper['beta1'], hyper['beta2']))
    criterion = nn.CrossEntropyLoss() 

    seed = 42
    verbose = True                      # printing error/acc while training
    num_threads = 10                    # number of threads
    device = torch.device('mps')        # computing device i.e. cpu, cuda or mps
    torch.manual_seed(seed)
    torch.set_num_threads(num_threads) 
    
    dataloader_train, dataloader_test = load_mnist(ROOT_DATA, batch_size=batch_size)

    model.to(device)
    print(f'Start training on {DATASET} for {epochs} epochs\n'
          f'Using device: {device}\n'
          f'Learning rate: {hyper['lr']}\n'
          f'Batch size: {batch_size}'
    )
    
    # finally start training on mnist
    train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        dataloader=dataloader_train,
        epochs=epochs,
        lamb=hyper['lamb'],
        c=hyper['c'],
        device=device,
        verbose=True)

    # evaluating 
    acc_train, error_train = evaluate(model, dataloader_train, device=device)
    acc_test, error_test = evaluate(model, dataloader_test, device=device)

    # printing results
    print(f'(train report)\terror: {error_train:04f}\tacc: {acc_train:04f}')
    print(f'(test report)\t error: {error_test:04f}\tacc: {acc_test:04f}')