import functools

import torch
from torch.utils.data import DataLoader
import optuna

from experiments.mnist import MaxoutMLP
from experiments.helpers import MaxNorm
from experiments.helpers import load_mnist


def train_epoch(model, optimizer, criterion, lamb, train_loader, device):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)

        # Manual L2 regularization
        l2_reg = torch.tensor(0., device=device)
        for param in model.parameters():
            l2_reg += torch.sum(param ** 2)
        loss += lamb * l2_reg

        loss.backward()
        optimizer.step()


def evaluate(model, criterion, eval_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in eval_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            pred_class = pred.argmax(dim=1)
            correct += (pred_class == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / total


def objective(trial, model_class, train_loader, val_loader, device):
    lr = trial.suggest_float('lr', 1e-5, 1e-1)
    beta1 = trial.suggest_float('beta1', 0.85, 0.95)
    beta2 = trial.suggest_float('beta2', 0.95, 0.999)
    lamb = trial.suggest_float('lamb', 1e-6, 1e-2)
    c = trial.suggest_float('c', 1.4, 2.4)
    gamma = trial.suggest_float('gamma', 0.85, 0.999) 
    batch_size = trial.suggest_categorical('batch_size', [64, 100, 128, 256])

    train_loader = DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_loader.dataset, batch_size=batch_size)

    model = model_class().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
    criterion = torch.nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    maxnorm = MaxNorm(c) 

    epochs = 7
    for _ in range(epochs):
        train_epoch(model, optimizer, criterion, lamb, train_loader, device)
        model.apply(maxnorm)
        lr_scheduler.step()

    val_acc = evaluate(model, criterion, val_loader, device)
    return val_acc


def search_hyper_mlp() -> None:
    """
    Best hyperparameters:  {'lr': 0.05631780855034772, 'beta1': 0.9237233261662299, 'beta2': 0.9800475633526649, 'lamb': 0.0034916502911934598, 'c': 3.951541976591348}
    """
    train_loader, val_loader, _ = load_mnist('./mnist/', batch_size=128, validation_set=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.mps.is_available():
        device = torch.device('mps')
    else: device = torch.device('cpu') # have fun with that :P

    objective_with_args = functools.partial(
        objective,
        model_class=MaxoutMLP,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    study = optuna.create_study(direction='maximize')
    study.optimize(objective_with_args, n_trials=20)


if __name__ == '__main__':
    search_hyper_mlp()
