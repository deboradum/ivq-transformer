import time
import torch
import wandb

import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Optimizer
from torch.utils.data import DataLoader

from config import Config, load_config
from utils import get_avg_metrics, get_optimizer, get_loaders, get_net


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def loss_fn(model, X, y, metrics):
    acc = (torch.argmax(model(X), dim=1) == y).float().mean()
    loss = (F.cross_entropy(model(X), y)).mean()

    metrics["acc"].append(acc.item())
    metrics["loss"].append(loss.item())

    return loss


def validate(net, loader, loss_fn):
    val_metrics = {"acc": [], "loss": []}
    net.train(False)
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        # X = X.permute(0, 2, 3, 1)

        _ = loss_fn(net, X, y, val_metrics)
    avg_acc, avg_loss = get_avg_metrics(val_metrics, len(loader))

    return avg_acc, avg_loss


def train(
    net: nn.Module,
    optimizer: Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn: callable,
    config: Config,
):
    s = time.perf_counter()
    for epoch in range(config.train.num_epochs):
        train_metrics = {"acc": [], "loss": []}
        net.train(True)
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            loss = loss_fn(net, X, y, train_metrics)
            loss.backward()
            optimizer.step()

            if i > 0 and i % config.train.log_interval == 0:
                taken = time.perf_counter() - s
                itps = taken / config.train.log_interval

                # Get train metrics
                avg_acc, avg_loss = get_avg_metrics(train_metrics, config.train.log_interval)
                # Get val metrics
                val_acc, val_loss = validate(net, val_loader, loss_fn)

                print(
                    f"Epoch {epoch}, step {i}/{len(train_loader)} -",
                    f"train loss: {avg_loss:.5f}, train acc: {avg_acc:.5f},"
                    f"val loss: {val_loss:.5f}, val acc: {val_acc:.5f},"
                    f"took {taken:.5f}s ({itps} it/s)"
                )
                s = time.perf_counter()

        if epoch % config.train.save_interval == 0:
            model_path = f"checkpoint_epoch_{epoch}.npz"
            print(f"Saving model in {model_path}")
            torch.save(net.state_dict(), model_path)

    return validate(net, test_loader, loss_fn)

if __name__ == "__main__":
    config = load_config("config.yaml")

    print("Getting model")
    net = get_net(config)
    net.to(device)
    optimizer = get_optimizer(net, config)

    print("Getting loaders")
    train_loader, val_loader, test_loader = get_loaders(config)

    print("Starting training")
    test_acc, test_loss = train(
        net=net,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        config=config,
    )

    print(
        f"test loss: {test_loss:.5f}, test acc: {test_acc:.5f}"
    )
