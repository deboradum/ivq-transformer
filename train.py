import time
import torch
import wandb

import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Optimizer
from torch.utils.data import DataLoader

from wandb_utils import sweep_config
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
    logits = model(X)
    acc = (torch.argmax(logits, dim=1) == y).float().mean()
    loss = (F.cross_entropy(logits, y)).mean()

    metrics["acc"].append(acc.item())
    metrics["loss"].append(loss.item())

    return loss


def validate(net, loader, loss_fn):
    val_metrics = {"acc": [], "loss": []}
    net.train(False)
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)

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
    val_acc, val_loss = validate(net, val_loader, loss_fn)
    if config.train.wandb_log:
        wandb.log({"eval_loss": val_loss, "eval_acc": val_acc, "global_step": 0})

    best_val_loss = val_loss
    best_model_path = "best_model.pth"
    patience_counter = 0
    torch.save(net.state_dict(), best_model_path)

    s = time.time()
    for epoch in range(config.train.num_epochs):
        train_metrics = {"acc": [], "loss": []}
        global_step = epoch * len(train_loader.dataset)  # Num training examples
        net.train(True)
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            loss = loss_fn(net, X, y, train_metrics)
            loss.backward()
            optimizer.step()
            global_step += X.size(0)

            if i > 0 and i % config.train.log_interval == 0:
                taken = time.time() - s
                itps = taken / config.train.log_interval

                avg_acc, avg_loss = get_avg_metrics(train_metrics, config.train.log_interval)
                val_acc, val_loss = validate(net, val_loader, loss_fn)
                if config.train.wandb_log:
                    wandb.log(
                        {
                            "train_loss": avg_loss,
                            "train_acc": avg_acc,
                            "eval_loss": val_loss,
                            "eval_acc": val_acc,
                            "global_step": global_step,
                        }
                    )

                print(
                    f"Epoch {epoch}, step {i}/{len(train_loader)} -",
                    f"train loss: {avg_loss:.3f}, train acc: {avg_acc:.3f},"
                    f"val loss: {val_loss:.3f}, val acc: {val_acc:.3f},"
                    f"took {taken:.2f}s ({itps:.2f} it/s)"
                )

                # Patience stuff
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(net.state_dict(), best_model_path)
                else:
                    patience_counter += 1
                if patience_counter >= config.train.patience:
                    print("Early stopping")
                    break

                s = time.time()

        if epoch % config.train.save_interval == 0 and config.train.save_interval != -1:
            model_path = f"checkpoint_epoch_{epoch}.pth"
            print(f"Saving model in {model_path}")
            torch.save(net.state_dict(), model_path)

        if patience_counter >= config.train.patience:
            break

    net.load_state_dict(torch.load(best_model_path))
    return validate(net, test_loader, loss_fn)


def train_from_config(config: Config):
    if config.train.wandb_log:
        wandb.init(project="vqi_transformer", config=config)
    net = get_net(config)
    net.to(device)
    optimizer = get_optimizer(net, config)
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
    if config.train.wandb_log:
        wandb.log(
            {
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )


def train_with_wandb(config: Config):
    wandb.init()
    wandb_config = wandb.config

    config.train.optimizer = wandb_config.optimizer
    config.train.learning_rate = wandb_config.learning_rate
    config.train.weight_decay = wandb_config.weight_decay
    config.transformer.num_heads = wandb_config.num_transformer_heads
    config.transformer.num_layers = wandb_config.num_transformer_layers
    config.transformer.use_rms_norm = wandb_config.transformer_use_rms_norm
    wandb.config.update(config.to_dict(), allow_val_change=True)

    net = get_net(config)
    net.to(device)
    optimizer = get_optimizer(net, config)
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
    wandb.log(
        {
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
    )
    del net, optimizer, train_loader, val_loader, test_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    config = load_config("config.yaml")

    if config.train.wandb_optimize:
        sweep_id = wandb.sweep(sweep_config, project="vqi_transformer")
        wandb.agent(sweep_id, lambda: train_with_wandb(config), count=5)
    else:
        train_from_config(config)
