import os
import torch

import torch.optim as optim

from config import Config
from model import GeoTransformer

from torch.utils.data import DataLoader, Dataset


def get_avg(values):
    return sum(values) / len(values)


def get_avg_metrics(metrics, window):
    avg_acc = get_avg(metrics["acc"][-window:-1])
    avg_loss = get_avg(metrics["loss"][-window:-1])

    return avg_acc, avg_loss


# TODO: in the future perhaps add lr scheduling?
def get_optimizer(config: Config):
    if config.train.optimizer == "adam":
        optimizer = optim.Adam(lr=config.train.learning_rate)
    elif config.train.optimizer == "adamW":
        optimizer = optim.AdamW(
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
    elif config.train.optimizer == "adagrad":
        optimizer = optim.Adagrad(lr=config.train.learning_rate)
    else:
        raise NotImplementedError(f"Optimizer {config.train.optimizer} not supported")

    return optimizer


def get_loaders(config: Config):
    train_loader, val_loader, test_loader = ()

    return train_loader, val_loader, test_loader


def get_net(config: Config):
    net = GeoTransformer(
        encoder_h_dim=config.vqvae.encoder_h_dim,
        res_h_dim=config.vqvae.res_h_dim,
        num_res_layers=config.vqvae.num_res_layers,
        k=config.vqvae.k,
        d=config.vqvae.d,
        beta=config.vqvae.beta,
        num_transformer_layers=config.transformer.num_layers,
        num_heads=config.transformer.num_heads,
        num_classes=config.transformer.num_classes,
        use_rms_norm=config.transformer.use_rms_norm,
    )

    # Load vqvae2 parameters from file
    if os.path.isfile(config.vqvae.pretrain_path):
        net.load_state_dict(torch.load(config.vqvae.pretrain_path, weights_only=True))
    # Load GeoTransformer parameters from file
    if os.path.isfile(config.vqvae.pretrain_path):
        net.load_state_dict(torch.load(config.geotransformer.pretrain_path, weights_only=True))

    return net
