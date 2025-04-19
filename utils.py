import os
import torch

import torch.optim as optim
import torchvision.transforms as transforms

from config import Config
from model import GeoTransformer

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import Caltech101


def get_avg(values):
    return sum(values) / len(values)


def get_avg_metrics(metrics, window):
    avg_acc = get_avg(metrics["acc"][-window:-1])
    avg_loss = get_avg(metrics["loss"][-window:-1])

    return avg_acc, avg_loss


# TODO: in the future perhaps add lr scheduling?
def get_optimizer(net, config: Config):
    if config.train.optimizer == "adam":
        optimizer = optim.Adam(params=net.parameters(), lr=config.train.learning_rate)
    elif config.train.optimizer == "adamW":
        optimizer = optim.Adam(
            params=net.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
    elif config.train.optimizer == "adagrad":
        optimizer = optim.Adagrad(
            params=net.parameters(), lr=config.train.learning_rate
        )
    else:
        raise NotImplementedError(f"Optimizer {config.train.optimizer} not supported")

    return optimizer


def get_loaders_Caltech101(batch_size):
    transform = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.ToTensor(),
        ]
    )

    full_dataset = Caltech101(root="./data", download=True, transform=transform)

    train_size = int(0.8 * len(full_dataset))
    val_test_size = len(full_dataset) - train_size
    val_size = int(
        0.5 * val_test_size
    )  # Split the remaining data into validation and test
    test_size = val_test_size - val_size

    train_dataset, val_test_dataset = random_split(
        full_dataset, [train_size, val_test_size]
    )
    val_dataset, test_dataset = random_split(val_test_dataset, [val_size, test_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    x_train_var = 1.0472832505104722e-06
    # print("Computing train var")
    # x_train_var = 0.0
    # count = 0
    # for images, _ in train_loader:
    #     images = images.view(images.size(0), -1) / 255.0
    #     x_train_var += images.var(dim=1).sum().item()
    #     count += images.size(0)
    # x_train_var /= count
    # print("train var:", x_train_var)

    return train_loader, val_loader, test_loader, x_train_var


def get_loaders(config):
    dataset_name = config.train.dataset_name
    batch_size = config.train.batch_size

    if dataset_name == "caltech101":
        train_loader, val_loader, test_loader, x_train_var = get_loaders_Caltech101(
            batch_size
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported yet.")

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
        print(f"Loading vqvae model weights from {config.vqvae.pretrain_path}")
        net.encoder.load_state_dict(
            torch.load(config.vqvae.pretrain_path, weights_only=True)
        )
    # Load GeoTransformer parameters from file
    if os.path.isfile(config.geotransformer.pretrain_path):
        print(
            f"Loading ivq-transformer model weights from {config.geotransformer.pretrain_path}"
        )
        net.load_state_dict(
            torch.load(config.geotransformer.pretrain_path, weights_only=True)
        )

    return net
