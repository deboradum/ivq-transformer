import os
import torch

import torch.optim as optim
import torchvision.transforms as transforms

from config import Config
from model import GeoTransformer

from torchvision import datasets
from torch.utils.data import DataLoader, random_split


def get_avg(values):
    return sum(values) / len(values)


def get_avg_metrics(metrics, window):
    avg_acc = get_avg(metrics["acc"][-window:-1])
    avg_loss = get_avg(metrics["loss"][-window:-1])

    return avg_acc, avg_loss


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


def get_loaders_imagenet256(batch_size, data_dir="data/imagenet256"):
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    train_size = int(0.8 * len(full_dataset))
    val_test_size = len(full_dataset) - train_size
    val_size = val_test_size // 2
    test_size = val_test_size - val_size

    train_dataset, val_test_dataset = random_split(
        full_dataset, [train_size, val_test_size]
    )
    val_dataset, test_dataset = random_split(val_test_dataset, [val_size, test_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, val_loader, test_loader


def get_loaders(config):
    dataset_name = config.train.dataset_name
    batch_size = config.train.batch_size

    if dataset_name == "imagenet256":
        train_loader, val_loader, test_loader = get_loaders_imagenet256(
            batch_size
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported yet.")

    return train_loader, val_loader, test_loader


def get_net(config: Config):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

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
            torch.load(config.vqvae.pretrain_path, weights_only=True, map_location=device)
        )
    # Load GeoTransformer parameters from file
    if os.path.isfile(config.geotransformer.pretrain_path):
        print(
            f"Loading ivq-transformer model weights from {config.geotransformer.pretrain_path}"
        )
        net.load_state_dict(
            torch.load(config.geotransformer.pretrain_path, weights_only=True, map_location=device)
        )

    return net
