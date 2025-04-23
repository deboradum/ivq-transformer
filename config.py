import yaml
import dataclasses

from dataclasses import dataclass


@dataclass
class VQVAEConfig:
    encoder_h_dim: int
    res_h_dim: int
    num_res_layers: int
    k: int
    d: int
    beta: float
    pretrain_path: str


@dataclass
class TransformerConfig:
    num_heads: int
    num_layers: int
    num_classes: int
    use_rms_norm: bool


@dataclass
class GeoTransformerConfig:
    pretrain_path: str


@dataclass
class TrainConfig:
    wandb_optimize: bool
    wandb_log: bool
    fp16: bool
    num_epochs: int
    patience: int
    dataset_name: str
    batch_size: int
    optimizer: str
    learning_rate: float
    weight_decay: float
    log_interval: int
    save_interval: int


@dataclass
class Config:
    vqvae: VQVAEConfig
    transformer: TransformerConfig
    geotransformer: GeoTransformerConfig
    train: TrainConfig

    def to_dict(self):
        return dataclasses.asdict(self)


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(
        vqvae=VQVAEConfig(**data["vqvae"]),  # Now accepts pretrain_path
        transformer=TransformerConfig(**data["transformer"]),
        geotransformer=GeoTransformerConfig(**data["geotransformer"]),
        train=TrainConfig(**data["train"]),
    )
