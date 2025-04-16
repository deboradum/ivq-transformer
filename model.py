import torch.nn as nn

from vqvae.models import VQVAE2


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, widening_factor=4):
        super().__init__()

        dim = embedding_dim * widening_factor

        self.l1 = nn.Linear(embedding_dim, dim, bias=False)
        self.l2 = nn.Linear(embedding_dim, dim, bias=False)
        self.silu = nn.SiLU()
        self.final = nn.Linear(dim, embedding_dim, bias=False)

    def __call__(self, x):
        x1 = self.l1(x)
        x2 = self.l2(x)
        x = self.silu(x1) * x2

        return self.final(x)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, rms_norm=False):
        super().__init__()
        self.norm1 = (
            nn.RMSNorm(embedding_dim) if rms_norm else nn.LayerNorm(embedding_dim)
        )
        self.attention = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
        )
        self.norm2 = (
            nn.RMSNorm(embedding_dim) if rms_norm else nn.LayerNorm(embedding_dim)
        )

        self.feedForward = FeedForward(embedding_dim, widening_factor=4)

    def __call__(self, x):
        x_ln = self.norm1(x)
        x_f = self.attention(
            x_ln,
            x_ln,
            x_ln,
            mask=self.attention.create_additive_causal_mask(x.shape[1]),
        )
        x = x + x_f

        x_ln = self.norm2(x)
        x_f = self.feedForward(x_ln)
        x = x + x_f

        return x


class GeoTransformer(nn.Module):
    def __init__(
        self,
        encoder_h_dim,
        res_h_dim,
        num_res_layers,
        k,
        d,
        beta,
        num_transformer_layers,
        num_heads,
        num_classes,
        use_rms_norm,
    ):
        self.encoder = VQVAE2(
            encoder_h_dim=encoder_h_dim,
            res_h_dim=res_h_dim,
            num_res_layers=num_res_layers,
            k=k,
            d=d,
            beta=beta,
        )
        self.layers = nn.Sequential(
            TransformerBlock(
                embedding_dim=d, num_heads=num_heads, rms_norm=use_rms_norm
            )
            for _ in range(num_transformer_layers)
        )
        self.final_layer = nn.Linear(d, num_classes)

    def __cal__(self, x):
        h = self.encoder.encode(x)
        h = self.layers(h)
        logits = self.final_layer(h)

        return logits[:, -1, :]  # Only check logits for last token
