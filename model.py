import torch

import torch.nn as nn

from vqvae.models import VQVAE2_large

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def create_additive_causal_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, widening_factor=4):
        super().__init__()

        dim = embedding_dim * widening_factor

        self.l1 = nn.Linear(embedding_dim, dim, bias=False)
        self.l2 = nn.Linear(embedding_dim, dim, bias=False)
        self.silu = nn.SiLU()
        self.final = nn.Linear(dim, embedding_dim, bias=False)

    def forward(self, x):
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
            batch_first=True,
        )
        self.norm2 = (
            nn.RMSNorm(embedding_dim) if rms_norm else nn.LayerNorm(embedding_dim)
        )

        self.feedForward = FeedForward(embedding_dim, widening_factor=4)

    def forward(self, x):
        x_ln = self.norm1(x)
        x_f, _ = self.attention(
            x_ln,
            x_ln,
            x_ln,
            attn_mask=create_additive_causal_mask(x.shape[1]),
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
        super(GeoTransformer, self).__init__()
        self.encoder = VQVAE2_large(
            encoder_h_dim=encoder_h_dim,
            res_h_dim=res_h_dim,
            num_res_layers=num_res_layers,
            k=k,
            d=d,
            beta=beta,
        )
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.layers = nn.Sequential(
            *[
                TransformerBlock(
                    embedding_dim=d, num_heads=num_heads, rms_norm=use_rms_norm
                )
                for _ in range(num_transformer_layers)
            ]
        )
        self.final_layer = nn.Linear(d, num_classes)

    def forward(self, x):
        with torch.no_grad():
            h, *losses, _, embeddings = self.encoder.encode(x)
        h = self.layers(embeddings)
        logits = self.final_layer(h)

        vqvae_loss = sum(losses)

        return logits[:, -1, :], vqvae_loss
