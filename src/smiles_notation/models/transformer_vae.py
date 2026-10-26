"""Trainable Transformer–VAE architecture for molecular translation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    """Default architecture dimensions for training a new Transformer–VAE."""

    vocab_size: int = 129
    width: int = 512
    heads: int = 8
    encoder_layers: int = 8
    decoder_layers: int = 8
    latent_size: int = 512
    max_length: int = 250
    dropout: float = 0.0

    def validate(self) -> None:
        if (
            min(
                self.vocab_size,
                self.width,
                self.heads,
                self.encoder_layers,
                self.decoder_layers,
                self.latent_size,
            )
            < 1
        ):
            raise ValueError("Model dimensions must be positive")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if self.max_length < 3:
            raise ValueError("max_length must be at least 3")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class SinusoidalEmbedding(nn.Module):
    """Token embedding plus deterministic sinusoidal positions."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.width, padding_idx=0)
        positions = torch.arange(config.max_length).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, config.width, 2) * -(math.log(10000.0) / config.width))
        encoding = torch.zeros(config.max_length, config.width)
        encoding[:, 0::2] = torch.sin(positions * divisor)
        encoding[:, 1::2] = torch.cos(positions * divisor)
        self.register_buffer("positions", encoding)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.dropout(
            self.embedding(token_ids) + self.positions[: token_ids.shape[1]].unsqueeze(0)
        )


class NoAffinePooler(nn.Module):
    """Concatenate normalized mean, start-token, and maximum features."""

    def __init__(self, width: int):
        super().__init__()
        self.mean_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.start_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.max_norm = nn.LayerNorm(width, elementwise_affine=False)

    def forward(self, encoded: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        valid = ~padding_mask
        counts = valid.sum(dim=1)
        if (counts == 0).any():
            raise ValueError("Every encoded sequence must contain a non-padding token")
        mean = (encoded * valid.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(-1)
        maximum = encoded.masked_fill(padding_mask.unsqueeze(-1), -torch.inf).max(dim=1).values
        return torch.cat(
            [self.mean_norm(mean), self.start_norm(encoded[:, 0]), self.max_norm(maximum)],
            dim=-1,
        )


@dataclass
class VAEOutput:
    logits: torch.Tensor
    targets: torch.Tensor
    mu: torch.Tensor
    log_variance: torch.Tensor

    def loss(self, beta: float) -> torch.Tensor:
        reconstruction = F.cross_entropy(
            self.logits.reshape(-1, self.logits.shape[-1]),
            self.targets.reshape(-1),
            ignore_index=0,
        )
        kl = (
            -0.5
            * (1 + self.log_variance - self.mu.square() - self.log_variance.exp()).sum(dim=1).mean()
        )
        return reconstruction + beta * kl


class TransformerVAE(nn.Module):
    """Transformer encoder--decoder VAE for randomized-to-canonical translation."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.enc_embedding = SinusoidalEmbedding(config)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.width,
            nhead=config.heads,
            dim_feedforward=4 * config.width,
            dropout=config.dropout,
            activation="gelu",
            layer_norm_eps=1e-9,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, config.encoder_layers, enable_nested_tensor=False
        )
        self.pooler = NoAffinePooler(config.width)
        self.latent2mu = nn.Linear(3 * config.width, config.latent_size)
        self.latent2log_variance = nn.Linear(3 * config.width, config.latent_size)
        self.latent2decoder = nn.Linear(config.latent_size, config.width)
        self.dec_embedding = SinusoidalEmbedding(config)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=config.width,
            nhead=config.heads,
            dim_feedforward=4 * config.width,
            dropout=config.dropout,
            activation="gelu",
            layer_norm_eps=1e-9,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, config.decoder_layers)
        self.dec2proba = nn.Sequential(
            nn.LayerNorm(config.width, elementwise_affine=False),
            nn.Linear(config.width, config.vocab_size),
        )

    def _validate_tokens(self, values: torch.Tensor, label: str) -> None:
        if values.ndim != 2 or not 3 <= values.shape[1] <= self.config.max_length:
            raise ValueError(f"{label} must have shape [batch, 3..max_length]")
        if values.dtype != torch.long:
            raise TypeError(f"{label} must use torch.long token IDs")
        if (values < 0).any() or (values >= self.config.vocab_size).any():
            raise ValueError(f"{label} contains an out-of-vocabulary token ID")
        if (values[:, 0] != 1).any() or (values == 2).sum(dim=1).ne(1).any():
            raise ValueError(f"{label} must start with one start token and contain one end token")

    def encode_parameters(self, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_tokens(source, "source")
        padding = source.eq(0)
        encoded = self.encoder(self.enc_embedding(source), src_key_padding_mask=padding)
        pooled = self.pooler(encoded, padding)
        return self.latent2mu(pooled), self.latent2log_variance(pooled)

    def encode_mean(self, source: torch.Tensor) -> torch.Tensor:
        return self.encode_parameters(source)[0]

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> VAEOutput:
        self._validate_tokens(target, "target")
        mu, log_variance = self.encode_parameters(source)
        latent = mu + torch.randn_like(mu) * torch.exp(0.5 * log_variance)
        decoder_input = target[:, :-1]
        expected = target[:, 1:]
        length = decoder_input.shape[1]
        causal_mask = torch.ones(
            length, length, dtype=torch.bool, device=decoder_input.device
        ).triu(1)
        decoded = self.decoder(
            self.dec_embedding(decoder_input),
            self.latent2decoder(latent).unsqueeze(1),
            tgt_mask=causal_mask,
            tgt_key_padding_mask=decoder_input.eq(0),
        )
        return VAEOutput(self.dec2proba(decoded), expected, mu, log_variance)
