"""Adopted Transformer–VAE encoder definitions for strict checkpoint inference."""

import csv
import math
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

MAX_TOKEN_LENGTH = 250
SAVED_POSITION_COUNT = 200
VOCABULARY_SIZE = 129
MODEL_WIDTH = 512


class PretrainedTokenizer:
    def __init__(self, path):
        rows = []
        with open(path, encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append((int(row["token_id"]), row["token"]))
        self.tok2voc = [token for _, token in sorted(rows)]
        self.voc2tok = {token: index for index, token in enumerate(self.tok2voc)}
        self.lengths = sorted(
            {len(token) for token in self.tok2voc if not token.startswith("<")}, reverse=True
        )

    def tokenize(self, smiles):
        output = [1]
        left = smiles
        while left:
            for length in self.lengths:
                prefix = left[:length]
                if prefix in self.voc2tok:
                    output.append(self.voc2tok[prefix])
                    left = left[length:]
                    break
            else:
                raise KeyError(f"unknown token at {left!r}")
        output.append(2)
        return output


def newgelu(value):
    return 0.5 * value * (
        1 + torch.tanh(math.sqrt(2 / math.pi) * (value + 0.044715 * value.pow(3)))
    )


def _sinusoidal_positions(max_len=MAX_TOKEN_LENGTH):
    positions = torch.arange(max_len).unsqueeze(1)
    divisor = torch.exp(torch.arange(0, MODEL_WIDTH, 2) * -(math.log(10000) / MODEL_WIDTH))
    encoding = torch.zeros(max_len, MODEL_WIDTH)
    encoding[:, 0::2] = torch.sin(positions * divisor)
    encoding[:, 1::2] = torch.cos(positions * divisor)
    return encoding.unsqueeze(1)


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len=MAX_TOKEN_LENGTH):
        super().__init__()
        if max_len != MAX_TOKEN_LENGTH:
            raise ValueError(f"Legacy positional encoding requires max_len={MAX_TOKEN_LENGTH}")
        self.embedding = nn.Embedding(VOCABULARY_SIZE, MODEL_WIDTH, padding_idx=0)
        self.register_buffer("pe", _sinusoidal_positions(max_len))

    def forward(self, tokens):
        if tokens.ndim != 2:
            raise ValueError("Expected a two-dimensional token batch")
        if tokens.shape[1] > MAX_TOKEN_LENGTH:
            raise ValueError(
                f"Tokenized input exceeds {MAX_TOKEN_LENGTH} positions; truncation is prohibited"
            )
        embedded = self.embedding(tokens.T.contiguous())
        return embedded + self.pe[: embedded.size(0)]


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            MODEL_WIDTH, 8, dropout=0, batch_first=False
        )
        self.linear1 = nn.Linear(MODEL_WIDTH, 2048)
        self.dropout = nn.Dropout(0)
        self.linear2 = nn.Linear(2048, MODEL_WIDTH)
        self.norm1 = nn.LayerNorm(MODEL_WIDTH, eps=1e-9)
        self.norm2 = nn.LayerNorm(MODEL_WIDTH, eps=1e-9)
        self.dropout1 = nn.Dropout(0)
        self.dropout2 = nn.Dropout(0)

    def forward(self, inputs, mask):
        query = self.norm1(inputs)
        attended, _ = self.self_attn(
            query, query, query, key_padding_mask=mask, need_weights=False
        )
        inputs = inputs + self.dropout1(attended)
        return inputs + self.dropout2(
            self.linear2(self.dropout(newgelu(self.linear1(self.norm2(inputs)))))
        )


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([Layer() for _ in range(8)])

    def forward(self, inputs, mask):
        for layer in self.encoder.layers:
            inputs = layer(inputs, mask)
        return inputs


class Pooler(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_norm = nn.LayerNorm(MODEL_WIDTH, elementwise_affine=False)
        self.start_norm = nn.LayerNorm(MODEL_WIDTH, elementwise_affine=False)
        self.max_norm = nn.LayerNorm(MODEL_WIDTH, elementwise_affine=False)

    def forward(self, inputs, transposed_mask):
        valid = ~transposed_mask
        masked = inputs + torch.log(valid.to(inputs.dtype)).unsqueeze(-1)
        mean = (inputs * valid.unsqueeze(-1)).sum(0) / valid.sum(0).unsqueeze(-1)
        return torch.cat(
            [self.mean_norm(mean), self.start_norm(inputs[0]), self.max_norm(masked.max(0)[0])],
            -1,
        )


def _require_exact_keys(state, expected, label):
    if not isinstance(state, Mapping):
        raise TypeError(f"{label} must contain a state_dict mapping")
    actual = set(state)
    expected = set(expected)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise KeyError(f"{label} keys mismatch: missing={missing}, unexpected={unexpected}")


def _require_tensor(tensor, shape, dtype, label):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{label} must be a tensor")
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{label} shape must be {tuple(shape)}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise TypeError(f"{label} dtype must be {dtype}, got {tensor.dtype}")


def _validate_module_state(state, module, label):
    expected = module.state_dict()
    _require_exact_keys(state, expected, label)
    for key, expected_tensor in expected.items():
        _require_tensor(state[key], expected_tensor.shape, expected_tensor.dtype, f"{label}.{key}")


class PretrainedTransformerVAEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_embedding = PositionalEmbedding()
        self.encoder = Encoder()
        self.pooler = Pooler()
        self.latent2mu = nn.Linear(3 * MODEL_WIDTH, MODEL_WIDTH)

    def load_pretrained(self, checkpoint):
        checkpoint = Path(checkpoint)
        embedding_state = torch.load(
            checkpoint / "enc_embedding.pth", map_location="cpu", weights_only=True
        )
        _require_exact_keys(embedding_state, {"embedding.weight", "pe"}, "enc_embedding")
        _require_tensor(
            embedding_state["embedding.weight"],
            (VOCABULARY_SIZE, MODEL_WIDTH),
            torch.float32,
            "enc_embedding.embedding.weight",
        )
        _require_tensor(
            embedding_state["pe"],
            (SAVED_POSITION_COUNT, 1, MODEL_WIDTH),
            torch.float32,
            "enc_embedding.pe",
        )

        generated = self.enc_embedding.pe.detach().clone()
        saved = embedding_state["pe"]
        if not torch.isfinite(saved).all():
            raise ValueError("enc_embedding.pe must contain only finite values")
        generated[:SAVED_POSITION_COUNT].copy_(saved)
        self.enc_embedding.load_state_dict(
            {"embedding.weight": embedding_state["embedding.weight"], "pe": generated},
            strict=True,
        )

        encoder_state = torch.load(
            checkpoint / "encoder.pth", map_location="cpu", weights_only=True
        )
        _validate_module_state(encoder_state, self.encoder, "encoder")
        self.encoder.load_state_dict(encoder_state, strict=True)

        latent_state = torch.load(
            checkpoint / "latent2mu.pth", map_location="cpu", weights_only=True
        )
        _require_exact_keys(
            latent_state, {"modules_.0.weight", "modules_.0.bias"}, "latent2mu"
        )
        mapped_latent = {
            "weight": latent_state["modules_.0.weight"],
            "bias": latent_state["modules_.0.bias"],
        }
        _validate_module_state(mapped_latent, self.latent2mu, "latent2mu")
        self.latent2mu.load_state_dict(mapped_latent, strict=True)
        return self

    def forward(self, tokens):
        mask = tokens.eq(0)
        encoded = self.enc_embedding(tokens)
        encoded = self.encoder(encoded, mask)
        return self.latent2mu(self.pooler(encoded, mask.T))
