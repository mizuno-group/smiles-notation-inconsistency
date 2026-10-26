"""Contracts for the adopted Transformer–VAE pretrained encoder."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from smiles_notation.models.pretrained_encoder import (  # noqa: E402
    PositionalEmbedding,
    PretrainedTransformerVAEEncoder,
)


def _write_checkpoint(folder: Path, model: PretrainedTransformerVAEEncoder) -> None:
    folder.mkdir()
    torch.save(
        {
            "pe": model.enc_embedding.pe[:200].clone(),
            "embedding.weight": model.enc_embedding.embedding.weight.detach().clone(),
        },
        folder / "enc_embedding.pth",
    )
    torch.save(model.encoder.state_dict(), folder / "encoder.pth")
    torch.save(
        {
            "modules_.0.weight": model.latent2mu.weight.detach().clone(),
            "modules_.0.bias": model.latent2mu.bias.detach().clone(),
        },
        folder / "latent2mu.pth",
    )


def test_checkpoint_pe_is_preserved_and_deterministically_extended(tmp_path):
    source = PretrainedTransformerVAEEncoder()
    expected_pe = source.enc_embedding.pe.clone()
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint, source)
    saved_state = torch.load(
        checkpoint / "enc_embedding.pth", map_location="cpu", weights_only=True
    )
    saved_state["pe"][17, 0, 23] += 1e-5
    torch.save(saved_state, checkpoint / "enc_embedding.pth")

    loaded = PretrainedTransformerVAEEncoder().load_pretrained(checkpoint)

    saved = torch.load(checkpoint / "enc_embedding.pth", map_location="cpu", weights_only=True)[
        "pe"
    ]
    assert torch.equal(loaded.enc_embedding.pe[:200], saved)
    assert torch.equal(loaded.enc_embedding.pe[200:], expected_pe[200:])


def test_loaded_probe_is_numerically_identical_to_previous_adapter(tmp_path):
    torch.manual_seed(7)
    source = PretrainedTransformerVAEEncoder()
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint, source)

    previous = PretrainedTransformerVAEEncoder()
    embedding = torch.load(checkpoint / "enc_embedding.pth", map_location="cpu", weights_only=True)
    previous.enc_embedding.embedding.load_state_dict(
        {"weight": embedding["embedding.weight"]}, strict=True
    )
    previous.encoder.load_state_dict(
        torch.load(checkpoint / "encoder.pth", map_location="cpu", weights_only=True),
        strict=True,
    )
    latent = torch.load(checkpoint / "latent2mu.pth", map_location="cpu", weights_only=True)
    previous.latent2mu.load_state_dict(
        {"weight": latent["modules_.0.weight"], "bias": latent["modules_.0.bias"]},
        strict=True,
    )
    loaded = PretrainedTransformerVAEEncoder().load_pretrained(checkpoint)
    ids = torch.tensor([[1, 118, 118, 120, 2]], dtype=torch.long)

    with torch.inference_mode():
        expected = previous.eval()(ids)
        actual = loaded.eval()(ids)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ({"embedding.weight": torch.zeros(129, 512)}, "keys"),
        (
            {
                "embedding.weight": torch.zeros(129, 512),
                "pe": torch.zeros(200, 1, 512),
                "unexpected": torch.zeros(1),
            },
            "keys",
        ),
        (
            {
                "embedding.weight": torch.zeros(128, 512),
                "pe": torch.zeros(200, 1, 512),
            },
            "embedding.weight shape",
        ),
        (
            {
                "embedding.weight": torch.zeros(129, 512),
                "pe": torch.zeros(199, 1, 512),
            },
            "pe shape",
        ),
        (
            {
                "embedding.weight": torch.zeros(129, 512, dtype=torch.float64),
                "pe": torch.zeros(200, 1, 512),
            },
            "embedding.weight dtype",
        ),
        (
            {
                "embedding.weight": torch.zeros(129, 512),
                "pe": torch.zeros(200, 1, 512, dtype=torch.float64),
            },
            "pe dtype",
        ),
    ],
)
def test_embedding_checkpoint_rejects_incomplete_or_incompatible_state(tmp_path, state, message):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    torch.save(state, checkpoint / "enc_embedding.pth")
    with pytest.raises((KeyError, ValueError, TypeError), match=message):
        PretrainedTransformerVAEEncoder().load_pretrained(checkpoint)


def test_embedding_checkpoint_rejects_nonfinite_pe(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    embedding = PositionalEmbedding()
    pe = embedding.pe[:200].clone()
    pe[17, 0, 23] = torch.nan
    torch.save(
        {"embedding.weight": embedding.embedding.weight.detach().clone(), "pe": pe},
        checkpoint / "enc_embedding.pth",
    )
    with pytest.raises(ValueError, match=r"enc_embedding.pe.*finite"):
        PretrainedTransformerVAEEncoder().load_pretrained(checkpoint)


def test_positional_embedding_accepts_250_tokens_and_rejects_251():
    embedding = PositionalEmbedding()
    for length in (200, 201, 250):
        result = embedding(torch.zeros((1, length), dtype=torch.long))
        assert result.shape == (length, 1, 512)
    with pytest.raises(ValueError, match="250.*truncation is prohibited"):
        embedding(torch.zeros((1, 251), dtype=torch.long))
