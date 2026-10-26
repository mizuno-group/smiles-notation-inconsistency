"""CPU contracts for the publication Transformer–VAE training path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from smiles_notation.inference import extract_latent_means  # noqa: E402
from smiles_notation.models.transformer_vae import ModelConfig, TransformerVAE  # noqa: E402
from smiles_notation.preprocessing import SmilesTokenizer  # noqa: E402
from smiles_notation.training import TrainingConfig, train_from_csv  # noqa: E402


def _write_vocab(path: Path) -> None:
    path.write_text(
        "token_id,token\n0,<padding>\n1,<start>\n2,<end>\n3,C\n4,O\n5,N\n6,Cl\n",
        encoding="utf-8",
    )


def test_tokenizer_is_greedy_strict_and_never_truncates(tmp_path: Path):
    vocab = tmp_path / "vocab.csv"
    _write_vocab(vocab)
    tokenizer = SmilesTokenizer.from_csv(vocab)
    assert tokenizer.encode("CCl", max_length=5) == [1, 3, 6, 2]
    with pytest.raises(ValueError, match="Unknown SMILES token"):
        tokenizer.encode("C?", max_length=5)
    with pytest.raises(ValueError, match="exceeds"):
        tokenizer.encode("CCCC", max_length=5)


def test_full_transformer_vae_shape_contract_on_cpu():
    config = ModelConfig(
        vocab_size=7,
        width=16,
        heads=4,
        encoder_layers=1,
        decoder_layers=1,
        latent_size=8,
        max_length=12,
    )
    model = TransformerVAE(config)
    source = torch.tensor([[1, 3, 3, 2, 0], [1, 3, 4, 2, 0]])
    target = torch.tensor([[1, 3, 3, 2, 0], [1, 3, 4, 2, 0]])
    output = model(source, target)
    assert output.logits.shape == (2, 4, 7)
    assert output.mu.shape == output.log_variance.shape == (2, 8)
    assert torch.isfinite(output.loss(beta=0.01))


def test_synthetic_cpu_training_saves_reloadable_checkpoint(tmp_path: Path):
    vocab = tmp_path / "vocab.csv"
    _write_vocab(vocab)
    data = tmp_path / "train.csv"
    data.write_text(
        "record_id,input_smiles,target_smiles\nr1,CCO,CCO\nr2,CCl,CCl\nr3,CCN,CCN\nr4,CO,CO\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"
    report = train_from_csv(
        data,
        validation_csv=data,
        vocabulary_csv=vocab,
        output_dir=output,
        model_config=ModelConfig(
            vocab_size=7,
            width=16,
            heads=4,
            encoder_layers=1,
            decoder_layers=1,
            latent_size=8,
            max_length=12,
        ),
        training_config=TrainingConfig(
            device="cpu", epochs=1, batch_size=2, learning_rate=1e-3, beta=0.01, seed=7
        ),
    )
    assert report["training_examples"] == 4
    assert report["validation_examples"] == 4
    assert report["epochs"] == 1
    checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
    restored = TransformerVAE(ModelConfig(**checkpoint["model_config"]))
    restored.load_state_dict(checkpoint["model_state"], strict=True)
    assert json.loads((output / "training_report.json").read_text()) == report

    inference_input = tmp_path / "inference.csv"
    inference_input.write_text("record_id,smiles\nq1,CCO\nq2,CCl\n", encoding="utf-8")
    latent_csv = tmp_path / "latent.csv"
    manifest_path = tmp_path / "latent_manifest.json"
    manifest = extract_latent_means(
        inference_input,
        checkpoint=output / "checkpoint.pt",
        vocabulary_csv=vocab,
        output_csv=latent_csv,
        manifest_path=manifest_path,
        device="cpu",
        batch_size=2,
    )
    assert manifest["status"] == "PASS"
    assert manifest["records"] == 2
    assert manifest["latent_size"] == 8
    rows = latent_csv.read_text(encoding="utf-8").splitlines()
    assert rows[0].split(",") == ["record_id", *(f"mu_{index:03d}" for index in range(8))]
    assert len(rows) == 3
