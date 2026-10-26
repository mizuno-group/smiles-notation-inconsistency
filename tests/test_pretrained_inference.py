"""Contracts for the standalone adopted Transformer–VAE latent CLI."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from smiles_notation import pretrained_inference  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_inputs(root: Path) -> tuple[Path, Path, Path]:
    smiles = root / "smiles.csv"
    smiles.write_text("record_id,smiles\na,CCO\nb,C\n", encoding="utf-8")
    mapping = root / "tokens.csv"
    mapping.write_text("token_id,token\n0,<pad>\n1,<start>\n2,<end>\n3,C\n4,O\n", encoding="utf-8")
    weights = root / "weights"
    weights.mkdir()
    for name in pretrained_inference.EXPECTED_WEIGHT_SHA256:
        (weights / name).write_bytes(name.encode("ascii"))
    return smiles, mapping, weights


def test_read_smiles_requires_exact_header_unique_ids_and_nonempty_values(tmp_path: Path) -> None:
    valid = tmp_path / "valid.csv"
    valid.write_text("record_id,smiles\na,CCO\nb,C\n", encoding="utf-8")
    assert pretrained_inference.read_smiles_csv(valid) == [("a", "CCO"), ("b", "C")]

    invalid = tmp_path / "invalid.csv"
    invalid.write_text("record_id,smiles,extra\na,CCO,x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        pretrained_inference.read_smiles_csv(invalid)
    invalid.write_text("record_id,smiles\na,CCO\na,C\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate record_id"):
        pretrained_inference.read_smiles_csv(invalid)
    invalid.write_text("record_id,smiles\na,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        pretrained_inference.read_smiles_csv(invalid)


def test_validate_model_assets_requires_adopted_hashes(tmp_path: Path, monkeypatch) -> None:
    _smiles, mapping, weights = _write_inputs(tmp_path)
    monkeypatch.setattr(pretrained_inference, "EXPECTED_TOKEN_MAPPING_SHA256", _sha256(mapping))
    monkeypatch.setattr(
        pretrained_inference,
        "EXPECTED_WEIGHT_SHA256",
        {name: _sha256(weights / name) for name in pretrained_inference.EXPECTED_WEIGHT_SHA256},
    )
    observed = pretrained_inference.validate_model_assets(mapping, weights)
    assert observed["token_mapping"] == _sha256(mapping)
    (weights / "encoder.pth").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch.*encoder.pth"):
        pretrained_inference.validate_model_assets(mapping, weights)


def test_run_writes_record_aligned_512_dimensional_output_and_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    smiles, mapping, weights = _write_inputs(tmp_path)
    monkeypatch.setattr(pretrained_inference, "EXPECTED_TOKEN_MAPPING_SHA256", _sha256(mapping))
    monkeypatch.setattr(
        pretrained_inference,
        "EXPECTED_WEIGHT_SHA256",
        {name: _sha256(weights / name) for name in pretrained_inference.EXPECTED_WEIGHT_SHA256},
    )

    class FakeTokenizer:
        def __init__(self, _path):
            pass

        def tokenize(self, value):
            return [1, *([3] * len(value)), 2]

    class FakeProbe(torch.nn.Module):
        def load_pretrained(self, _weights):
            return self

        def forward(self, tokens):
            return torch.arange(512, dtype=torch.float32).repeat(tokens.shape[0], 1)

    monkeypatch.setattr(pretrained_inference, "PretrainedTokenizer", FakeTokenizer)
    monkeypatch.setattr(pretrained_inference, "PretrainedTransformerVAEEncoder", FakeProbe)
    output = tmp_path / "latent.csv"
    manifest = tmp_path / "manifest.json"
    pretrained_inference.run(smiles, mapping, weights, output, manifest, batch_size=2)

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["record_id"] for row in rows] == ["a", "b"]
    assert len(rows[0]) == 513
    assert float(rows[0]["mu_000"]) == 0.0
    assert float(rows[0]["mu_511"]) == 511.0
    record = json.loads(manifest.read_text(encoding="utf-8"))
    assert record["status"] == "PASS"
    assert record["records"] == 2
    assert record["latent_dimension"] == 512
    assert record["max_token_length_observed"] == 5
    assert record["output_sha256"] == _sha256(output)
    assert record["training_performed"] is False
    assert record["decoder_inference_performed"] is False


def test_run_rejects_251_tokens_without_creating_outputs(tmp_path: Path, monkeypatch) -> None:
    smiles, mapping, weights = _write_inputs(tmp_path)
    monkeypatch.setattr(pretrained_inference, "EXPECTED_TOKEN_MAPPING_SHA256", _sha256(mapping))
    monkeypatch.setattr(
        pretrained_inference,
        "EXPECTED_WEIGHT_SHA256",
        {name: _sha256(weights / name) for name in pretrained_inference.EXPECTED_WEIGHT_SHA256},
    )

    class TooLongTokenizer:
        def __init__(self, _path):
            pass

        def tokenize(self, _value):
            return list(range(251))

    monkeypatch.setattr(pretrained_inference, "PretrainedTokenizer", TooLongTokenizer)
    output = tmp_path / "latent.csv"
    manifest = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="250.*truncation is prohibited"):
        pretrained_inference.run(smiles, mapping, weights, output, manifest)
    assert not output.exists()
    assert not manifest.exists()
